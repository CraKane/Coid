import copy
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from modules.layers.attention import EntityAttentionLayer, ScaledDotProductAttention


class Coach(nn.Module):
    def __init__(self, args):
        super(Coach, self).__init__()
        self.args = args
        de = args.entity_shape
        dh = args.attn_embed_dim
        od = args.rnn_hidden_dim
        self.influ_lambda = args.influ_lambda
        self.dh = dh
        self.na = args.n_actions
        self.device = args.device
        self.n_agents = args.n_agents
        self.infer_emb = torch.randn([1, args.n_agents - 1, self.dh]).to(self.device)

        self.agent_emb = nn.Linear(dh * 2, dh)
        self.action_embedding = nn.Linear(self.na, dh)
        self.influence_embedding = nn.Linear(dh * 2 + od, dh)
        self.influence_attention = ScaledDotProductAttention(d_model=dh,
                                                             d_k=dh,
                                                             d_v=dh,
                                                             h=args.n_heads)
        self.attention_layer_norm = nn.LayerNorm([1, dh])

        # policy for continouos team strategy
        self.mean = nn.Linear(dh * 2, od)
        self.logvar = nn.Linear(dh * 2, od)

        self.entity_encode = nn.Linear(de, dh)
        self.mha = EntityAttentionLayer(dh, dh, dh, args)

    def _logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def encode(self, batch, cal_infer=False):
        a = batch['pre_actions']
        pre_z_team = batch['pre_role']
        infer_encode = None
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        obs_mask = batch['obs_mask']
        restore_ts = False
        if len(entities.shape) == 4:
            restore_ts = True
            bt, ts = entities.shape[:2]
            entities = entities.flatten(0, 1)
            entity_mask = entity_mask.flatten(0, 1)
            obs_mask = obs_mask.flatten(0, 1)
            a = a.flatten(0, 1)
            pre_z_team = pre_z_team.flatten(0, 1)

        he = self.entity_encode(entities)
        bs, _, dh = he.shape
        h_state_team = self.mha(F.relu(he), pre_mask=obs_mask,
                                post_mask=entity_mask[:, :self.args.n_agents])
        a_e_x = torch.cat([h_state_team, he[:, :self.args.n_agents]], -1)  # [batch, n_agents, dh*2]
        a_e = self.agent_emb(a_e_x)  # [batch, n_agents, dh]

        pre_actions = self.action_embedding(a)  # [batch, n_agents, dh]
        i_e_x = torch.cat([he[:, :self.args.n_agents], pre_actions, pre_z_team], -1)  # [batch, n_agents, dh+na+od]
        i_encode = self.influence_embedding(i_e_x)  # [batch, n_agents, dh]

        actor_attention = []
        if cal_infer:
            infer_attention = []
        for i in range(self.args.n_agents):
            idx_tmp = torch.zeros_like(i_encode)
            idx_tmp[:, i, :] = 1
            idx = idx_tmp >= 1
            actor_emb = a_e[idx].reshape(bs, -1, dh)
            influ_emb = i_encode[~idx].reshape(bs, -1, dh)
            actor_att = self.influence_attention(actor_emb, influ_emb, influ_emb)
            actor_norm = self.attention_layer_norm(actor_att)
            actor_attention.append(actor_norm)
            if cal_infer:
                infer_emb = self.infer_emb.repeat(bs, 1, 1)
                infer_att = self.influence_attention(actor_emb, infer_emb, infer_emb)
                infer_norm = self.attention_layer_norm(infer_att)
                infer_attention.append(infer_norm)

        i_e = torch.stack(actor_attention, 1).squeeze(2)  # [batch, n_agents, dh]
        if cal_infer:
            infer_e = torch.stack(infer_attention, 1).squeeze()  # [batch, n_agents, dh]

        x_all_emb = torch.cat([a_e, i_e], -1)  # [batch, n_agents, dh*2]
        if cal_infer:
            infer_encode = torch.cat([a_e, infer_e], -1)  # [batch, n_agents, dh*2]

        if restore_ts:
            x_all_emb = x_all_emb.reshape(bt, ts, self.args.n_agents, -1)
            if cal_infer:
                infer_encode = infer_encode.reshape(bt, ts, self.args.n_agents, -1)
        return x_all_emb, infer_encode

    def strategy(self, h):
        # h = F.relu(h)
        mu, logvar = self.mean(h), self.logvar(h)
        logvar = logvar.clamp_(-10, 0)
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, batch):
        h, _ = self.encode(batch)  # (o, e, c, ms, pre_actions) # [batch, n_agents, dh*2]
        z_team, mu, logvar = self.strategy(h)  # [batch, n_agents, dh]
        return z_team, mu, logvar

    def calcu_loss(self, coach_mu, infer_mu, coach_logvar, infer_logvar):
        p_ = D.normal.Normal(coach_mu, (0.5 * coach_logvar).exp())
        p_infer = D.normal.Normal(infer_mu, (0.5 * infer_logvar).exp())

        ################# --------------- Influence Role Reward ----------------- ##################
        # print("influence reward: ")
        influence_loss = torch.tanh(self.influ_lambda * kl_divergence(p_, p_infer).mean())
        # print(influence_loss)

        ################# --------------- Team Role Diversity ------------------- ##################
        role_simi_matrix = torch.ones((self.n_agents, self.n_agents)).to(self.device)
        for i in range(self.n_agents - 1):
            for j in range(i + 1, self.n_agents):
                mu_i, scale_i = p_.loc[:, i], p_.scale[:, i]
                mu_j, scale_j = p_.loc[:, j], p_.scale[:, j]
                p_i = D.normal.Normal(mu_i, scale_i)
                p_j = D.normal.Normal(mu_j, scale_j)
                dis = kl_divergence(p_i, p_j).mean() + kl_divergence(p_j, p_i).mean()

                role_simi_matrix[i, j] = role_simi_matrix[j, i] = torch.exp(-dis).clone()

        # print(role_simi_matrix)
        team_role_diversity = torch.det(role_simi_matrix)
        # print(team_role_diversity)

        entropy = p_.entropy().clamp_(0, 10).sum(-1).mean()
        return influence_loss, team_role_diversity, entropy
