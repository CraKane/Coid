"""
Adapted from https://github.com/Cranial-XIX/marl-copa/blob/master/modules/coach.py
"""
from numpy import imag
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.attention import EntityAttentionLayer

class Coach(nn.Module):
    def __init__(self, args):
        super(Coach, self).__init__()
        self.args = args
        de = args.entity_shape
        dh = args.attn_embed_dim
        od = args.rnn_hidden_dim

        self.entity_encode = nn.Linear(de, dh)
        self.mha = EntityAttentionLayer(dh, dh, dh, args)

        # policy for continuous team strategy
        self.mean = nn.Linear(dh, od)
        self.logvar = nn.Linear(dh, od)

    def _logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def encode(self, batch, cal_infer=False):
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        obs_mask = batch['obs_mask']
        restore_ts = False
        if len(entities.shape) == 4:
            restore_ts = True
            bs, ts = entities.shape[:2]
            entities = entities.flatten(0, 1)
            entity_mask = entity_mask.flatten(0, 1)
            obs_mask = obs_mask.flatten(0, 1)

        he = self.entity_encode(entities)
        hidden = self.mha(F.relu(he), pre_mask=obs_mask,
                          post_mask=entity_mask[:, :self.args.n_agents])
        if restore_ts:
            hidden = hidden.reshape(bs, ts, self.args.n_agents, -1)
        return hidden, None

    def strategy(self, h):
        # h = F.relu(h)
        mu, logvar = self.mean(h), self.logvar(h)
        logvar = logvar.clamp_(-10, 0)
        std = (logvar * 0.5).exp()
        eps = th.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, batch):
        h, _ = self.encode(batch)
        z_team, mu, logvar = self.strategy(h)
        return z_team, mu, logvar

class RecognitionModel(nn.Module):
    # I(z^a ; s^a_t+1:t+T-1 | s_t)
    def __init__(self, args):
        super(RecognitionModel, self).__init__()
        self.args = args
        de = args.entity_shape
        dh = args.attn_embed_dim
        od = args.rnn_hidden_dim

        na = args.n_actions
        self.action_embedding = nn.Linear(na, dh, bias=False)
        self.entity_encode = nn.Linear(de, dh)

        self.mha = EntityAttentionLayer(dh, dh, dh, args)

        self.mean = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, od))
        self.logvar = nn.Sequential(
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, od))

    def forward(self, batch):
        entities = batch['entities']
        entity_mask = batch['entity_mask']
        obs_mask = batch['obs_mask']
        actions = batch['actions_onehot']
        restore_ts = False
        if len(entities.shape) == 4:
            restore_ts = True
            bs, ts = entities.shape[:2]
            entities = entities.flatten(0, 1)
            entity_mask = entity_mask.flatten(0, 1)
            actions = actions.flatten(0, 1)
            obs_mask = obs_mask.flatten(0, 1)

        he = self.entity_encode(entities)
        ha = self.action_embedding(actions)
        ha_pad = th.zeros_like(he)
        ha_pad[:, :self.args.n_agents] = ha
        hidden = self.mha(F.relu(he + ha_pad), pre_mask=obs_mask,
                          post_mask=entity_mask[:, :self.args.n_agents])
        if restore_ts:
            hidden = hidden.reshape(bs, ts, self.args.n_agents, -1)
        hidden = F.relu(hidden)
        mean = self.mean(hidden)
        logvar = self.logvar(hidden).clamp(-10, 0)

        return mean, logvar