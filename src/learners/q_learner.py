import copy
from components.episode_buffer import EpisodeBatch
from functools import partial
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
import torch as th
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.use_coach = args.name == 'copa' or args.name == 'coid'
        self.copa = args.name == 'copa'
        self.coid = args.name == 'coid'
        if self.use_coach:
            self.comm_interval = self.args.centralized_every

        self.params = list(mac.parameters())
        if self.use_coach:
            self.params += list(self.mac.coach.parameters())
            if self.args.coach_vi:
                self.params += list(self.mac.copa_recog.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def _get_mixer_ins(self, batch, repeat_batch=1):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1),
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                     batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                    (entities[:, 1:],
                     batch["entity_mask"][:, 1:]))

    def _broadcast_decisions_to_batch(self, decisions, decision_pts):
        decision_pts = decision_pts.squeeze(-1)
        decision_pts_ = decision_pts.to(th.bool).clone()
        bs, ts = decision_pts.shape
        bcast_decisions = {k: th.zeros_like(v[[0]]).unsqueeze(0).repeat(bs * rep, ts, *(1 for _ in range(len(v.shape) - 1))) for k, (v, rep) in decisions.items()}
        for decname in bcast_decisions:
            value, rep = decisions[decname]
            bcast_decisions[decname][decision_pts_.repeat(rep, 1)] = value
        for t in range(1, ts):
            for decname in bcast_decisions:
                rep = decisions[decname][1]
                prev_value = bcast_decisions[decname][:, t - 1]
                bcast_decisions[decname][:, t] = ((decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1).to(prev_value.dtype) * bcast_decisions[decname][:, t])
                                                  + ((1 - decision_pts[:, t].repeat(rep).reshape(bs * rep, 1, 1)).to(prev_value.dtype) * prev_value))
        return bcast_decisions

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # episode over (not including timeout) - determines when to bootstrap
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        org_mask = mask.clone()
        avail_actions = batch["avail_actions"]

        will_log = (t_env - self.log_stats_t >= self.args.learner_log_interval)

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.target_mac.eval()
        if self.mixer is not None:
            self.mixer.train()
            self.target_mixer.eval()

        coach_h = None
        targ_coach_h = None
        coach_z = None
        targ_coach_z = None

        if self.use_coach:
            coach_h, infer_h = self.mac.coach.encode(batch, cal_infer=True)
            targ_coach_h, _ = self.target_mac.coach.encode(batch)
            decision_points = batch['hier_decision'].squeeze(-1)
            bs_rep = 1
            if 'imagine' in self.args.agent:
                bs_rep = 3
            decision_points = decision_points.to(th.bool)
            coach_h_t0 = coach_h[decision_points.repeat(bs_rep, 1)]
            targ_coach_h_t0 = targ_coach_h[decision_points]
            coach_z_t0, coach_mu_t0, coach_logvar_t0 = self.mac.coach.strategy(coach_h_t0)
            coach_mu_t0 = coach_mu_t0.chunk(bs_rep, dim=0)[0]
            coach_logvar_t0 = coach_logvar_t0.chunk(bs_rep, dim=0)[0]
            targ_coach_z_t0, _, _ = self.target_mac.coach.strategy(targ_coach_h_t0)

            bcast_ins = {
                'coach_z_t0': (coach_z_t0, bs_rep),
                'coach_mu_t0': (coach_mu_t0, 1),
                'coach_logvar_t0': (coach_logvar_t0, 1),
                'targ_coach_z_t0': (targ_coach_z_t0, 1),
            }
            bcast_decisions = self._broadcast_decisions_to_batch(bcast_ins, batch['hier_decision'])
            coach_z = bcast_decisions['coach_z_t0']
            coach_mu = bcast_decisions['coach_mu_t0']
            coach_logvar = bcast_decisions['coach_logvar_t0']
            targ_coach_z = bcast_decisions['targ_coach_z_t0']

        if 'imagine' in self.args.agent:
            all_mac_out, groups = self.mac.forward(batch, t=None, imagine=True,
                                                   coach_z=coach_z,
                                                   use_gt_factors=self.args.train_gt_factors,
                                                   use_rand_gt_factors=self.args.train_rand_gt_factors)
            # Pick the Q-Values for the actions taken by each agent
            rep_actions = actions.repeat(3, 1, 1, 1)
            all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

            mac_out, moW, moI = all_mac_out.chunk(3, dim=0)
            chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)
            caq_imagine = th.cat([caqW, caqI], dim=2)

            if will_log and self.args.test_gt_factors:
                gt_all_mac_out, gt_groups = self.mac.forward(batch, t=None, imagine=True, use_gt_factors=True)
                # Pick the Q-Values for the actions taken by each agent
                gt_all_chosen_action_qvals = th.gather(gt_all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim

                gt_mac_out, gt_moW, gt_moI = gt_all_mac_out.chunk(3, dim=0)
                gt_chosen_action_qvals, gt_caqW, gt_caqI = gt_all_chosen_action_qvals.chunk(3, dim=0)
                gt_caq_imagine = th.cat([gt_caqW, gt_caqI], dim=2)
        else:
            mac_out = self.mac.forward(batch, t=None, coach_z=coach_z)
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        self.target_mac.init_hidden(batch.batch_size)

        target_mac_out = self.target_mac.forward(batch, coach_z=targ_coach_z, t=None, target=True)
        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if 'imagine' in self.args.agent:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                chosen_action_qvals = self.mixer(chosen_action_qvals,
                                                 mix_ins)
                # don't need last timestep
                groups = [gr[:, :-1] for gr in groups]
                if will_log and self.args.test_gt_factors:
                    caq_imagine, ingroup_prop = self.mixer(
                        caq_imagine, mix_ins,
                        imagine_groups=groups,
                        ret_ingroup_prop=True)
                    gt_groups = [gr[:, :-1] for gr in gt_groups]
                    gt_caq_imagine, gt_ingroup_prop = self.mixer(
                        gt_caq_imagine, mix_ins,
                        imagine_groups=gt_groups,
                        ret_ingroup_prop=True)
                else:
                    caq_imagine = self.mixer(caq_imagine, mix_ins,
                                             imagine_groups=groups)
            else:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                chosen_action_qvals = self.mixer(chosen_action_qvals, mix_ins)
            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if 'imagine' in self.args.agent:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * loss + im_prop * im_loss

        if self.copa and self.args.coach_vi:
            # VI loss
            q_mu, q_logvar = self.mac.copa_recog(batch)
            q_t = D.normal.Normal(q_mu, (0.5 * q_logvar).exp())
            coach_z = coach_z.chunk(bs_rep, dim=0)[0]  # if combining with REFIL, only train full info Z
            log_prob = q_t.log_prob(coach_z).clamp_(-1000, 0).sum(-1)
            # entropy loss
            p_ = D.normal.Normal(coach_mu, (0.5 * coach_logvar).exp())
            entropy = p_.entropy().clamp_(0, 10).sum(-1)

            # mask inactive agents
            agent_mask = 1 - batch['entity_mask'][:, :, :self.args.n_agents].float()
            log_prob = (log_prob * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)
            entropy = (entropy * agent_mask).sum(-1) / (agent_mask.sum(-1) + 1e-8)

            vi_loss = (-log_prob[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()
            entropy_loss = (-entropy[:, :-1] * org_mask.squeeze(-1)).sum() / org_mask.sum()

            loss += vi_loss * self.args.vi_lambda + entropy_loss * self.args.vi_lambda / 10

        if self.coid and self.args.copa_influ_loss:
            infer_h_t0 = infer_h[decision_points.repeat(bs_rep, 1)]
            _, infer_mu_t0, infer_logvar_t0 = self.mac.coach.strategy(infer_h_t0)
            infer_mu_t0 = infer_mu_t0.chunk(bs_rep, dim=0)[0]
            infer_logvar_t0 = infer_logvar_t0.chunk(bs_rep, dim=0)[0]
            influence_loss, team_diversity, entropy = self.mac.coach.calcu_loss(coach_mu_t0, infer_mu_t0, coach_logvar_t0, infer_logvar_t0)

            loss += -(influence_loss+team_diversity) * self.args.vi_lambda - entropy * self.args.entropy_lambda / 10

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("losses/q_loss", loss.item(), t_env)
            if 'imagine' in self.args.agent:
                self.logger.log_stat("losses/im_loss", im_loss.item(), t_env)
            if self.copa and self.args.coach_vi:
                self.logger.log_stat("losses/copa_vi_loss", vi_loss.item(), t_env)
                self.logger.log_stat("losses/copa_entropy_loss", entropy_loss.item(), t_env)
            if self.coid and self.args.copa_influ_loss:
                self.logger.log_stat("losses/coid_influence_loss", influence_loss.item(), t_env)
                self.logger.log_stat("losses/coid_team_diversity", team_diversity.item(), t_env)
                self.logger.log_stat("losses/coid_entropy_loss", entropy.item(), t_env)
            if self.args.test_gt_factors:
                self.logger.log_stat("ingroup_prop", ingroup_prop.item(), t_env)
                self.logger.log_stat("gt_ingroup_prop", gt_ingroup_prop.item(), t_env)
            self.logger.log_stat("train_metrics/q_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("train_metrics/td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("train_metrics/q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("train_metrics/target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            if batch.max_seq_length == 2:
                # We are in a 1-step env. Calculate the max Q-Value for logging
                max_agent_qvals = mac_out_detach[:,0].max(dim=2, keepdim=True)[0]
                max_qtots = self.mixer(max_agent_qvals, batch["state"][:,0])
                self.logger.log_stat("max_qtot", max_qtots.mean().item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path, evaluate=False):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if not evaluate:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
