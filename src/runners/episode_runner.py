from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.use_coach = args.name == 'copa' or args.name == 'coid'
        self.copa = args.name == 'copa'
        self.coid = args.name == 'coid'
        if self.use_coach:
            self.comm_interval = self.args.centralized_every
        assert self.batch_size == 1

        if ('sc2' in self.args.env) or ('group_matching' in self.args.env):
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info(self.args)

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, test=False, index=None):
        self.batch = self.new_batch()
        self.env.reset(test=test, index=index)
        self.t = 0

    def _get_pre_transition_data(self, pre_chosen=None, terminal=False):
        if self.args.entity_scheme:
            masks = self.env.get_masks()
            if len(masks) == 2:
                obs_mask, entity_mask = masks
                gt_mask = None
            else:
                obs_mask, entity_mask, gt_mask = masks
            pre_transition_data = {
                "entities": [self.env.get_entities()],
                "obs_mask": [obs_mask],
                "entity_mask": [entity_mask],
                "avail_actions": [self.env.get_avail_actions()]
            }
            if gt_mask is not None:
                pre_transition_data["gt_mask"] = gt_mask
        else:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

        if self.use_coach:
            if self.t == 0:
                pre_transition_data['hier_decision'] = [(1,)]
            else:
                if self.hier_timer >= self.comm_interval and not terminal:
                    self.hier_timer = 0
                    pre_transition_data['hier_decision'] = [(1,)]
                else:
                    pre_transition_data['hier_decision'] = [(0,)]
            if self.coid:
                if pre_chosen is None:
                    pre_transition_data.update(
                        {'pre_actions': th.zeros(1, self.args.n_agents, self.args.n_actions).to(self.args.device)})
                    pre_transition_data.update(
                        {'pre_role': th.zeros(1, self.args.n_agents, self.args.rnn_hidden_dim).to(self.args.device)})
                else:
                    pre_transition_data.update(pre_chosen)

        return pre_transition_data

    def run(self, test_mode=False, test_scen=None, index=None, vid_writer=None):
        """
        test_mode: whether to use greedy action selection or sample actions
        test_scen: whether to run on test scenarios. defaults to matching test_mode.
        vid_writer: imageio video writer object
        """
        if test_scen is None:
            test_scen = test_mode
        self.reset(test=test_scen, index=index)
        if vid_writer is not None:
            vid_writer.append_data(self.env.render())
        terminated = False
        self.hier_timer = 0
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        # make sure things like dropout are disabled
        self.mac.eval()

        pre_chosen = None
        while not terminated:
            pre_transition_data = self._get_pre_transition_data(pre_chosen)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.use_coach:
                actions, z_team, action_outs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                                       test_mode=test_mode)
                if self.coid:
                    pre_chosen = {
                        "pre_actions": action_outs,
                        "pre_role": z_team
                    }
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0].cpu())
            if vid_writer is not None:
                vid_writer.append_data(self.env.render())
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            self.hier_timer += 1

        last_data = self._get_pre_transition_data(pre_chosen, terminal=True)
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.use_coach:
            actions, _, _ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
