from .basic_controller import BasicMAC
import torch as th


# This multi-agent controller shares parameters between agents and takes
# entities + observation masks as input
class EntityMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(EntityMAC, self).__init__(scheme, groups, args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with entity + observation mask inputs.
        bs = batch.batch_size
        entities = []
        entities.append(batch["entities"][:, t])  # bs, ts, n_entities, vshape
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entities.append(ent_acs)
        entities = th.cat(entities, dim=3)
        if self.args.gt_mask_avail:
            return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t], batch["gt_mask"][:, t])
        return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t])

    def _get_input_shape(self, scheme):
        input_shape = scheme["entities"]["vshape"]
        if self.args.entity_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape

    def init_hidden(self, batch_size):
        super().init_hidden(batch_size)
        if self.use_coach:
            self.coach_z = th.zeros(
                batch_size, self.n_agents, self.args.rnn_hidden_dim, device=self.device
            )

    def save_models(self, path):
        super().save_models(path)
        if self.use_coach:
            if self.coid:
                th.save(self.coach.state_dict(), "{}coid_coach.th".format(path))
            elif self.copa:
                th.save(self.coach.state_dict(), "{}copa_coach.th".format(path))
                if self.args.coach_vi:
                    th.save(self.copa_recog.state_dict(), "{}copa_recog.th".format(path))

    def load_models(self, path, pi_only=False):
        super().load_models(path)
        if pi_only:
            return
        if self.use_coach:
            if self.coid:
                self.coach.load_state_dict(
                    th.load("{}coid_coach.th".format(path), map_location=lambda storage, loc: storage))
            elif self.copa:
                self.coach.load_state_dict(th.load("{}copa_coach.th".format(path), map_location=lambda storage, loc: storage))
                if self.args.coach_vi:
                    self.copa_recog.load_state_dict(th.load("{}copa_recog.th".format(path), map_location=lambda storage, loc: storage))

    def cuda(self):
        super().cuda()
        if self.use_coach:
            self.coach.cuda()
            if self.args.coach_vi:
                self.copa_recog.cuda()

    def eval(self):
        super().eval()
        if self.use_coach:
            self.coach.eval()
            if self.args.coach_vi:
                self.copa_recog.eval()

    def train(self):
        super().train()
        if self.use_coach:
            self.coach.train()
            if self.args.coach_vi:
                self.copa_recog.train()

    def load_state(self, other_mac):
        super().load_state(other_mac)
        if self.use_coach:
            self.coach.load_state_dict(other_mac.coach.state_dict())
            if self.args.coach_vi:
                self.copa_recog.load_state_dict(other_mac.copa_recog.state_dict())

    def _make_meta_batch(self, ep_batch, t_ep):
        # Add quantities necessary for meta-controller (only used for acting as
        # this doesn't compute rewards/term, etc.)
        decision_pts = ep_batch['hier_decision'][:, t_ep].flatten()
        d_inds = (decision_pts == 1)
        meta_batch = {
            'entities': ep_batch['entities'][d_inds, t_ep],
            'obs_mask': ep_batch['obs_mask'][d_inds, t_ep],
            'entity_mask': ep_batch['entity_mask'][d_inds, t_ep],
            'avail_actions': ep_batch['avail_actions'][d_inds, t_ep],
        }
        if self.coid:
            meta_batch.update({
                'pre_actions': ep_batch['pre_actions'][d_inds, t_ep],
                'pre_role': ep_batch['pre_role'][d_inds, t_ep],
            })
        return meta_batch