from copy import deepcopy
import torch

from agents.gradient.base import BaseGradientAgent
from net import MAFCVNoisyNet, MAFCVNet
from utils import avg


class DQN_PDV(BaseGradientAgent):
    def __init__(self, cfg, env_cfg):
        super().__init__(cfg, env_cfg, use_pdv_buffer=True)

        if self.cfg.noisy_critic:
            self.pdv_model = MAFCVNoisyNet(self.env_cfg.flatten_state_shape, sigma_init=self.cfg.sigma_init)
            self.pdv_model_twin = MAFCVNoisyNet(self.env_cfg.flatten_state_shape, sigma_init=self.cfg.sigma_init)
        else:
            self.pdv_model = MAFCVNet(self.env_cfg.flatten_state_shape)
            self.pdv_model_twin = MAFCVNet(self.env_cfg.flatten_state_shape)

        self.pdv_model_target = deepcopy(self.pdv_model)
        self.pdv_model_target_twin = deepcopy(self.pdv_model_twin)

        self.pdv_optimizer = self.optimizer_cls(self.pdv_model.parameters(), lr=self.cfg.pdv_lr)
        self.pdv_optimizer_twin = self.optimizer_cls(self.pdv_model_twin.parameters(), lr=self.cfg.pdv_lr)

        self.gamma_2 = self.cfg.gamma ** 2

    def _update_single_network(self, agent_id, model, model_optimizer, model_target, pdv_model, pdv_optimizer,
                               pdv_model_other) -> tuple:
        ids, states, actions, rewards, opponent_finishes, opponent_next_states, opponent_actions, pdv_rewards, \
            unit_finishes, unit_next_states = self.buffer.sample(agent_id)

        q_values = torch.gather(model(states), dim=-1, index=actions.unsqueeze(-1)).squeeze()
        next_pdv_values = pdv_model_other(opponent_next_states).squeeze().detach()

        q_targets = rewards + self.cfg.gamma*(1-opponent_finishes)*next_pdv_values

        if self.cfg.multi_agent_mode == "Independent":
            loss = self.loss_fn(q_values, q_targets)
        elif self.cfg.multi_agent_mode == "Decomposition":
            loss = self.loss_fn(q_values.sum(-1), q_targets.sum(-1))
        else:
            raise ValueError()

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        pdv_values = pdv_model(opponent_next_states).squeeze()

        next_actions = model(unit_next_states).argmax(-1)
        next_q_values = torch.gather(model_target(unit_next_states), dim=-1, index=next_actions.unsqueeze(-1)).squeeze().detach()
        pdv_targets = pdv_rewards + self.cfg.gamma*(1-unit_finishes)*next_q_values

        td_errors = (q_targets - q_values).sum(-1).abs().detach().cpu()
        pdv_loss = self.loss_fn(pdv_values, pdv_targets)

        pdv_optimizer.zero_grad()
        pdv_loss.backward()
        pdv_optimizer.step()

        self.buffer.set_weights(1, ids, td_errors)

        return loss.item(), pdv_loss.item()

    def update_network(self) -> dict:
        loss, loss_pdv = map(avg, [
            self._update_single_network(1, self.model, self.model_optimizer, self.model_target,
                                        self.pdv_model, self.pdv_optimizer, self.pdv_model_target_twin),
            self._update_single_network(2, self.model_twin, self.model_optimizer_twin, self.model_target_twin,
                                        self.pdv_model_twin, self.pdv_optimizer_twin, self.pdv_model_target),
        ])

        return {"gradient/loss": loss, "gradient/loss_pdv": loss_pdv}
