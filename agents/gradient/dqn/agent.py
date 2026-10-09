import torch

from agents.gradient.base import BaseGradientAgent
from utils import avg


class DQN(BaseGradientAgent):
    def _update_single_network(self, agent_id, model, model_optimizer, model_target) -> tuple:
        ids, states, actions, rewards, pre_finishes, post_rewards, finishes, next_states = self.buffer.sample(agent_id)

        next_actions = model(states).argmax(-1)
        next_q_values = torch.gather(model_target(states), index=next_actions.unsqueeze(-1), dim=-1).squeeze()

        q_values = torch.gather(model(states), index=actions.unsqueeze(-1), dim=-1).squeeze()
        q_targets = rewards + self.gamma * (1 - pre_finishes) * post_rewards + self.gamma_2 * (
                    1 - finishes) * next_q_values.detach()

        if self.cfg.multi_agent_mode == "Independent":
            loss = self.loss_fn(q_values, q_targets)
        elif self.cfg.multi_agent_mode == "Decomposition":
            loss = self.loss_fn(q_values.sum(-1), q_targets.sum(-1))
        else:
            raise ValueError()

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        td_errors = (q_targets-q_values).sum(-1).abs().detach().cpu()
        self.buffer.set_weights(agent_id, ids, td_errors)

        return loss.item()

    def update_network(self) -> dict:
        loss = avg([
            self._update_single_network(1, self.model, self.model_optimizer, self.model_target_twin),
            self._update_single_network(2, self.model_twin, self.model_optimizer_twin, self.model_target)
        ])

        return {"gradient/loss": loss}
