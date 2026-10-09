import torch
import random

from MABattle.MABattleV0.board import GameResults
from MABattle.utils.env import make_env


def play_game(model, model_, env):
    agent_turn = random.choice((-1, 1))

    state, info = env.reset()
    global_reward, prev_state, logits, alive_agents = None, None, None, None
    prev_reward = None
    finished = False
    step_discounted_rewards = 0.0
    rewards = 0.0
    pdv_rewards = 0.0
    zero_rewards = 0.0
    c = 0
    logs = []
    while not finished:
        legals = info["legals"]

        if env.unwrapped.to_play() == agent_turn:
            prev_finished = info["local_dones"] | finished

            actions = model.get_actions(state.to(model.device), legals)
            next_state, reward, finished, truncated, info = env.step(actions)
            ur_reward = info["raw_reward"]
            finished = finished or truncated

            if prev_reward is not None:
                logs.append(
                    [
                        prev_state,
                        torch.tensor(prev_actions),
                        prev_reward,
                        pdv_finished,
                        pdv_state,
                        torch.tensor(opponent_actions),
                        opponent_reward,
                        unit_finished,
                        unit_next_state
                    ]
                )

                step_discounted_rewards += sum(reward)
                pdv_rewards += sum(reward)
                zero_rewards += sum(ur_reward) - sum(or_reward) * 0.9
                rewards += sum(ur_reward)

            prev_state = state
            prev_actions = actions
            prev_reward = reward
            pdv_finished = prev_finished
            pdv_state = info["pdv_state"]

            state = next_state
            c += 1
            global_reward = info["is_win"].value

        else:
            opponent_actions = model_.get_actions(state, legals)
            unit_next_state, _, finished, truncated, info = env.step(opponent_actions)

            pdv_rewards += sum(info["opponent_reward"])
            opponent_reward = info["opponent_reward"]
            or_reward = info["raw_reward"]
            finished = finished or truncated
            unit_finished = info["local_dones"] | finished

            state = unit_next_state
            global_reward = GameResults.Lose.value if info["is_win"] == GameResults.Win else GameResults.Draw.value


    return logs, global_reward, rewards, zero_rewards, step_discounted_rewards, pdv_rewards, c


def test_play_game(env_cfg, model_state, model_cls, opponent):
    env = make_env(env_cfg)
    agent_turn = 1

    model = model_cls(env_cfg.flatten_state_shape, env_cfg.num_agents, env_cfg.action_space).eval()
    model.load_state_dict(model_state)

    models = {agent_turn: model, -agent_turn: opponent}

    state, info = env.reset()
    finished = False
    rewards = 0.0
    c = 0
    images = []

    while not finished:
        legals = info["legals"]

        actions = models[env.unwrapped.to_play()].get_actions(state, legals)
        c += 1
        state, reward, finished, truncated, info = env.step(actions)
        finished = finished or truncated

        images.append(env.unwrapped.get_image())

        if env.unwrapped.to_play() == -agent_turn:
            rewards += sum(info["raw_reward"]) - 1
        else:
            rewards -= sum(info["raw_reward"])

    return rewards, images
