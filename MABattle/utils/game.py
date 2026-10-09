import torch
from torch import nn
import random

from MABattle.MABattleV0.board import GameResults
from MABattle.utils.env import make_env


# from MABattle.MABattleV0.env import MABattleEnv
# from rl.buffer import ReplayBuffer, PERBuffer


def play_game(model, model_, env):
    agent_turn = random.choice((-1, 1))

    # models = {agent_turn: model, -agent_turn: model_}

    state, info = env.reset()
    global_reward, prev_state, logits, alive_agents = None, None, None, None
    prev_reward = None
    finished = False
    reward = None
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
            # unit_finished = finished
            # unit_local_dones = info["local_dones"] and unit_finished
            # opponent_alive_agents = info["alive_agents"]

            if prev_reward is not None:
                logs.append(
                    [
                        prev_state,
                        torch.tensor(prev_actions),
                        prev_reward,
                        pdv_finished,
                        pdv_state,#opponent_state,
                        torch.tensor(opponent_actions),
                        opponent_reward,
                        unit_finished,
                        unit_next_state
                    ]
                )
                #buffer.add_samples(
                #    prev_state,
                #    torch.tensor(prev_actions),
                #    prev_reward,
                #    opponent_finished,
                #    opponent_state,
                #    torch.tensor(opponent_actions),
                #    unit_finished,
                #    unit_next_state
                #)
                step_discounted_rewards += sum(reward)
                pdv_rewards += sum(reward)
                zero_rewards += sum(ur_reward) - sum(or_reward) * 0.9
                rewards += sum(ur_reward)

            prev_state = state
            prev_actions = actions
            prev_reward = reward
            pdv_finished = prev_finished
            #prev_alive_agents = alive_agents

            #opponent_finished = info["local_dones"] | finished
            pdv_state = info["pdv_state"]
            #opponent_state = next_state

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
            # opponent_finished = finished
            # unit_local_dones = info["local_dones"] and opponent_finished
            # unit_finished = info["local_dones"] or finished
            unit_finished = info["local_dones"] | finished

            state = unit_next_state
            global_reward = GameResults.Lose.value if info["is_win"] == GameResults.Win else GameResults.Draw.value


    return logs, global_reward, rewards, zero_rewards, step_discounted_rewards, pdv_rewards, c
    # return rewards


def test_play_game(model_state, model_cls, opponent, device):
    env = make_env()
    agent_turn = 1

    model = model_cls().eval()
    model.load_state_dict(model_state)
    #total = sum(p.sum().item() for p in model.parameters())
    #print(f"Total sum of parameters: {total}")

    models = {agent_turn: model, -agent_turn: opponent}

    state, info = env.reset()
    finished = False
    rewards = 0.0
    c = 0
    images = []

    while not finished:
        legals = info["legals"]

        # actions = models[env.unwrapped.to_play()].get_actions(state.to(device), legals.to(device))
        actions = models[env.unwrapped.to_play()].get_actions(state, legals)
        #print(env.unwrapped.to_play(), legals, actions)
        c += 1
        state, reward, finished, truncated, info = env.step(actions)
        finished = finished or truncated

        images.append(env.unwrapped.get_image())

        if env.unwrapped.to_play() == -agent_turn:
            rewards += sum(info["raw_reward"]) - 1
        else:
            rewards -= sum(info["raw_reward"])

    #images[0].save(f"output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    return rewards, images
