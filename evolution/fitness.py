import random

from .merl_evolution import Evolution
from MABattle.utils.env import make_env

from MABattle.utils.game import play_game
from utils.agent import RandomAgent


def ma_battle_fit(solution, solution_, model_cls):
    env = make_env()
    model, model_ = model_cls(), model_cls()#to(device)

    #model_ = random.choice(evolution.population).to(device)
    logs, fit, _, _, _, _, _ = play_game(model, model_, env)
    env.close()

    return logs, fit
def ma_battle_fit_random(solution, _, model_cls):
    env = make_env()

    model, model_ = model_cls(), RandomAgent()

    #print(type(solution), type(model_cls))
    model.load_state_dict(solution)
    #model.load_state_dict(solution)

    #model, model_ = model.cpu(), RandomAgent()

    logs, fit, _, _, _, _, _ = play_game(model, model_, env)
    env.close()

    return logs, fit

random_percent=0.25
def ma_battle_fit_best(solution, best, model_cls):
    env = make_env()

    model = model_cls()
    model.load_state_dict(solution)

    if isinstance(best, RandomAgent):
        model_ = best
    else:
        model_ = model_cls()
        model_.load_state_dict(best)
    #random_agent = RandomAgent()
    #opponents = [random_agent] * int(num_games * random_percent) + [model_] * (num_games - int(num_games * random_percent))

    logs, fit, _, _, _, _, _ = play_game(model, model_, env)
    env.close()

    return logs, fit
