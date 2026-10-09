import os
import time
import datetime
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter

import const
from const import device_name
from net import MADeterministicFCNet, MAStochasticFCNet, MAFCQNet, MAFCQNoisyNet
from evolution import Evolution, ma_battle_fit, ma_battle_fit_random, ma_battle_fit_best
from utils.utils import save_json

archs = {
    "MAStochasticFCNet": MAStochasticFCNet,
    "MADeterministicFCNet": MADeterministicFCNet
}
shutdown = True


def run():
    def callback(evolution: Evolution):
        epoch = evolution.epoch
        for metric in evolution.metrics.items():
            writer.add_scalar(*metric, epoch)

        if epoch % 5 == 0:
            state = {'info': "NNE-V1",  # описание
                     'date': datetime.datetime.now(),  # дата и время
                     'epochs': epoch,
                     'model': evolution.best,
                     }
            str_dir = os.path.join(root_dir, f'Models/evolution/{name}.pt')
            torch.save(state, str_dir)


            state = {'info': "NNE-V1(gradient)",  # описание
                     'date': datetime.datetime.now(),  # дата и время
                     'epochs': epoch,
                     'model': evolve.gradient_alg.model.state_dict(),
                     }
            str_dir = os.path.join(root_dir, f'Models/gradient/{name}_gradient.pt')
            torch.save(state, str_dir)

    try:
        name = f"nne-random-BOARD8"
        epochs = 300
        root_dir = f"logs/{str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))}"

        print(root_dir)
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        model_dir = os.path.join(root_dir, "Models")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        evolution_model_dir = os.path.join(root_dir, "Models/evolution")
        if not os.path.isdir(evolution_model_dir):
            os.makedirs(evolution_model_dir)

        gradient_model_dir = os.path.join(root_dir, "Models/gradient")
        if not os.path.isdir(gradient_model_dir):
            os.makedirs(gradient_model_dir)

        writer = SummaryWriter(os.path.join(root_dir, f"learning-{name}"), comment="-" + name,
                               flush_secs=120)

        device = torch.device(device_name)
        evolve = Evolution(
            epochs=epochs,
            model_cls=MAFCQNoisyNet,
            fitness_fn=ma_battle_fit_random,
            population_size=128,
            device=device,
            callback_fn=callback
        )

        save_json(asdict(evolve.cfg), os.path.join(root_dir, "evolution_config.json"))
        save_json(asdict(evolve.gradient_alg.cfg), os.path.join(root_dir, "gradient_config.json"))

        evolve.train(epochs)

        state = {'info': "NNE-V1",  # описание
                 'date': datetime.datetime.now(),  # дата и время
                 'epochs': evolve.epoch,
                 'model': evolve.best,
                 }
        str_dir = os.path.join(root_dir, f'Models/evolution/{name}.pt')
        torch.save(state, str_dir)

        state = {'info': "NNE-V1(gradient)",  # описание
                 'date': datetime.datetime.now(),  # дата и время
                 'epochs': evolve.epoch,
                 'model': evolve.gradient_alg.model.state_dict(),
                 }
        str_dir = os.path.join(root_dir, f'Models/gradient/{name}_gradient.pt')
        torch.save(state, str_dir)

    finally:
        evolve.gradient_alg.env.close()
        writer.close()

        del evolve.gradient_alg.buffer
        del evolve.gradient_alg
        del evolve

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        import gc
        gc.collect()




if __name__ == "__main__":
    run()

    if shutdown:
        time.sleep(5)
        os.system("shutdown /s /f /t 0")
