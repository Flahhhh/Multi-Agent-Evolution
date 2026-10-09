# MultiAgentEvolution

This project contains implementations of multi-agent gradient and evolutionary algorithms, along with a chess-like competitive environment. It provides a comprehensive framework for researching and comparing different multi-agent reinforcement learning (MARL) methods, including independent learning, value decomposition, and hybrid evolution-gradient approaches.

The environment is a turn-based board game where two opposing teams of units move on a grid. Agents receive local observations and must learn to coordinate or compete to maximize their reward.

## Results

The following plot shows the performance comparison of several algorithms trained on the environment. The results illustrate the learning dynamics and final convergence of each method.

![Training Results](results/result.png)

The plot above compares the mean reward (and its standard deviation) for different algorithms:
- **DQN (Independent & Decomposition)**: Standard DQN with either independent Q-learning or a VDN-style decomposition.
- **DQN-PDV**: DQN augmented with a post-decision value (PDV) network, which helps in dealing with multi-agent stochasticity.
- **MERL**: A hybrid approach that combines evolutionary optimization with gradient-based updates.

## Project Structure

- **`agents/`**: Contains all agent implementations.
  - **`gradient/`**: Gradient-based algorithms:
    - `DQN`: Deep Q-Network with two multi-agent variants:
      - **Independent**: each agent maximizes its own reward.
      - **Decomposition (VDN)**: a joint action-value function is decomposed into individual utilities.
    - `DQN_PDV`: DQN with a post-decision state value network, which captures the expected value after the environment's transition but before the opponent's move.
  - **`evolution/`**: Evolutionary algorithms:
    - **`InGA`**: Independent Genetic Algorithm. Each individual is a complete neural network policy. Fitness is evaluated through self-play or against a random opponent.
    - **`MERL`**: Multi-agent Evolutionary Reinforcement Learning. Combines a gradient-based agent (DQN or DQN-PDV) with an evolutionary population. The gradient agent's experience is used to guide the evolutionary search, and the population contributes to the replay buffer.

- **`MABattle/`**: The multi-agent environment.
  - Chess-like grid with two players (teams) taking turns.
  - Supports both competitive and coordination tasks.
  - Includes a visualization module for rendering game states.
  - Configuration via `EnvCfg` dataclass.

- **`utils/`**: Helper functions:
  - Fitness evaluation functions for evolutionary algorithms.
  - Rule-based opponent agents (e.g., `RandomAgent`).
  - Logging and data processing utilities.

- **`net.py`**: Neural network architectures:
  - `MAFCQNet`: Standard Dueling Q-network.
  - `MAFCQNoisyNet`: Dueling Q-network with NoisyLinear layers for exploration.
  - `MAFCVNet` & `MAFCVNoisyNet`: Value networks used for post-decision states.
  - `MAPolicyNet` (optional).

- **`buffer.py`**: Replay buffer implementations with prioritized experience replay (PER). Supports standard and PDV-specific storage.

- **`test.py`**: Script to load a trained model and evaluate it against a random opponent, generating a GIF of the gameplay.

- **`test_var.py`**: Script to compute statistical metrics (mean, variance, standard deviation) across multiple training runs and generate performance plots.

- **`make_graphic.py`**: Script to produce comparison plots from saved JSON results.

- **`main2.py`**: Entry point for training an agent. An example using `InGA` is provided.

## Usage

### Installation

Install the required dependencies using pip:

bash
pip install -r requirements.txt
Training
To train a specific agent, modify the configuration in main2.py and run:

bash
python main2.py
You can switch between agents by importing the desired class and its configuration:

DQN, DQNCfg

DQN_PDV, DQN_PDVCfg

InGA, InGACfg

MERL, MERLCfg

Example for training DQN_PDV:

python
from agents import DQN_PDV, DQN_PDVCfg
from MABattle.MABattleV0 import EnvCfg

if __name__ == "__main__":
    agent_cfg = DQN_PDVCfg()
    env_cfg = EnvCfg()
    agent = DQN_PDV(agent_cfg, env_cfg)
    agent.train(300)
All training logs, models, and configuration files are saved in the logs/ directory with a timestamp.

Testing a Trained Model
After training, you can test a model using test.py. Update the path to the saved model checkpoint and run:

bash
python test.py
This script will:

Load the model weights.

Run 10,000 test episodes in parallel using multiprocessing.

Generate a GIF (output.gif) showing a representative game (median reward).

To customize the test, modify the state_dict path and the num_tests variable.

Generating Performance Plots
To generate a comparison plot of multiple algorithms, use make_graphic.py. It reads JSON files from the results/ directory and plots the mean reward (or other metrics) over training episodes.

You can adjust the results dictionary to point to your own experiment directories.

For detailed statistical analysis (mean, variance, standard deviation across runs), use test_var.py. It aggregates results from multiple training runs and produces a multi-panel figure showing both environment and algorithm variability.

Configuration
All agent configurations are dataclasses located in the respective config.py files. Key parameters include:

multi_agent_mode: "Independent" or "Decomposition".

opponent_type: "Random", "Greedy", or "Self-play".

loss: "HuberLoss" or "MSELoss".

lr: Learning rate for gradient-based agents.

buffer_size, batch_size, init_alpha (for PER).

gamma: Discount factor.

tau: Target network update rate.

sigma_init: Initial noise standard deviation for NoisyLinear layers.

For evolutionary agents:

population_size, elite_amount, num_relatives.

init_noise_scale, min_noise_scale.

fitness_type: "Random", "Randomizer", or "Self-play".

Visualization
The environment includes a renderer that can save game frames as images. The test.py script combines these frames into an animated GIF (output_new.gif) for qualitative inspection of the learned policy.

Extending the Framework
To add a new agent:

Subclass BaseAgent or BaseGradientAgent/BaseEvolutionAgent.

Implement the train_epoch and callback methods.

Define a configuration dataclass.

Register the agent in agents/__init__.py.

To modify the environment:

Adjust EnvCfg parameters (board size, number of units, action set, rewards).

Extend the Board class to change game rules or unit dynamics.

Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs or feature requests.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or feedback, please contact the project maintainer.