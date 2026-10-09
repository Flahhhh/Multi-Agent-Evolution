# MultiAgentEvolution
This project contains implementations of multi-agent gradient and evolutionary algorithms and chess-like competitive environment for it.
There also has testing scripts
Here is the result of training:
![output_new.gif](../nne_v2%28QZero%29%20%E2%80%94%20%D0%BA%D0%BE%D0%BF%D0%B8%D1%8F/output_new.gif)

# Results


# Project structure
- agents: realizations of DQN, InGA, MERL and DQN with post-decision values. There also you can find abstract classes and configurations of every type of agent in this project
    - agents/gradient contains realizations of all gradient algorithms: DQN(Deep Q Networks) and DQN_PDV(Deep Q Networks with post-decision values)
      - There also two multi-agent variations of DQN: 
      - IDQN(Independent Deep Q Network), that maximizes local reward of every agent, there is no problem with credit assigment, but it can has problems with coordination
      - DQN with decomposition(VDN style). This unites all single actions to join action. This method simular with VDN algorithm:\(Q_{tot}(\tau ,\mathbf{u})=\sum _{i=1}^{n}Q_{i}(z_{i},u_{i})\)
    - agents/evolution contains realizations of all evolutionary algorithms: InGA(Independent Genetic Algorithm) and MERL(Multi-agent Evolutionary Reinforcement Learning)
        - MERL algorithm can be used with any version of DQN or DQN_PDV
    
- MABattle: chess-like multi-agent gym environment. 
  - It has two multi-agent levels, that includes competition(two players, that makes turns) and coordination(every unit on the desk performs a single action to maximize total reward)
  - This env also has visualization that you can see when using test.py script

- images directory contains all png images for the environment
- in utils are located all fitness functions, rule-based agents for this env and other useful functions
- net.py contains all networks, including agent and post-decision value-network. There also has noisy variations of this networks

# Usage
To install all dependencies:
    pip install -r requirements.txt

To run training:
    python3 main.py

To test model:
    python3 test.py
