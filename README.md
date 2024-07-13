# Deep Reinforcement Learning Based Decision-Making in Autonomous Driving Tasks

![GitHub stars](https://img.shields.io/github/stars/sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks)
![GitHub forks](https://img.shields.io/github/forks/sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks)
![GitHub issues](https://img.shields.io/github/issues/sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks)
![GitHub license](https://img.shields.io/github/license/sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks)

## Table of Contents

- [Introduction](#introduction)
- [Environment](#environment)
- [Action Space](#action-space)
- [State Space](#state-space)
- [Reward Function](#reward-function)
- [Deep Q-Learning Network (DQN) Algorithm](#deep-q-learning-network-dqn-algorithm)
- [Results](#results)
  - [Merge-v0 Task](#merge-v0-task)
  - [Highway-Fast Task](#highway-fast-task)
  - [CNN Network and Observation](#cnn-network-and-observation)

## Introduction

In this project, we leverage Deep Reinforcement Learning (DRL), specifically the Deep Q-Learning Network (DQN), to develop decision-making algorithms for various driving tasks including merging, highway driving, and navigating intersections.

## Environment

We utilize the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) environment for this project. HighwayEnv is a highly configurable simulator for highway driving scenarios, providing realistic settings for testing various autonomous driving tasks. It supports multiple driving scenarios and offers a flexible API for integration with reinforcement learning frameworks.

## Action Space

The action space defines the set of possible actions that the agent (autonomous vehicle) can take. In HighwayEnv, the action space consists of 5 discrete meta-actions:

- **0: Lane-Left**: Move the vehicle one lane to the left.
- **1: IDLE**: Maintain the current lane and speed.
- **2: Lane-Right**: Move the vehicle one lane to the right.
- **3: Faster**: Increase the vehicle's speed.
- **4: Slower**: Decrease the vehicle's speed.

The `DiscreteMetaAction` type adds a layer of speed and steering controllers on top of the continuous low-level control, so that the ego-vehicle can automatically follow the road at a desired velocity. Then, the available meta-actions consist in changing the target lane and speed that are used as setpoints for the low-level controllers.

## State Space

The state space represents the current situation of the environment, which the agent uses to make decisions. In HighwayEnv, each state is a $V \times F$ array that describes a list of $V$ nearby vehicles by a set of features of size $F$. For instance:

| Presence | Vehicle      | x     | y     | vx    | vy    |
|----------|--------------|-------|-------|-------|-------|
| 1        | ego-vehicle  | 0.05  | 0.04  | 0.75  | 0     |
| 1        | vehicle 1    | -0.1  | 0.04  | 0.6   | 0     |
| 1        | vehicle 2    | 0.13  | 0.08  | 0.675 | 0     |
| ...      | ...          | ...   | ...   | ...   | ...   |
| 1        | vehicle V    | 0.222 | 0.105 | 0.9   | 0.025 |

- **Rows**: Each row represents a vehicle, with the first row always representing the ego vehicle.
- **Columns**: Each column is a feature that is described below:

  | Feature | Description |
  | :--- | :--- |
  | presence | Disambiguate agents at 0 offset from non-existent agents. |
  | Vehicle | Indicates the vehicle's name. |
  | $x$ | World offset of ego vehicle or offset to ego vehicle on the $x$ axis. |
  | $y$ | World offset of ego vehicle or offset to ego vehicle on the y axis. |
  | $v x$ | Velocity on the $x$ axis of vehicle. |
  | $v y$ | Velocity on the y axis of vehicle. |

## Reward Function

In HighwayEnv, the reward function balances speed optimization and collision avoidance:

$$
R(s, a) = a \frac{v - v_{\min}}{v_{\max} - v_{\min}} - b \cdot \text{collision}
$$

### Components of the Reward Function

- **Speed Reward**: Encourages the agent to drive at higher speeds, scaled between the minimum $v_{\min}$ and maximum $v_{\max}$ speeds.
- **Collision Penalty**: Penalizes the agent for collisions with other vehicles, promoting safer driving behavior.
- **Coefficients $a$ and $b$**: Adjust the influence of speed optimization and collision avoidance in the overall reward.

## Deep Q-Learning Network (DQN) Algorithm

Deep Q-learning is a value-based reinforcement learning algorithm where a neural network is used to approximate the Q-value function, which predicts the expected cumulative reward for taking an action in a given state.

### Hyperparameters

The following hyperparameters are used in our DQN implementation:

| Parameter                           | Value  |
|-------------------------------------|--------|
| BUFFER_SIZE                         | 10000  |
| BATCH_SIZE                          | 64     |
| GAMMA                               | 0.99   |
| UPDATE_EVERY                        | 4      |
| Learning rate (LR)                  | 0.0005 |
| α (Q-learning parameter)            | 0.001  |
| Epsilon start                       | 1      |
| Epsilon end                         | 0.001  |
| Epsilon decay                       | 0.995  |
| Number of iterations (runs)         | 5      |
| Number of episodes                  | 3600   |
| Max step                            | 10000  |

### DQN Training Process

1. **Experience Replay**: The agent's experiences (state, action, reward, next state) are stored in a replay buffer.
2. **Batch Training**: At each training step, a random batch of experiences is sampled from the replay buffer to update the network.
3. **Target Network**: A separate target network is used to stabilize training by reducing correlations between the action values.
4. **Epsilon-Greedy Strategy**: The agent initially explores the environment with random actions (high epsilon) and gradually shifts to exploiting learned policies (low epsilon).

## Results

The results section provides insights into the performance of the DQN algorithm on different driving tasks. We analyze the rewards obtained during training and evaluation, comparing different approaches and configurations.

### Merge-v0 Task

The Merge-v0 task simulates a vehicle merging into traffic. The average reward obtained during the learning episodes is shown below:

* Average Reward during the learning episodes for the merge-v0 task:

![Merge-v0 Reward](Images/1-Merge-task-state.png)

[Merge-v0 Reward Video](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/992e83ff-4718-4f5c-adf0-30152425714d)

### Highway-Fast Task

The Highway-Fast task involves high-speed driving on a highway. We compare the performance of the DQN algorithm with random initial weights and with weights transferred from the Merge-v0 task (Transfer Learning).

* Average Reward during the learning episodes for the highway-fast task with and without transfer learning:

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/5ba23876-b866-42da-ac41-353449ae4622)

https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/1c77c4c0-b4d5-415e-afbc-02f0a181e7a5

https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/123d34c8-780e-4af0-a43b-b8f48a7e41ee

### CNN Network and Observation

In this section, we use convolutional neural networks (CNNs) and image-based observations instead of state matrices. Each observation is derived from the difference between two rendered images.

* Average reward during learning episodes for the merge-v0 task with state, and CNN network and observation:

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/8338a98e-29c0-4544-aeb1-78cb8e12ef97)

[CNN Reward Video](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/af35234e-ff2a-440f-98bf-fe6a9edca7f5)

The DQN algorithm with CNN performs better than the DQN with linear networks and state matrices. It has earned more average rewards in all episodes except the initial episodes. However, this advantage is not guaranteed, and the performance depends on the quality of the observations. So, The DQN algorithm with observation won't always perform better than the DQN algorithm with state. If the observation can be estimated to have the characteristics of the state, they may be better than when we train the algorithm with the state.








