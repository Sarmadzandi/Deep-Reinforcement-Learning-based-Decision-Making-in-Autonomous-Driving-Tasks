# Decision-Making in Autonomous Driving Tasks

In this repository, we will explore the Deep Q-Learning Network (DQN) algorithm and implement it on autonomous driving merge and highway-fast tasks. 

## Environment
We use [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) for this project. 

## Action Space
In this environment, there are 5 Discrete Meta-Actions:

0: 'Lane-Left'

1: 'IDLE'

2: 'Lane-Right'

3: 'Faster'

4: 'Slower'

## State Space
In this environment, each state is a continuous 5x5 matrix. The first row of the matrix represents the agent vehicle (also known as the ego vehicle), while the other rows represent other vehicles. The first column of the matrix indicates the presence or absence of vehicles in the current state and can only take a value of 0 or 1. The second and third columns of the matrix, along with the first row, show the position of the agent vehicle in the x and y directions. In contrast, the second and third columns of the other rows show the position of the surrounding vehicles in the x and y directions relative to the agent vehicle. The third and fourth columns of all rows display the speed of the corresponding vehicle relative to the agent vehicle. All values in the matrix are normalized between -1 and 1. 

Here's an example of what a state would look like:

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/cf0645a8-fb38-4db5-9433-a553801347c2)

## Reward Function
The reward function in the Highway-Env environment is obtained from car speed and collision with other cars. Environmental rewards are normalized in the range between 0 and 1. Also, the components of Vmin, Vmax, and V are respectively the lowest speed, the highest speed, and the instantaneous speed of the vehicle. Coefficients a and b also adjust the speed bonus and avoid collisions with other cars.

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/efda065d-4906-466a-ac7d-a07f59ca3e61)

## DRL Algorithm
we aim to implement the Deep Q-Learning Network (DQN) algorithm, with the following hyperparameters.

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/5d581a95-e4fa-4474-97df-4d1397962e53)


## Results
We applied the DQN algorithm to the merge-v0 task, and the average Reward during the learning episodes is shown Below:

* Average Reward during the learning episodes for merge-v0 task

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/blob/main/Images/1-Merge-task-state.png)

https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/992e83ff-4718-4f5c-adf0-30152425714d

Below is the average reward diagram for the highway-fast-v0 task using the DQN algorithm. One curve is with random initial weighting, and the other is with the initial weighting of the weights taught by the merge-v0 task (Transfer Learning).

* Average Reward during the learning episodes for highway-fast task

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/5ba23876-b866-42da-ac41-353449ae4622)


https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/1c77c4c0-b4d5-415e-afbc-02f0a181e7a5

https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/123d34c8-780e-4af0-a43b-b8f48a7e41ee

As it is shown, when we use transfer learning, we get closer to the average reward faster. The speed of our convergence has increased and regret has decreased. 

In contrast to the previous part, we use observation instead of state in this section. Additionally, due to the use of images, we utilize convolution layers in this network. It is important to note that each observation is obtained by calculating the difference between two rendered images.

* Average reward during learning episodes for merge-v0 task with state, and CNN network and observation.

![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/8338a98e-29c0-4544-aeb1-78cb8e12ef97)

The DQN algorithm with CNN network (first approach) has been found to perform better than the DQN algorithm with linear network and state (second approach). It has earned more average rewards in all episodes except the initial episodes. However, it's worth noting that this superiority doesn't always apply, and the DQN algorithm with observation won't always perform better than the DQN algorithm with state. If the observation can be estimated to have the characteristics of the state, they may be better than when we train the algorithm with the state.

https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/af35234e-ff2a-440f-98bf-fe6a9edca7f5



















