# Decision-Making in Autonomous Driving Tasks

In this repository, we will explore Deep RL algorithms and implement them on autonomous driving tasks to compare results.

## Environment
We use [highwayEnv](https://github.com/Farama-Foundation/HighwayEnv) for this project. 

## Action Space
In this environment, there are 5 Discrete Meta-Actions. Let us explain each of them:
0: 'Lane-Left'
1: 'IDLE'
2: 'Lane-Right'
3: 'Faster'
4: 'Slower'

## State Space
In this scenario, each state is a continuous 5x5 matrix. The matrix is displayed on the following and has a specific structure. The first row of the matrix represents the agent vehicle (also known as the ego vehicle), while the other rows represent other vehicles in the vicinity. The first column of the matrix indicates the presence or absence of vehicles in the current state and can only take a value of 0 or 1. 

The second and third columns of the matrix, along with the first row, show the position of the agent vehicle in the x and y directions, while the second and third columns of the other rows show the position of the surrounding vehicles in the x and y directions relative to the agent vehicle. The third and fourth columns of all rows display the speed of the corresponding vehicle relative to the agent vehicle. All values in the matrix are normalized between -1 and 1. 

Here's an example of what a state would look like:
![image](https://github.com/Sarmadzandi/Decision-Making-in-Autonomous-Driving-Tasks/assets/44917340/cf0645a8-fb38-4db5-9433-a553801347c2)
