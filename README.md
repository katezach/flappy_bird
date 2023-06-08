# Flappy Bird for Gymnasium

This repository contains the implementation of four approaches; Heuristics, NEAT, PPO-Clip and simple DQN on Gymnasium environment of
the Flappy Bird game. 

The implementation of the game's logic and graphics wasbased on the [flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) project, by
[@Talendar](https://github.com/Talendar). We also used the [github repository](https://github.com/markub3327/flappy-bird-gymnasium) project, by [@markub3327](https://github.com/markub3327)

## State space we used
The "FlappyBird-v0" environment yields simple numerical information about the game's state as observations, including:

### `FlappyBird-v0`
* the last pipe's horizontal position
* the last top pipe's vertical position
* the last bottom pipe's vertical position
* the next pipe's horizontal position
* the next top pipe's vertical position
* the next bottom pipe's vertical position
* the next next pipe's horizontal position
* the next next top pipe's vertical position
* the next next bottom pipe's vertical position
* player's vertical position
* player's vertical velocity
* player's rotation

## Action space

* 0 - **do nothing**
* 1 - **flap**

## Rewards

* +0.1 - **every frame it stays alive**
* +1.0 - **successfully passing a pipe**
* -1.0 - **dying**

<br>

<p align="center">
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/dqn.gif?raw=true" 
       width="200"/>
</p>
    

## Run the code

To run the code, run the following command:

    $ flappy_bird_gymnasium

To see NEAT results, add an argument to the command:
  

To execute the PPO agent, add an argument to the command:

    $ flappy_bird_gymnasium --mode random

To see a Deep Q Network agent, add an argument to the command:

    $ flappy_bird_gymnasium --mode dqn
