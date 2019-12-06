# RLFinalProject
This repository contains the code for the CS394R Reinforcement Learning final project for Michael Stecklein and Aravind Srinivasan. This README briefly summarizes how to use the code.

## Comparing agents
Running return_comparison.py will compare the averaged returns of all pretrained agents and output a graph showing the results.
Running sources_found_comparison.py will compare the average number of sources found by all pretrained agents and output a graph showing the results.

## Running the environment with random agent
Our environment is an extension of the OpenAI gym environment. An example rendering with a random agent can be viewed by running the environment.py file.

## Training
Individual agents that require training (SARSA, DQN, REINFORCE) can be trained by running the python file for that agent. Models will automatically be saved and loaded. However, training was a time consuming process, so pretrained models are already provided, and will automatically be used in any of the comparison code mentioned above.

## Unused code
All code which was created during experimentation but not used in the final results is stored in the 'other_tries' directory. This code cannot be run, but stands as evidence of the trial-and-error process we underwent throughout this project.
