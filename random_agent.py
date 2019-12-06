import gym
import numpy as np
from agent import Agent
from environment import Environment_v1
from random import randint


class RandomAgent(Agent):
    def __init__(self, env: gym.Env):
        super(RandomAgent, self).__init__(env)

        self.name = "Random-Agent"
        self.num_actions = env.action_space.n

    def get_name(self):
        return self.name

    def train(self):
        pass

    def run_episode(self):
        done = False
        rewards = []
        self.env.reset()

        while not done:
            action = randint(0, self.num_actions - 1)
            _, reward, done, _ = self.env.step(action)
            rewards.append(reward)

        return rewards


if __name__ == '__main__':
        
    env = Environment_v1()
    agent = RandomAgent(env)

    returns = []
    ep_lens = []
    num_episodes = 1000
    for i in range(num_episodes):
        episode_rewards = agent.run_episode()
        returns.append(sum(episode_rewards))
        ep_lens.append(len(episode_rewards))
        
    print("RandomAgent, %d episodes:" % num_episodes)
    print("Return:    mean:%.2f   std:%.2f" % (np.mean(returns), np.std(returns)))
    print("Length:    mean:%.2f   std:%.2f" % (np.mean(ep_lens), np.std(ep_lens)))
