from agent import Agent
from environment import Environment_v0
from random import randint


class RandomAgent(Agent):
    def __init__(self, env):
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
        env.reset()

        while not done:
            action = randint(0, self.num_actions - 1)
            _, reward, done, _ = self.env.step(action)
            rewards.append(reward)

        return rewards


if __name__ == '__main__':
    env = Environment_v0()
    agent = RandomAgent(env)

    for i in range(10):
        episode_rewards = agent.run_episode()
        print('The episode took {} steps and had a reward of {}'.format(len(episode_rewards), sum(episode_rewards)))
