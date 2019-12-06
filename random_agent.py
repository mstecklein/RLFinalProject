import gym
import numpy as np
from agent import Agent
from environment import Environment_v1
from random import randint
from time import sleep
from gym.wrappers.monitoring.video_recorder import VideoRecorder


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
        env.reset()
        video_recorder = VideoRecorder(self.env, path="/tmp/gym_%s_run.mp4"%self.get_name())
        video_recorder.capture_frame()
        while not done:
            action = randint(0, self.num_actions - 1)
            _, reward, done, _ = self.env.step(action)
            self.env.render()
            rewards.append(reward)
            video_recorder.capture_frame()
            sleep(0.005)

        video_recorder.close()
        return rewards


if __name__ == '__main__':
        
    env = Environment_v1(debug=True)
    agent = RandomAgent(env)

    returns = []
    ep_lens = []
    num_episodes = 1
    for i in range(num_episodes):
        episode_rewards = agent.run_episode()
        returns.append(sum(episode_rewards))
        ep_lens.append(len(episode_rewards))

    print("Random Agent, %d episodes:" % num_episodes)
    print("Return:    mean:%.2f   std:%.2f" % (np.mean(returns), np.std(returns)))
    print("Length:    mean:%.2f   std:%.2f" % (np.mean(ep_lens), np.std(ep_lens)))
