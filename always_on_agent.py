from agent import Agent
from environment import *
import numpy as np




class AlwaysOnAgent(Agent):
    
    def __init__(self, env):
        super().__init__(env)
        
    def get_name(self):
        return "AlwaysOnAgent"
    
    def run_episode(self):
        rewards = []
        state = self.env.reset()
        done = False
        while not done:
            a = self._next_action(state)
            state_next, R, done, info = self.env.step(a)
            rewards.append(R)
            state = state_next
        return rewards
    
    def _next_action(self, s):
        snsr_statuses = s[SENSOR_STATUSES]
        if np.sum(snsr_statuses == 0) == 0:
            return self.env.NO_OP_ACTION
        snsr_num = np.random.randint(len(snsr_statuses))
        while snsr_statuses[snsr_num] == 1:
            snsr_num = np.random.randint(len(snsr_statuses))
        return self.env.get_action_turn_on(snsr_num)
    
    

if __name__ == '__main__':
    env = Environment_v0()
    agent = AlwaysOnAgent(env)

    returns = []
    ep_lens = []
    num_episodes = 10000
    for i in range(num_episodes):
        episode_rewards = agent.run_episode()
        returns.append(sum(episode_rewards))
        ep_lens.append(len(episode_rewards))
        
    print("AlwaysOnAgent, %d episodes:" % num_episodes)
    print("Return:    mean:%.2f   std:%.2f" % (np.mean(returns), np.std(returns)))
    print("Length:    mean:%.2f   std:%.2f" % (np.mean(ep_lens), np.std(ep_lens)))