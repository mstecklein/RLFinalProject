from agent import Agent
from environment import *




class NoOpAgent(Agent):

    def __init__(self, env):
        super().__init__(env)

    def get_name(self):
        return "NoOpAgent"
    
    def run_episode(self):
        rewards = []
        state = self.env.reset()
        done = False
        while not done:
            a = self.env.NO_OP_ACTION
            state_next, R, done, info = self.env.step(a)
            rewards.append(R)
            state = state_next
        return rewards


if __name__ == '__main__':
        
    env = Environment_v1()
    agent = NoOpAgent(env)

    returns = []
    ep_lens = []
    num_episodes = 1000
    for i in range(num_episodes):
        episode_rewards = agent.run_episode()
        returns.append(sum(episode_rewards))
        ep_lens.append(len(episode_rewards))
        
    print("NoOpAgent, %d episodes:" % num_episodes)
    print("Return:    mean:%.2f   std:%.2f" % (np.mean(returns), np.std(returns)))
    print("Length:    mean:%.2f   std:%.2f" % (np.mean(ep_lens), np.std(ep_lens)))
