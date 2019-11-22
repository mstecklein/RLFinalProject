from agent import Agent
from environment import Environment_v0, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES
from collections import deque
import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim




class DQNAgent(Agent):
    """
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self,
                 env: gym.Env,
                 gamma: float = 1.,
                 epsilon: float = 1.,
                 epsilon_decay: float = 0.999,
                 min_epsilon: float = 4e-2,
                 save_path: str = "./data/models/",
                 name: str = "DQN",
                 minibatch_size: int = 32,
                 max_mem_len: int = 100000,
                 num_training_episodes: int = 1000,
                 debug:bool = False):
        super().__init__(env)
        self._gamma = gamma
        self.epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._save_path = save_path
        self._name = name
        self._minibatch_size = minibatch_size
        self._max_mem_len = max_mem_len
        self._num_training_episodes = num_training_episodes
        self._debug = debug
        self.episode_num = 0
    
    
    def get_name(self):
        return self._name
    
    
    def run_episode(self):
        self.load()
        rewards = []
        raw_state = self.env.reset()
        self.summarize_history(raw_state)
        done = False
        while not done:
            a = np.argmax(self.Q_valid_only(state))
            raw_next_state, R, done, info = self.env.step(a)
            next_state = self.summarize_history(raw_next_state, info)
            rewards.append(R)
            state = next_state
        return rewards
    
    
    def train(self):
        save_freq = 10
        if self._debug:
            debug_update_freq = 10
            debug_ep_cnt = 0
            debug_sum_r = 0.
            debug_sum_steps = 0
            import time
            debug_start_t = time.time()
        replay_memory = deque(maxlen=self._max_mem_len)
        self.load()
        for _ in range(self._num_training_episodes):
            raw_state = self.env.reset()
            state = self.summarize_history(raw_state)
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    a = np.random.randint(self.env.action_space.n)
                else:
                    a = np.argmax(self.Q_valid_only(state))
                raw_next_state, R, done, info = self.env.step(a)
                next_state = self.summarize_history(raw_next_state, info)
                replay_memory.append((state, a, R, next_state, done))
                if self._debug: debug_sum_steps += 1; debug_sum_r += R
                if len(replay_memory) >= self._minibatch_size:
                    sample_idxs = np.random.randint(len(replay_memory), size=self._minibatch_size)
                    for sample_idx in sample_idxs:
                        s, _, R_r, s1, done_r = replay_memory[sample_idx]
                        Q_all_actions = self.Q(s)
                        if done_r:
                            target = R_r
                        else:
                            target = R_r + self._gamma * np.max(Q_all_actions)
                        target_all_actions = self.Q(s)
                        target_all_actions[a] = target
                        self.update(s, target_all_actions)
                state = next_state
            self.epsilon *= self._epsilon_decay
            if self.epsilon < self._min_epsilon: self.epsilon = self._min_epsilon
            if self.episode_num % save_freq == 0:
                self.save()
            if self._debug:
                debug_ep_cnt += 1
                if self.episode_num % debug_update_freq == 0:
                    print("Training episode number %d.  Avg. reward: %.2f.  Avg. # steps: %.2f.  Time since last update: %.2f  Epsilon: %.2f" % \
                            (self.episode_num, debug_sum_r/debug_ep_cnt, debug_sum_steps/debug_ep_cnt, time.time()-debug_start_t, self.epsilon))
                    debug_ep_cnt = 0; debug_sum_r = 0.; debug_sum_steps = 0; debug_start_t = time.time()
            self.episode_num += 1
                
                
    def get_model_filename(self):
        return self._save_path + self._name + ".model"
    
    
    def Q_valid_only(self, s):
        all_actions = self.Q(s)
        num_sensors = len(s[SENSOR_STATUSES])
        for snsr_num in range(num_sensors):
            if s[SENSOR_STATUSES][snsr_num] == 1: # on
                all_actions[self.env.get_action_turn_on(snsr_num)] = 0
            else: # off
                all_actions[self.env.get_action_turn_off(snsr_num)] = 0
        return all_actions / np.sum(all_actions)
    
    
    def Q(self, s):
        # Returns actions values for each action
        raise NotImplementedError()
    
    
    def update(self, state, target):
        # Updates Q function given the target
        raise NotImplementedError()
    
    
    def save(self):
        # Saves Q function state
        raise NotImplementedError()
    
    
    def load(self):
        # Loads Q function state
        raise NotImplementedError()
    
    
    def summarize_history(self, raw_state, info=None):
        # Given the raw state from the environment and the info dictionary,
        # summarize the history of the current state with some heuristic.
        raise NotImplementedError()











class SimpleDQNAgent(DQNAgent):
    """
        Only looks at first two components of the state space:
         - SENSOR_STATUSES
         - LOCATED_SOURCES
        And uses a regular NN for the action value function Q.
    """
    
    def __init__(self, env, **args):
        args["name"] = "SimpleDQN"
        super().__init__(env, **args)
        
        self._input_len = env.observation_space.spaces[SENSOR_STATUSES].n + \
                          env.observation_space.spaces[LOCATED_SOURCES].n
        
        self.model = nn.Sequential(
                    nn.Linear(self._input_len, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
#                     nn.Linear(100, 100),
#                     nn.ReLU(),
                    nn.Linear(100, self.env.action_space.n)) 
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                     lr=0.0001,
                                     betas=(0.9, 0.999))
    
    def Q(self, s):
        self.model.eval()
        input = torch.FloatTensor(self._flatten_state(s))
        output = self.model(input)
        return output.detach().numpy()
    
    
    def _flatten_state(self, s):
        return np.concatenate((
            s[SENSOR_STATUSES].flatten(),
            s[LOCATED_SOURCES].flatten()
        ))
    
    
    def update(self, state, target):
        self.optimizer.zero_grad()
        target_tnsr = torch.Tensor([target]).squeeze()
        predicted_tnsr = self.model(torch.FloatTensor(self._flatten_state(state)))
        loss = self.criterion(predicted_tnsr, target_tnsr)
        loss.backward()
        self.optimizer.step()
    
    
    def save(self):
        torch.save({
            'episode_num' : self.episode_num,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'epsilon' : self.epsilon
        }, self.get_model_filename())
    
    
    def load(self):
        try:
            checkpoint = torch.load(self.get_model_filename())
            self.episode_num = checkpoint['episode_num']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
        except FileNotFoundError:
            pass
    
    
    def summarize_history(self, raw_state, info=None):
        del raw_state[SENSOR_COVERAGES]
        return raw_state













if __name__ == "__main__":
    SimpleDQNAgent(Environment_v0(max_allowed_steps=100), debug=True).train()