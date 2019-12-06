from agent import Agent
from environment import Environment_v0, Environment_v1, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES
from collections import deque
import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from handcrafted_features import  Q_FeatureVector, HandCraftedFeatureVector




class DQNAgentAbstract(Agent):
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
                 skip_validation:bool = False,
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
        self._skip_validation = skip_validation
        self._debug = debug
        self.episode_num = 0
    
    
    def get_name(self):
        return self._name
    
    
    def run_episode(self, record=False):
        self.load()
        rewards = []
        raw_state = self.env.reset()
        state = self.summarize_history(raw_state)
        done = False
        if self._debug and record:
            video_recorder = VideoRecorder(self.env, path="/tmp/gym_%s_run.mp4"%self.get_name())
            video_recorder.capture_frame()
        while not done:
            Q_vals = self.Q_valid_only(state)
            a = np.argmax(Q_vals)
            raw_next_state, R, done, info = self.env.step(a)
            next_state = self.summarize_history(raw_next_state, info)
            rewards.append(R)
            state = next_state
            if self._debug and record:
                video_recorder.capture_frame()
        if self._debug and record:
            video_recorder.close()
        return rewards
    
    
    def train(self, num_episodes, printout_statuses=True):
        save_freq = 10
        printout_statuses |= self._debug
        if printout_statuses:
            debug_update_freq = save_freq
            debug_ep_cnt = 0
            debug_sum_r = 0.
            debug_sum_steps = 0
            import time
            debug_start_t = time.time()
        replay_memory = deque(maxlen=self._max_mem_len)
        self.load()
        for _ in range(num_episodes):
            raw_state = self.env.reset()
            state = self.summarize_history(raw_state)
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    a = self.get_epsilon_action(state)
                else:
                    a = np.argmax(self.Q_valid_only(state))
                raw_next_state, R, done, info = self.env.step(a)
                next_state = self.summarize_history(raw_next_state, info)
                replay_memory.append((state, a, R, next_state, done))
                if printout_statuses: debug_sum_steps += 1; debug_sum_r += R
                if len(replay_memory) >= self._minibatch_size:
                    sample_idxs = np.random.randint(len(replay_memory), size=self._minibatch_size)
                    for sample_idx in sample_idxs:
                        s, a_r, R_r, s1, done_r = replay_memory[sample_idx]
                        Q_all_actions = self.Q_valid_only(s1)
                        if done_r:
                            target = R_r
                        else:
                            target = R_r + self._gamma * np.max(Q_all_actions)
                        self.update(s, target, a_r)
                state = next_state
            self.epsilon *= self._epsilon_decay
            if self.epsilon < self._min_epsilon: self.epsilon = self._min_epsilon
            if self.episode_num % save_freq == 0:
                self.save()
            if printout_statuses:
                debug_ep_cnt += 1
                if self.episode_num % debug_update_freq == 0:
                    print("Training episode number %d.  Avg. reward: %.2f.  Avg. # steps: %.2f.  Time since last update: %.2f  Epsilon: %.2f" % \
                            (self.episode_num, debug_sum_r/debug_ep_cnt, debug_sum_steps/debug_ep_cnt, time.time()-debug_start_t, self.epsilon))
                    debug_ep_cnt = 0; debug_sum_r = 0.; debug_sum_steps = 0; debug_start_t = time.time()
            self.episode_num += 1
    
    
    def get_epsilon_action(self, state):
        return np.random.randint(self.env.action_space.n)
    
                
    def get_model_filename(self):
        return self._save_path + self._name + ".model"
    
    
    def Q_valid_only(self, s):
        if self._skip_validation:
            return self.Q(s)
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
    
    
    def update(self, state, target, action):
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
    
    
    
    
    
    







class DQNAgent(DQNAgentAbstract):
    
    def __init__(self, env, Qfunc: Q_FeatureVector, **args):
        if not "name" in args:
            args["name"] = "DQN"
        super().__init__(env, **args)
        self.Qfunc = Qfunc
    
    
    def Q(self, s):
        # Returns actions values for each action
        return self.Qfunc.get_all_action_values(s)
    
    
    def update(self, state, target, action):
        # Updates Q function given the target
        self.Qfunc.update(state, target, action)
    
    
    def save(self):
        # Saves Q function state
        torch.save({
            'episode_num' : self.episode_num,
            'model_state_dict' : self.Qfunc.model.state_dict(),
            'optimizer_state_dict' : self.Qfunc.optimizer.state_dict(),
            'epsilon' : self.epsilon
        }, self.get_model_filename())
        if self._debug:
            print("Saved model state dict:", self.Qfunc.model.state_dict())
    
    
    def load(self):
        # Loads Q function state
        try:
            checkpoint = torch.load(self.get_model_filename())
            self.episode_num = checkpoint['episode_num']
            self.Qfunc.model.load_state_dict(checkpoint['model_state_dict'])
            self.Qfunc.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print("LOADED:", self.get_model_filename())
        except FileNotFoundError:
            pass
    
    
    def summarize_history(self, raw_state, info=None):
        # Given the raw state from the environment and the info dictionary,
        # summarize the history of the current state with some heuristic.
        return self.Qfunc.parse_new_state_info(raw_state, info)













if __name__ == "__main__":
    env = Environment_v1()
    agent = DQNAgent(env, Q_FeatureVector(HandCraftedFeatureVector(env)))
    agent.train(1000)
