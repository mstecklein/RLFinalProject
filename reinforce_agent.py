from agent import Agent
from environment import Environment_v0, Environment_v1, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES
from handcrafted_features import DotProductModel9, Q_HandCraftedFeatureVector
from collections import deque
import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
from gym.wrappers.monitoring.video_recorder import VideoRecorder







class SoftmaxDotProductModel9(DotProductModel9):
    
    def __init__(self, length):
        super().__init__(length, smart_init=False)
    
    def forward(self, all_feature_vectors):
        dot_prods = []
        for x in all_feature_vectors:
            dot_prod = super().forward(x)
            dot_prods.append(dot_prod.unsqueeze(0))
        return nn.functional.softmax( torch.cat(dot_prods), dim=0)





class Pi_HandCraftedFeatureVector(Q_HandCraftedFeatureVector):
    
    def __init__(self, env:Environment_v0, lr=0.001, epsilon=0.):
        super().__init__(env)
        self.epsilon = epsilon
        self.model = SoftmaxDotProductModel9(self.get_feature_vector_len())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def get_action_probs(self, state):
        if np.random.rand() < self.epsilon:
            num_actions = self.env.action_space.n
            action_probs = np.ones(num_actions) / num_actions
        else:
            action_probs = self.model(self.get_all_feature_vectors(state)).detach().numpy()
        return self._renormalize_valid_actions(action_probs)
    
    def _renormalize_valid_actions(self, all_action_probs):
        for snsr_num in range(self.env._num_sensors):
            if self.env._hidden_state[SENSOR_STATUSES][snsr_num] == 1: #on
                all_action_probs[self.env.get_action_turn_on(snsr_num)] = 0
            else: # off
                all_action_probs[self.env.get_action_turn_off(snsr_num)] = 0
        return all_action_probs / np.sum(all_action_probs)
            
    
    def update(self, state, action, G):
        self.optimizer.zero_grad()
        action_tnsr = torch.FloatTensor([action])
        probs_tnsr = self.model(self.get_all_feature_vectors(state))
        m = Categorical(probs_tnsr)
        loss = -m.log_prob(action_tnsr) * G
        loss.backward()
        self.optimizer.step()










class ReinforceAgent(Agent):
    
    def __init__(self,
                 env: gym.Env,
                 gamma: float = 1.,
                 save_path: str = "./data/models/",
                 name: str = "REINFORCE",
                 num_training_episodes: int = 10000,
                 lr=0.001,
                 epsilon=0.,
                 debug:bool = False):
        super().__init__(env)
        self.gamma = gamma
        self._save_path = save_path
        self._name = name
        self._debug = debug
        self.episode_num = 0
        self.epsilon = epsilon
        self.pi = Pi_HandCraftedFeatureVector(env, lr=lr, epsilon=epsilon)
        self._previously_saved_w = None


    def get_name(self):
        return self._name


    def train(self, num_episodes=10000):
        self.load()
        self.pi.epsilon = self.epsilon
        for _ in range(num_episodes):
            
            # Generate an episode
            rawS = self.env.reset()
            S = self.pi.parse_new_state_info(rawS, None)
            Ss = [S] # states
            Rs = [0] # rewards (first rewards is R1 at index 1, from S0 -> S1)
            As = []  # actions
            T = 0
            done = False
            while not done:
                A_probs = self.pi.get_action_probs(S)
                A = np.random.choice(len(A_probs), p=A_probs)
                As.append(A)
                rawS, R, done, info = self.env.step(A)
                S = self.pi.parse_new_state_info(rawS, info)
                Ss.append(S)
                Rs.append(R)
                T += 1
        
            # Iterate over episode to calculate MC rewards (G's)
            Gs = np.zeros(T)
            Gs[T-1] = Rs[T]
            for t in reversed(range(T-1)): # T-2,...,1,0
                Gs[t] = Rs[t+1] + self.gamma * Gs[t+1]
        
            # Iterate over episode and perform updates
            for t in range(T): # 0,1,...,T-1
                self.pi.update(Ss[t], As[t], self.gamma**t * Gs[t])
            
            self.episode_num += 1
            if self.episode_num % 10 == 0:
                print("Finished episode %d" % (self.episode_num))
                self.save()


    def run_episode(self, record=False):
        self.load()
        self.pi.epsilon = 0.
        rewards = []
        raw_state = self.env.reset()
        state = self.pi.parse_new_state_info(raw_state, None)
        done = False
        if self._debug and record:
            video_recorder = VideoRecorder(self.env, path="/tmp/gym_%s_run.mp4"%self.get_name())
            video_recorder.capture_frame()
        while not done:
            A_probs = self.pi.get_action_probs(state)
            a = np.argmax(A_probs)
            raw_next_state, R, done, info = self.env.step(a)
            next_state = self.pi.parse_new_state_info(raw_next_state, info)
            rewards.append(R)
            state = next_state
            if self._debug and record:
                video_recorder.capture_frame()
        if self._debug and record:
            video_recorder.close()
        return rewards


    def save(self):
        torch.save({
            'episode_num' : self.episode_num,
            'model_state_dict' : self.pi.model.state_dict(),
            'optimizer_state_dict' : self.pi.optimizer.state_dict()
        }, self.get_model_filename())
        if self._debug:
            w = self.pi.model.weights.detach().numpy()
            if not self._previously_saved_w is None:
                print("\t\t\t max diff in w since last save:", np.max(np.abs(w-self._previously_saved_w)))
            self._previously_saved_w = w.copy()
            print("Saved model state dict:", self.pi.model.state_dict())


    def load(self):
        try:
            checkpoint = torch.load(self.get_model_filename())
            self.episode_num = checkpoint['episode_num']
            self.pi.model.load_state_dict(checkpoint['model_state_dict'])
            self.pi.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self._debug:
                print("Loaded model state dict:", self.pi.model.state_dict())
                print("\tfrom path ", self.get_model_filename())
        except FileNotFoundError:
            print("Did not load a model")
            print("\tcurrent w is:", self.pi.model.weights.detach().numpy())
    
                
    def get_model_filename(self):
        return self._save_path + self._name + ".model"
    
    
 




if __name__ == '__main__':
    env = Environment_v1(debug=False)
    agent = ReinforceAgent(env, debug=True, lr=0.001,
                        epsilon=0.5
            )
    agent.train(10000)
 
    env._debug = False
    returns = []
    num_eps = 100
    for i in range(num_eps):
        ep_rewards = agent.run_episode(False)
        print("TOTAL:", sum(ep_rewards))
        returns.append(sum(ep_rewards))
    print("ReinforceAgent, %d episodes:" % num_eps)
    print("Return:    mean:%.2f    std:%.2f" % (np.mean(returns), np.std(returns)))   