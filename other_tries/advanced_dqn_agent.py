from dqn_agent import DQNAgent
from environment import Environment_v0, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES, SENSOR_OBSERVATIONS
from always_on_agent import AlwaysOnAgent
import torch
from torch import nn
import torch.optim as optim
from collections import deque
import numpy as np



class SmallNNsAggregatedModel(nn.Module):
    def __init__(self, input_shape, num_hidden_layers, hidden_layer_width):
        super().__init__()
        self.poly_degree = 2
        self.nn_arr_shape = input_shape[:-1]
        self._nn_lookup_table = dict()
        # Create grid square small NN
#         self.small_nn = nn.Sequential()
#         h_layer_input_sz = input_shape[2]
#         for i in range(num_hidden_layers):
#             self.small_nn.add_module("H%d"%(i+1), nn.Linear(h_layer_input_sz, hidden_layer_width))
#             self.small_nn.add_module("ReLU%d"%(i+1), nn.ReLU())
#             h_layer_input_sz = hidden_layer_width
#         self.small_nn.add_module("FinalL", nn.Linear(hidden_layer_width, 1))
        # Weights
#         self.weights = torch.randn((self.poly_degree+1)**2, requires_grad=True)
        self.stage2_w = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        
        self.stage1_w = torch.nn.Parameter(torch.randn(27, requires_grad=True).unsqueeze(1))
        
    def forward(self, X_feature_mat, num_sensors_on):
        # Sum small NNs
#         nn_outputs = []
#         for r in range(self.nn_arr_shape[0]):
#             for c in range(self.nn_arr_shape[1]):
#                 nn_input_tnsr = torch.from_numpy(X_feature_mat[r,c]).float()
#                 nn_outputs.append( self.small_nn(nn_input_tnsr) )
#         nn_sum = sum(nn_outputs)
        X_feature_mat_tnsr = torch.from_numpy(X_feature_mat).float()
        nn_sum = torch.sum(torch.matmul(X_feature_mat_tnsr, self.stage1_w))
        # Create polynomial feature space
#         features = []
#         num_sensors_on_tnsr = torch.Tensor([num_sensors_on])
#         for x in range(self.poly_degree+1):
#             for s in range(self.poly_degree+1):
#                 features.append( torch.pow(nn_sum,x)*torch.pow(num_sensors_on_tnsr,s) )
#         features = torch.stack(features).squeeze() # or torch.cat ?
#         return torch.dot(features, self.weights)
        return nn_sum + self.stage2_w * torch.Tensor([num_sensors_on])


class AdvancedDQNAgent(DQNAgent):
    """
        Small NNs for each grid space share weights.
        Predict value of that square given an action.
            Inputs:
            - whether that square will be 1-covered (0/1)
            - whether that square will be 2-covered (0/1)
            - whether that square will be 3-covered (0/1)
            - sample mean (# observed sources / # observations)
            - sample st. dev.
            - whether a source has been found in that square (0/1)
            - whether a sensor is in that square (0/1)
            - history of (10) recent observations, each w/ 2 inputs:
              -- 1 / the # of grid squares in that observation
              -- the time since the observation
            - ...
            
        
        The small NNs will be combined by a summation, then used as an input
        along with the # of active sensors to a polynomial feature space of
        degree two (n=2). These features will be dotted with a weight vector
        to produce an output action-value, Q(s,a).
    """
        
    def __init__(self, env, **args):
        args["name"] = "AdvancedDQN"
        super().__init__(env, **args)
        self.history_len = 10
        self.X_feat_mat = None
        self.sensor_map = None
        self.curr_time = 0
        self.always_on_agent = AlwaysOnAgent(self.env)
        self._observation_history = deque(maxlen=self.history_len)
        self.model = SmallNNsAggregatedModel(self._get_feat_mat_shape(), 2, 40)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        
        
    def get_epsilon_action(self, state):
        return self.always_on_agent._next_action(state)
    
    
    def Q(self, s_summarized):
        # Returns actions values for each action
        action_values = []
        for a in range(self.env.action_space.n):
            feature_matrix, num_sensors = self._format_state(s_summarized, a)
            action_val_tnsr = self.model(feature_matrix, num_sensors)
            action_values.append(action_val_tnsr.item())
        return action_values
    
    def _format_state(self, s_summarized, action):
        # create coverage count map
        cov_cnt_map = np.zeros(self.env.field_shape)
        for sensor_num in range(self.env._num_sensors):
            if s_summarized[SENSOR_STATUSES][sensor_num] == 1: # is on
                cov_cnt_map += s_summarized[SENSOR_COVERAGES][sensor_num]
        # adjust coverage count map for action
        if action != self.env.NO_OP_ACTION:
            sensor_num, turn_on = self.env.get_action_info(action)
            if turn_on:
                cov_cnt_map += s_summarized[SENSOR_COVERAGES][sensor_num]
            else:
                cov_cnt_map -= s_summarized[SENSOR_COVERAGES][sensor_num]
        # feature matrix
        if self.X_feat_mat is None:
            self.X_feat_mat = np.zeros(self._get_feat_mat_shape())
        curr_idx = 0
        # whether that square will be 1-covered (0/1)
        self.X_feat_mat[:,:,curr_idx] = cov_cnt_map >=1
        curr_idx += 1
        # whether that square will be 2-covered (0/1)
        self.X_feat_mat[:,:,curr_idx] = cov_cnt_map >=2
        curr_idx += 1
        # whether that square will be 3-covered (0/1)
        self.X_feat_mat[:,:,curr_idx] = cov_cnt_map >=3
        curr_idx += 1
        # sample mean (# observed sources / # observations)
        # TODO
        curr_idx += 1
        # sample st. dev.
        # TODO
        curr_idx += 1
        # whether a source has been found in that square (0/1)
        self.X_feat_mat[:,:,curr_idx] = s_summarized[LOCATED_SOURCES]
        curr_idx += 1
        # whether a sensor is in that square (0/1)
        if self.sensor_map is None:
            self.sensor_map = np.zeros(self.env.field_shape)
            for snsr_loc in self.env.sensor_locs:
                self.sensor_map[snsr_loc] = 1
        self.X_feat_mat[:,:,curr_idx] = self.sensor_map
        curr_idx += 1
        # history of (10) recent observations, each w/ 2 inputs:
        self.X_feat_mat[:,:,curr_idx:curr_idx+2*self.history_len] = s_summarized["observationhistory"]
        # return: feature matrix, num_sensors
        return self.X_feat_mat, len(s_summarized[SENSOR_COVERAGES])
    
    
    def _get_feat_mat_shape(self):
        return self.env.field_shape + (7+2*self.history_len,)
    
    
    def update(self, s_summarized, target, action):
        # Updates Q function given the target
        self.optimizer.zero_grad()
        target_tnsr = torch.Tensor([target]).squeeze()
        feature_matrix, num_sensors = self._format_state(s_summarized, action)
        predicted_tnsr = self.model(feature_matrix, num_sensors)
        loss = self.criterion(predicted_tnsr, target_tnsr)
        loss.backward()
        self.optimizer.step()
    
    
    def save(self):
        # Saves Q function state
        torch.save({
            'episode_num' : self.episode_num,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'epsilon' : self.epsilon
        }, self.get_model_filename())
        if self._debug:
            print("Saved model state_dict:", self.model.state_dict())
    
    
    def load(self):
        # Loads Q function state
        try:
            checkpoint = torch.load(self.get_model_filename())
            self.episode_num = checkpoint['episode_num']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            if self._debug:
                print("Loaded model state_dict:", self.model.state_dict())
        except FileNotFoundError:
            pass
    
    
    def summarize_history(self, raw_state, info=None):
        # Given the raw state from the environment and the info dictionary,
        # summarize the history of the current state with some heuristic.
        if info == None:
            self._observation_history.clear()
        else:
            for obs in info[SENSOR_OBSERVATIONS]:
                obs_map = np.ones(self.env.field_shape, dtype=np.int)
                for sensor_num in obs:
                    obs_map = obs_map & raw_state[SENSOR_COVERAGES][sensor_num].astype(np.int)
                    self._observation_history.append((self.curr_time, obs_map))
        history_mat = np.zeros(self.env.field_shape + (2*self.history_len,))
        i = 0
        for obs_map, obs_time in self._observation_history:
            history_mat[:,:,i] = obs_map
            history_mat[:,:,i+1] = (self.curr_time-obs_time) * np.ones(self.env.field_shape)
            i += 2
        raw_state["observationhistory"] = history_mat
        self.curr_time += 1
        return raw_state
            





if __name__ == "__main__":
    env = Environment_v0(max_allowed_steps=100)
    agent = AdvancedDQNAgent(env, epsilon_decay=0.99, debug=True, num_training_episodes=10)
    agent.train()

'''
    agent._debug = False
    returns = []
    ep_lens = []
    num_episodes = 100
    for i in range(num_episodes):
        episode_rewards = agent.run_episode()
        print("Episode rewards:", episode_rewards)
        print("TOTAL:", sum(episode_rewards))
        returns.append(sum(episode_rewards))
        ep_lens.append(len(episode_rewards))
        
    print("AlwaysOnAgent, %d episodes:" % num_episodes)
    print("Return:    mean:%.2f   std:%.2f" % (np.mean(returns), np.std(returns)))
    print("Length:    mean:%.2f   std:%.2f" % (np.mean(ep_lens), np.std(ep_lens)))
'''
