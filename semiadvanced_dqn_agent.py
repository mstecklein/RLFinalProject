from dqn_agent import DQNAgent
from environment import Environment_v1, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES, SENSOR_OBSERVATIONS
from always_on_agent import AlwaysOnAgent
import torch
from torch import nn
import torch.optim as optim
from collections import deque
import numpy as np


"""
    Linear model for each grid space share weights.
    Predict value of that square given an action.
        Inputs:
        1. whether that square will be 1-covered (0/1)
        2. whether that square will be 2-covered (0/1)
        3. whether that square will be 3-covered (0/1)
        4. # of 1-observations in that square
        5. # of 3-observations in that square
        6. whether a source has been found or a sensor exists in that square (0/1)
        7. # of sensors on
        8. time-discounted observation history of that square
        9. #8 x #2
       10. #8 x #3
        
    The linear models will be combined by a summation, then added with a weight*num_sensors_on
    to produce an output action-value, Q(s,a).
"""


class SmallAggregatedModel(nn.Module):
    """
    The model returns 100 independently generated values. The
    true action value is the sum of these.
    """
    def __init__(self, input_shape):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([
            10., # 1
            20., # 2
            30., # 3
            0.05, # 4
            -5., # 5
            0., # 6
            0., # 7
            0., # 8
            50., # 9
            100.  # 10
        ], requires_grad=True).unsqueeze(1))
        
    def forward(self, X_feature_mat):
        X_feature_mat_tnsr = torch.from_numpy(X_feature_mat).float()
        return torch.matmul(X_feature_mat_tnsr, self.weights)




class SemiAdvancedDQNAgent(DQNAgent):
        
    def __init__(self, env, **args):
        args["name"] = "SemiAdvancedDQN"
        super().__init__(env, **args)
        self.model = SmallAggregatedModel(self._get_feat_mat_shape())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        
        self._X_1_on = np.zeros(self._get_feat_mat_shape())
        self._X_no_change = np.zeros(self._get_feat_mat_shape())
        self._X_1_off = np.zeros(self._get_feat_mat_shape())
        self._obs_discount_factor = 0.5
        self._reset_things()
    
    
    def Q(self, s_summarized):
        # Returns actions values for each action
        X_1_on, X_no_change, X_1_off = self._create_feature_mats(s_summarized)
        gridsqr_vals_one_on = self.model(X_1_on).detach().numpy()
        gridsqr_vals_no_change = self.model(X_no_change).detach().numpy()
        gridsqr_vals_one_off = self.model(X_1_off).detach().numpy()
        action_values = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            snsr_num, turn_on = self.env.get_action_info(a)
            snsr_cov = s_summarized[SENSOR_COVERAGES][snsr_num]
            snsr_not_cov = ~ snsr_cov.astype(np.bool)
            if turn_on:
                a_val = np.sum( snsr_cov * gridsqr_vals_one_on + \
                                snsr_not_cov * gridsqr_vals_no_change )
            else:
                a_val = np.sum( snsr_cov * gridsqr_vals_one_off + \
                                snsr_not_cov * gridsqr_vals_no_change )
            action_values[a] = a_val
        return action_values
    
    
    def _create_feature_mats(self, s_summarized):
        idx = 0
        cov_cnt_map = s_summarized["coveragecountmap"]
        #  1. whether that square will be 1-covered (0/1)
        self._X_1_on[:,:,idx] = 1
        self._X_no_change[:,:,idx] = cov_cnt_map >= 1
        self._X_1_off[:,:,idx] = (cov_cnt_map - 1) >= 1
        idx += 1
        #  2. whether that square will be 2-covered (0/1)
        self._X_1_on[:,:,idx] = (cov_cnt_map + 1) >= 2
        self._X_no_change[:,:,idx] = cov_cnt_map >= 2
        self._X_1_off[:,:,idx] = (cov_cnt_map - 1) >= 2
        idx += 1
        #  3. whether that square will be 3-covered (0/1)
        self._X_1_on[:,:,idx] = (cov_cnt_map + 1) >= 3
        self._X_no_change[:,:,idx] = cov_cnt_map >= 3
        self._X_1_off[:,:,idx] = (cov_cnt_map - 1) >= 3
        idx += 1
        #  4. # of 1-observations in that square
        self._X_1_on[:,:,idx] = s_summarized["num1obs"] + 1
        self._X_no_change[:,:,idx] = s_summarized["num1obs"] + (cov_cnt_map >= 1)
        self._X_1_off[:,:,idx] = s_summarized["num1obs"] + ((cov_cnt_map - 1) >= 1)
        idx += 1
        #  5. # of 3-observations in that square
        self._X_1_on[:,:,idx] = s_summarized["num3obs"] + ((cov_cnt_map + 1) >= 3)
        self._X_no_change[:,:,idx] = s_summarized["num3obs"] + (cov_cnt_map >= 3)
        self._X_1_off[:,:,idx] = s_summarized["num3obs"] + ((cov_cnt_map - 1) >= 3)
        idx += 1
        #  6. whether a source has been found or a sensor exists in that square (0/1)
        self._X_1_on[:,:,idx] = s_summarized["sourcesensorlocs"]
        self._X_no_change[:,:,idx] = s_summarized["sourcesensorlocs"]
        self._X_1_off[:,:,idx] = s_summarized["sourcesensorlocs"]
        idx += 1
        #  7. # of sensors on
        self._X_1_on[:,:,idx] = s_summarized["numsensorson"]
        self._X_no_change[:,:,idx] = s_summarized["numsensorson"]
        self._X_1_off[:,:,idx] = s_summarized["numsensorson"]
        idx += 1
        #  8. time-discounted observation history of that square
        self._X_1_on[:,:,idx] = s_summarized["timediscountedobshistory"]
        self._X_no_change[:,:,idx] = s_summarized["timediscountedobshistory"]
        self._X_1_off[:,:,idx] = s_summarized["timediscountedobshistory"]
        idx += 1
        #  9. #8 x #2
        self._X_1_on[:,:,idx] = ((cov_cnt_map + 1) >= 2) * s_summarized["timediscountedobshistory"]
        self._X_no_change[:,:,idx] = (cov_cnt_map >= 2) * s_summarized["timediscountedobshistory"]
        self._X_1_off[:,:,idx] = ((cov_cnt_map - 1) >= 2) * s_summarized["timediscountedobshistory"]
        idx += 1
        # 10. #8 x #3
        self._X_1_on[:,:,idx] = ((cov_cnt_map + 1) >= 3) * s_summarized["timediscountedobshistory"]
        self._X_no_change[:,:,idx] = (cov_cnt_map >= 3) * s_summarized["timediscountedobshistory"]
        self._X_1_off[:,:,idx] = ((cov_cnt_map - 1) >= 3) * s_summarized["timediscountedobshistory"]
        idx += 1
        # return:
        return self._X_1_on, self._X_no_change, self._X_1_off
    
    
    def update(self, s_summarized, target, action):
        # Updates Q function given the target
        self.optimizer.zero_grad()
        target_tnsr = torch.Tensor([target]).squeeze()
        feature_matrix = self._create_action_specific_feature_mat(s_summarized, action)
        predicted_tnsr = torch.sum( self.model(feature_matrix) )
        loss = self.criterion(predicted_tnsr, target_tnsr)
        loss.backward()
        self.optimizer.step()
        
        
    def _create_action_specific_feature_mat(self, s_summarized, action):
        X_1_on, X_no_change, X_1_off = self._create_feature_mats(s_summarized)
        snsr_num, turn_on = self.env.get_action_info(action)
        cov_map = s_summarized[SENSOR_COVERAGES][snsr_num]
        cov_map = cov_map.reshape(cov_map.shape + (1,))
        X = X_no_change * (~ cov_map.astype(np.bool))
        if turn_on:
            X += X_1_on * cov_map
        else:
            X += X_1_off * cov_map
        return X
    
    
    def summarize_history(self, raw_state, info=None):
        if info is None:
            self._reset_things()
        # create coverage count map
        cov_cnt_map = np.zeros(self.env.field_shape)
        for sensor_num in range(self.env._num_sensors):
            if raw_state[SENSOR_STATUSES][sensor_num] == 1: # is on
                cov_cnt_map += raw_state[SENSOR_COVERAGES][sensor_num]
        raw_state["coveragecountmap"] = cov_cnt_map
        # number of 1-observations
        self._num1obs += (cov_cnt_map >= 1)
        raw_state["num1obs"] = self._num1obs
        # number of 3-observations
        self._num3obs += (cov_cnt_map >= 3)
        raw_state["num3obs"] = self._num3obs
        # source and sensor locations
        if self._sourcesensorlocs is None:
            self._sourcesensorlocs = np.zeros(self.env.field_shape, dtype=np.bool)
            for snsr_loc in self.env.sensor_locs:
                self._sourcesensorlocs[snsr_loc] = 1
        self._sourcesensorlocs |= raw_state[LOCATED_SOURCES].astype(np.bool)
        raw_state["sourcesensorlocs"] = self._sourcesensorlocs
        # number of sensors on
        raw_state["numsensorson"] = np.sum(raw_state[SENSOR_STATUSES])
        # time-discounted observation history
        self._timediscountedobshistory *= self._obs_discount_factor
        if not info is None:
            for obs in info[SENSOR_OBSERVATIONS]:
                obs_map = np.ones(self.env.field_shape, dtype=np.int)
                for sensor_num in obs:
                    obs_map = obs_map & raw_state[SENSOR_COVERAGES][sensor_num].astype(np.int)
                self._timediscountedobshistory += (obs_map.astype(np.float) / np.sum(obs_map))
        raw_state["timediscountedobshistory"] = self._timediscountedobshistory
        # return:
        return raw_state
        
    
    def _reset_things(self):
        self._num1obs = np.zeros(self.env.field_shape)
        self._num3obs = np.zeros(self.env.field_shape)
        self._sourcesensorlocs = None
        self._timediscountedobshistory = np.zeros(self.env.field_shape)
    
    
    def _get_feat_mat_shape(self):
        return self.env.field_shape + (10,)
    
    
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
            if self._debug:
                print("Did not load a model.")
            





if __name__ == "__main__":
    DEBUG = False
    env = Environment_v1(debug=DEBUG)
    agent = SemiAdvancedDQNAgent(env, epsilon_decay=0.99, debug=DEBUG)
    agent.train()
    print("Finished training.")

    returns = []
    ep_lens = []
    num_episodes = 100
    for i in range(num_episodes):
        episode_rewards = agent.run_episode()
#         print("Episode rewards:", episode_rewards)
#         print("TOTAL:", sum(episode_rewards))
        returns.append(sum(episode_rewards))
        ep_lens.append(len(episode_rewards))
        
    print("AlwaysOnAgent, %d episodes:" % num_episodes)
    print("Return:    mean:%.2f   std:%.2f" % (np.mean(returns), np.std(returns)))
    print("Length:    mean:%.2f   std:%.2f" % (np.mean(ep_lens), np.std(ep_lens)))
