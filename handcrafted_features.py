from environment import Environment_v0
from environment import Environment_v1, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES, SENSOR_OBSERVATIONS
from always_on_agent import AlwaysOnAgent
import torch
from torch import nn
import torch.optim as optim
from collections import deque
import numpy as np







class FeatureVector:
    
    def get_feature_vector_len(self):
        raise NotImplementedError
    
    def parse_new_state_info(self, raw_state, info):
        # Parses environment info into raw state. Returns parsed state.
        # NOTE: if 'info' is None, it resets all saved history
        raise NotImplementedError
    
    def __call__(self, parsed_state, action):
        # Returns feature vector X(s,a)
        # 'state' must be a previously parsed state from parse_new_state_info().
        raise NotImplementedError
    
    def get_all_feature_vectors(self, parsed_state):
        # Returns array of all feature vectors X(s,a) for all 'a'
        # 'state' must be a previously parsed state from parse_new_state_info().
        raise NotImplementedError







class MinimalFeatureVector(FeatureVector):
    
    def __init__(self, env:Environment_v0):
        self.env = env
    
    def get_feature_vector_len(self):
        return self.env._hidden_state[SENSOR_STATUSES].size + self.env._hidden_state[LOCATED_SOURCES].size
    
    def parse_new_state_info(self, raw_state, info):
        # Parses environment info into raw state. Returns parsed state.
        # NOTE: if 'info' is None, it resets all saved history
        return raw_state
    
    def __call__(self, parsed_state, action):
        # Returns feature vector X(s,a)
        # 'state' must be a previously parsed state from parse_new_state_info().
        sensor_state = parsed_state[SENSOR_STATUSES]
        located_state = parsed_state[LOCATED_SOURCES].flatten()
        return np.concatenate((sensor_state, located_state))
    
    def get_all_feature_vectors(self, parsed_state):
        # Returns array of all feature vectors X(s,a) for all 'a'
        # 'state' must be a previously parsed state from parse_new_state_info().
        x = [self(parsed_state, None)]
        return x * self.env.action_space.n









class HandCraftedFeatureVector(FeatureVector):
    """
    Vector index descriptions:
        0. sum of: whether each square will be 1-covered (0/1)
        1. sum of: whether each square will be 2-covered (0/1)
        2. sum of: whether each square will be 3-covered (0/1)
        3. # of sensors on
        4. # of sources found x # of sensors on
        5. sum of: whether each square will be 2-covered (0/1) 
             x time-discounted observation history of that square
        6. sum of: whether each square will be 3-covered (0/1) 
             x "                                                "
        7. sum of: whether each square will be 2-covered (0/1)
             x whether a source has been found or a sensor exists in each square (0/1)
        8. sum of: whether each square will be 3-covered (0/1) 
             x "                                                                     "
    """
    
    def __init__(self, env:Environment_v0):
        self.size = 9
        self.env = env
        self._X_1_on = np.zeros(self._get_feat_mat_shape())
        self._X_no_change = np.zeros(self._get_feat_mat_shape())
        self._X_1_off = np.zeros(self._get_feat_mat_shape())
        self._obs_discount_factor = 0.5
        self._scaling_factors = np.array([
            100., # 0
            100., # 1
            100., # 2
            20., # 3
            200., # 4
            100., # 5
            100., # 6
            100., # 7
            100., # 8
        ])
        
    
    def _get_feat_mat_shape(self):
        return self.env.field_shape + (self.get_feature_vector_len(),)
    
    
    def get_feature_vector_len(self):
        return self.size
        
        
    def parse_new_state_info(self, raw_state, info):
        # Parses environment info into raw state. Returns parsed state.
        # NOTE: if 'info' is None, it resets all saved history
        if info is None:
            self._reset_things()
        # create coverage count map
        cov_cnt_map = np.zeros(self.env.field_shape)
        for sensor_num in range(self.env._num_sensors):
            if raw_state[SENSOR_STATUSES][sensor_num] == 1: # is on
                cov_cnt_map += raw_state[SENSOR_COVERAGES][sensor_num]
        raw_state["coveragecountmap"] = cov_cnt_map
        # source and sensor locations
        if self._sourcesensorlocs is None:
            self._sourcesensorlocs = np.zeros(self.env.field_shape, dtype=np.bool)
            for snsr_loc in self.env.sensor_locs:
                self._sourcesensorlocs[snsr_loc] = 1
        self._sourcesensorlocs |= raw_state[LOCATED_SOURCES].astype(np.bool)
        raw_state["sourcesensorlocs"] = self._sourcesensorlocs
        # number of sensors on
        raw_state["numsensorson"] = np.sum(raw_state[SENSOR_STATUSES])
        # number of sources found
        raw_state["numsourcesfound"] = np.sum(raw_state[LOCATED_SOURCES])
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
        self._sourcesensorlocs = None
        self._timediscountedobshistory = np.zeros(self.env.field_shape)
    
    
    def __call__(self, state, action):
        # Returns feature vector X(s,a)
        # 'state' must be a previously parsed state from parse_new_state_info().
        X_1_on, X_no_change, X_1_off = self._create_feature_mats(state)
        if not action is self.env.NO_OP_ACTION:
            snsr_num, turn_on = self.env.get_action_info(action)
            coverage_map = state[SENSOR_COVERAGES][snsr_num]
            coverage_map = coverage_map.reshape(coverage_map.shape + (1,))
            X = X_no_change * (~ coverage_map.astype(np.bool))
            if turn_on:
                X += X_1_on * coverage_map
            else:
                X += X_1_off * coverage_map
        else:
            X = X_no_change
        return self._create_feat_vec_from_mat(X, state, action)
        
        
    def get_all_feature_vectors(self, state):
        # Returns array of all feature vectors X(s,a) for all 'a'
        # 'state' must be a previously parsed state from parse_new_state_info().
        X_1_on, X_no_change, X_1_off = self._create_feature_mats(state)
        feat_vecs = []
        for action in range(self.env.action_space.n):
            if not action is self.env.NO_OP_ACTION:
                snsr_num, turn_on = self.env.get_action_info(action)
                coverage_map = state[SENSOR_COVERAGES][snsr_num]
                coverage_map = coverage_map.reshape(coverage_map.shape + (1,))
                X = X_no_change * (~ coverage_map.astype(np.bool))
                if turn_on:
                    X += X_1_on * coverage_map
                else:
                    X += X_1_off * coverage_map
            else:
                X = X_no_change
            x_feat_vec = self._create_feat_vec_from_mat(X, state, action)
            feat_vecs.append(x_feat_vec)
        return feat_vecs
    
    
    def _create_feature_mats(self, s_summarized):
        idx = 0
        cov_cnt_map = s_summarized["coveragecountmap"]
        #  0. whether each square will be 1-covered (0/1)
        self._X_1_on[:,:,idx] = 1
        self._X_no_change[:,:,idx] = cov_cnt_map >= 1
        self._X_1_off[:,:,idx] = (cov_cnt_map - 1) >= 1
        idx += 1
        #  1. whether each square will be 2-covered (0/1)
        self._X_1_on[:,:,idx] = (cov_cnt_map + 1) >= 2
        self._X_no_change[:,:,idx] = cov_cnt_map >= 2
        self._X_1_off[:,:,idx] = (cov_cnt_map - 1) >= 2
        idx += 1
        #  2. whether each square will be 3-covered (0/1)
        self._X_1_on[:,:,idx] = (cov_cnt_map + 1) >= 3
        self._X_no_change[:,:,idx] = cov_cnt_map >= 3
        self._X_1_off[:,:,idx] = (cov_cnt_map - 1) >= 3
        idx += 1
        #  3. # of sensors on
        num_gridsquares = cov_cnt_map.size
        self._X_1_on[:,:,idx] = min(s_summarized["numsensorson"] + 1, self.env._num_sensors) / num_gridsquares
        self._X_no_change[:,:,idx] = s_summarized["numsensorson"] / num_gridsquares
        self._X_1_off[:,:,idx] = max(s_summarized["numsensorson"] - 1, 0) / num_gridsquares
        idx += 1
        #  4. # of sources found
        self._X_1_on[:,:,idx] = s_summarized["numsourcesfound"] * self._X_1_on[:,:,idx-1]
        self._X_no_change[:,:,idx] = s_summarized["numsourcesfound"] * self._X_no_change[:,:,idx-1]
        self._X_1_off[:,:,idx] = s_summarized["numsourcesfound"] * self._X_1_off[:,:,idx-1]
        idx += 1
        #  5. whether each square will be 2-covered (0/1) 
        #       x time-discounted observation history of that square
        self._X_1_on[:,:,idx] = ((cov_cnt_map + 1) >= 2) * s_summarized["timediscountedobshistory"]
        self._X_no_change[:,:,idx] = (cov_cnt_map >= 2) * s_summarized["timediscountedobshistory"]
        self._X_1_off[:,:,idx] = ((cov_cnt_map - 1) >= 2) * s_summarized["timediscountedobshistory"]
        idx += 1
        #  6. sum of: whether each square will be 3-covered (0/1) 
        #       x "                                                "
        self._X_1_on[:,:,idx] = ((cov_cnt_map + 1) >= 3) * s_summarized["timediscountedobshistory"]
        self._X_no_change[:,:,idx] = (cov_cnt_map >= 3) * s_summarized["timediscountedobshistory"]
        self._X_1_off[:,:,idx] = ((cov_cnt_map - 1) >= 3) * s_summarized["timediscountedobshistory"]
        idx += 1
        #  7. whether each square will be 2-covered (0/1)
        #       x whether a source has been found or a sensor exists in each square (0/1)
        self._X_1_on[:,:,idx] = ((cov_cnt_map + 1) >= 2) * s_summarized["sourcesensorlocs"]
        self._X_no_change[:,:,idx] = (cov_cnt_map >= 2) * s_summarized["sourcesensorlocs"]
        self._X_1_off[:,:,idx] = ((cov_cnt_map - 1) >= 2) * s_summarized["sourcesensorlocs"]
        idx += 1
        # 8. whether each square will be 3-covered (0/1) 
        #       x whether a source has been found or a sensor exists in each square (0/1)
        self._X_1_on[:,:,idx] = ((cov_cnt_map + 1) >= 3) * s_summarized["sourcesensorlocs"]
        self._X_no_change[:,:,idx] = (cov_cnt_map >= 3) * s_summarized["sourcesensorlocs"]
        self._X_1_off[:,:,idx] = ((cov_cnt_map - 1) >= 3) * s_summarized["sourcesensorlocs"]
        # return:
        return self._X_1_on, self._X_no_change, self._X_1_off
    
    
    def _create_feat_vec_from_mat(self, X, state_summarized, action):
        x = np.sum(X, axis=(0,1))
        snsr_num, turn_on = self.env.get_action_info(action)
        # num sensors on
        num_snsr_on = state_summarized["numsensorson"]
        x[3] = num_snsr_on
        if not action is self.env.NO_OP_ACTION:
            if turn_on:
                x[3] += 1
            else:
                x[3] -= 1
        # num of sources found  x  num of sensors on
        num_srcs_on = state_summarized["numsourcesfound"]
        x[4] = num_srcs_on * x[3]
        return x / self._scaling_factors










class DotProductModel(nn.Module):
    
    def __init__(self, length, smart_init=True):
        super().__init__()
        if smart_init and length == 9:
            self.weights = torch.nn.Parameter(torch.tensor([ 
                10.59154438,
                6.49373517,
                6.33886868,
                46.2918886,
                -204.60743457,
                54.47196403,
                49.93853715,
                -18.88485571,
                -26.46835565
            ], requires_grad=True))
        else:
            self.weights = torch.nn.Parameter(torch.rand(length), requires_grad=True)
        
    
    def forward(self, X):
        X_tnsr = torch.from_numpy(X).float()
        return torch.dot(X_tnsr, self.weights)




class Q_FeatureVector:
    
    def __init__(self, feat_vec: FeatureVector):
        self.feat_vec = feat_vec
        self.model = DotProductModel(self.feat_vec.get_feature_vector_len())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_all_action_values(self, state):
        act_vals = []
        for x in self.feat_vec.get_all_feature_vectors(state):
            act_val = self.model(x).detach().numpy()
            act_vals.append(act_val)
        return np.array(act_vals)
    
    def update(self, state, target, action):
        self.optimizer.zero_grad()
        target_tnsr = torch.Tensor([target]).squeeze()
        feature_vec = self.feat_vec(state, action)
        predicted_tnsr = self.model(feature_vec)
        loss = self.criterion(predicted_tnsr, target_tnsr)
        loss.backward()
        self.optimizer.step()
    
    def parse_new_state_info(self, raw_state, info):
        return self.feat_vec.parse_new_state_info(raw_state, info)
    
    
    
    