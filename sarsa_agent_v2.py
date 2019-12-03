import numpy as np
import gym
from environment import Environment_v1, SENSOR_STATUSES, LOCATED_SOURCES
from agent import Agent
from handcrafted_features import HandCraftedFeatureVector, FeatureVector
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

'''
SarsaAgentV1 differs from SarsaAgentV0 by taking the located sources into consideration
'''


class SarsaAgentV2(Agent):

    def __init__(
            self,
            env: gym.Env,
            X: FeatureVector,
            gamma: float = 0.9,    # Discount factor
            lam: float = 0.01,     # Decay Rate
            alpha: float = 0.01,  # Step size
            epsilon: float = 1.,
            epsilon_decay: float = 0.999,
            min_epsilon: float = .01,
            debug: bool = False
    ):
        super().__init__(env)

        self.env = env
        self.num_actions = env.action_space.n
        self.num_sensors = env._num_sensors
        self.field_size = env.field_size
        self.X = X
        self.w = np.zeros(self.X.get_feature_vector_len())
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episode_num = 0
        self._debug = debug
        self._previously_saved_w = None

    def train(self, num_episodes: int):
        
        self.load()

        start_episode = self.episode_num
        while self.episode_num - start_episode < num_episodes:
            if self.episode_num % 100 == 0 and self._debug:
                print('Episode #{}'.format(self.episode_num))

            raw_state = env.reset()
            state = self.X.parse_new_state_info(raw_state, None)
            z = np.zeros(self.X.get_feature_vector_len())
            action = self.get_action(self.w, state, self.epsilon)
            x = self.X(state, action)
            Q_old = 0.

            done = False
            while not done:
                raw_next_state, reward, done, info = env.step(action)
                next_state = self.X.parse_new_state_info(raw_next_state, info)
                # env.render()
                next_action = self.get_action(self.w, next_state, self.epsilon)
                next_x = self.X(next_state, next_action)
                Q = np.dot(self.w, x)
                next_Q = np.dot(self.w, next_x)
                delta = reward + self.gamma * next_Q - Q
                z = self.gamma*self.lam*z + (1 - self.alpha*self.gamma*self.lam*np.dot(z,x)) * x
                self.w += self.alpha*(delta + Q - Q_old)*z - self.alpha*(Q - Q_old)*x
                Q_old = next_Q
                x = next_x
                action = next_action
            
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
            self.episode_num += 1
            if self.episode_num % 100 == 0:
                self.save()
    

    # E-greedy
    def get_action(
            self,
            w: np.array,     # Weight vector
            state: dict,     # Sensor statuses
            epsilon: float,  # eplison greedy value
    ) -> int:

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        valid_actions = self._get_all_valid_actions(state[SENSOR_STATUSES])
        all_feat_vecs = self.X.get_all_feature_vectors(state)

        Q = [np.dot(w, all_feat_vecs[action]) for action in valid_actions]

        # The idx corresponds to the idx for the action in all_actions
        action_idx = np.argmax(Q)
        return valid_actions[action_idx]

    def _get_all_valid_actions(
            self,
            s: np.array,
    ) -> np.array:
        actions = []

        # Iterate through sensors and add valid actions
        for i in range(len(s)):
            if s[i] == 0:
                actions.append(i)       # Switch on if off
            else:
                actions.append(i + 20)  # Switch off if on

        # Append no-op action
        actions.append(40)

        return np.array(actions)

    def run_episode(self, render=False):
        self.load()
        rewards = []
        raw_state = self.env.reset()
        state = self.X.parse_new_state_info(raw_state, None)
        done = False
        if self._debug and render:
            video_recorder = VideoRecorder(self.env, path="/tmp/gym_%s_run.mp4"%self.get_model_filename())
            video_recorder.capture_frame()
        while not done:
            # Use a greedy policy with the values learnt from training
            a = self.get_action(self.w, state, 0.)
            raw_next_state, reward, done, info = self.env.step(a)
            if self._debug and render:
                env.render()
            next_state = self.X.parse_new_state_info(raw_next_state, info)
            rewards.append(reward)
            state = next_state
            if self._debug and render:
                video_recorder.capture_frame()
        if self._debug and render:
            video_recorder.close()
        return rewards
    
    
    def get_model_filename(self):
        return "SarsaAgentV2"
    
    
    def save(self):
        # Saves Q function state
        torch.save({
            'episode_num' : self.episode_num,
            'epsilon' : self.epsilon,
            'w' : self.w
        }, "./data/models/" + self.get_model_filename() + ".model")
        if self._debug:
            if not self._previously_saved_w is None:
                print("\t\t\t\t\t\t max diff in w since last save:", np.max(np.abs(self.w-self._previously_saved_w)))
            print("Saved w:", self.w)
            print("Saved epsilon:", self.epsilon)
            self._previously_saved_w = self.w.copy()
    
    
    def load(self):
        # Loads Q function state
        try:
            checkpoint = torch.load("./data/models/" + self.get_model_filename() + ".model")
            self.episode_num = checkpoint['episode_num']
            self.epsilon = checkpoint['epsilon']
            self.w = checkpoint['w']
            if self._debug:
                print("Loaded w:", self.w)
                print("Loaded epsilon:", self.epsilon)
        except FileNotFoundError:
            if self._debug:
                print("Did not load a model.")








if __name__ == '__main__':
    env = Environment_v1(debug=False)
    agent = SarsaAgentV2(env, HandCraftedFeatureVector(env), debug=True)#, epsilon=.5, epsilon_decay=1.)
    agent.train(20000)
 
    env._debug = False
    returns = []
    num_eps = 100
    for i in range(num_eps):
        ep_rewards = agent.run_episode(False)
        print("TOTAL:", sum(ep_rewards))
        returns.append(sum(ep_rewards))
    print("SarsaAgentV2, %d episodes:" % num_eps)
    print("Return:    mean:%.2f    std:%.2f" % (np.mean(returns), np.std(returns)))
