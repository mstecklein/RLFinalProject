import numpy as np
import gym
from environment import Environment_v0, SENSOR_STATUSES, LOCATED_SOURCES
from agent import Agent

'''
SarsaAgentV1 differs from SarsaAgentV0 by taking the located sources into consideration
'''


class SarsaAgentV1(Agent):

    def __init__(
            self,
            env: gym.Env,
            gamma: float = 0.9,    # Discount factor
            lam: float = 0.01,     # Decay Rate
            alpha: float = 0.01,  # Step size
            epsilon: float = 0.,   # Epsilon greedy param
    ):
        super(SarsaAgentV1, self).__init__(env)

        self.env = env
        self.num_actions = env.action_space.n
        self.num_sensors = env._num_sensors
        self.field_size = env.field_size
        self.w = np.zeros(self.num_sensors + self.field_size)
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epsilon = epsilon

    def train(self, num_episodes: int):

        # Pass in sensors on/off only for now
        w = np.zeros(self.num_sensors + self.field_size)

        for episode in range(num_episodes):
            if episode % 100 == 0:
                print('Episode #{}'.format(episode))

            state = env.reset()
            z = np.zeros(self.num_sensors + self.field_size)
            action = self.get_action(w, state, self.epsilon)

            while True:
                next_state, reward, done, _ = env.step(action)
                # env.render()
                s = self._get_feature_vector(state)
                delta = reward - np.dot(w, s)
                z += s

                if done:
                    w += self.alpha * delta * z
                    break

                next_s = self._get_feature_vector(next_state)
                delta += self.gamma * np.dot(w, next_s)
                w += self.alpha * delta * z
                z = self.gamma * self.lam * z
                state = next_state
                next_action = self.get_action(w, next_state, self.epsilon)
                action = next_action

        self.w = w

    def _get_feature_vector(
            self,
            state: dict
    ) -> np.array:
        sensor_state = state[SENSOR_STATUSES]
        located_state = state[LOCATED_SOURCES].flatten()
        return np.append(sensor_state, located_state)

    # E-greedy
    def get_action(
            self,
            w: np.array,     # Weight vector
            state: dict,     # Sensor statuses
            epsilon: float,  # eplison greedy value
    ) -> int:

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        all_actions = self._get_all_valid_actions(state[SENSOR_STATUSES])
        s = self._get_feature_vector(state)
        all_future_states = []
        for action in all_actions:
            cp = np.copy(s)

            # Check if no-op
            if action < 40:
                a = action % 20
                cp[a] = (cp[a] + 1) % 2  # Flip the bit

            all_future_states.append(cp)

        Q = [np.dot(w, future_state) for future_state in all_future_states]

        # The idx corresponds to the idx for the action in all_actions
        action_idx = np.argmax(Q)
        return all_actions[action_idx]

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

    def run_episode(self):
        rewards = []
        state = self.env.reset()
        done = False
        while not done:
            # Use a greedy policy with the values learnt from training
            a = self.get_action(self.w, state, 0.)
            next_state, reward, done, _ = self.env.step(a)
            rewards.append(reward)
            state = next_state
        return rewards



if __name__ == '__main__':
    env = Environment_v0()
    agent = SarsaAgentV1(env, epsilon=0.1)
    agent.train(1000)

    for i in range(10):
        episode_rewards = agent.run_episode()
        print('The episode took {} steps and had a reward of {}'.format(len(episode_rewards), sum(episode_rewards)))
