import numpy as np
import gym
import torch
import torch.optim as optim
from torch import nn
from environment import Environment_v0, SENSOR_STATUSES, LOCATED_SOURCES, SENSOR_COVERAGES
from agent import Agent


class SarsaAgent(Agent):

    def __init__(
            self,
            env: gym.Env,
            gamma: float = 0.9,    # Discount factor
            lam: float = 0.01,     # Decay Rate
            alpha : float = 0.01,  # Step size
            epsilon: float = 0.,   # Epsilon greedy param
    ):
        super(SarsaAgent, self).__init__(env)

        self.env = env
        self.num_actions = env.action_space.n
        self.num_sensors = env._num_sensors
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epsilon = epsilon

    def train(self, num_episodes: int):

        # Pass in sensors on/off only for now
        w = np.zeros(self.num_sensors)

        for episode in range(num_episodes):
            state = env.reset()
            z = np.zeros(self.num_sensors)
            sensor_state = state[SENSOR_STATUSES]
            action = self.get_action(w, sensor_state, self.epsilon)

            while True:
                next_state, reward, done, _  = env.step(action)
                env.render()
                sensor_state = state[SENSOR_STATUSES]
                delta = reward - np.dot(w, sensor_state)
                print('Delta is {}'.format(delta))
                z += sensor_state

                if done:
                    w += self.alpha * delta * z
                    break

                next_sensor_state = next_state[SENSOR_STATUSES]
                next_action = self.get_action(w, next_sensor_state, self.epsilon)
                delta += self.gamma * np.dot(w, next_sensor_state)
                w += self.alpha * delta * z
                z = self.gamma * self.lam * z
                state = next_state
                action = next_action

    # E-greedy
    def get_action(
            self,
            w: np.array,     # Weight vector
            s: np.array,     # Sensor statuses
            epsilon: float,  # eplison greedy value
    ) -> int:

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        all_actions = self.get_all_valid_actions(s)
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

    def get_all_valid_actions(
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

if __name__ == '__main__':
    env = Environment_v0()
    agent = SarsaAgent(env)
    agent.train(100)
