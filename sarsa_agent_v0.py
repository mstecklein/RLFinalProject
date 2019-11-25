import numpy as np
import gym
from environment import Environment_v0, SENSOR_STATUSES
from agent import Agent


class SarsaAgentV0(Agent):

    def __init__(
            self,
            env: gym.Env,
            gamma: float = 0.9,    # Discount factor
            lam: float = 0.01,     # Decay Rate
            alpha: float = 0.01,  # Step size
            epsilon: float = 0.,   # Epsilon greedy param
    ):
        super(SarsaAgentV0, self).__init__(env)

        self.env = env
        self.num_actions = env.action_space.n
        self.num_sensors = env._num_sensors
        self.w = np.zeros(self.num_sensors)
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epsilon = epsilon

    def train(
            self,
            num_episodes: int
    ):

        # Pass in sensors on/off only for now
        w = np.zeros(self.num_sensors)

        for episode in range(num_episodes):
            if episode % 100 == 0:
                print('Episode #{}'.format(episode))

            state = env.reset()
            z = np.zeros(self.num_sensors)
            action = self.get_action(w, state, self.epsilon)

            while True:
                next_state, reward, done, _ = env.step(action)
                # env.render()
                sensor_state = state[SENSOR_STATUSES]
                delta = reward - np.dot(w, sensor_state)
                z += sensor_state

                if done:
                    w += self.alpha * delta * z
                    break

                next_action = self.get_action(w, next_state, self.epsilon)
                next_sensor_state = next_state[SENSOR_STATUSES]
                delta += self.gamma * np.dot(w, next_sensor_state)
                w += self.alpha * delta * z
                z = self.gamma * self.lam * z
                state = next_state
                action = next_action

        self.w = w

    # E-greedy
    def get_action(
            self,
            w: np.array,     # Weight vector
            state: dict,     # Sensor statuses
            epsilon: float,  # eplison greedy value
    ) -> int:

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        s = state[SENSOR_STATUSES]
        all_actions = self._get_all_valid_actions(s)
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

    def run_episode(
            self
    ) -> np.array:
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
    agent = SarsaAgentV0(env, epsilon=0.1)
    agent.train(1000)

    for i in range(10):
        episode_rewards = agent.run_episode()
        print('The episode took {} steps and had a reward of {}'.format(len(episode_rewards), sum(episode_rewards)))
