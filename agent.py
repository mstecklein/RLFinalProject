class Agent:
    def __init__(self, env):
        self.env = env

    def get_name(self, env):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def run_episode(self):
        raise NotImplementedError()

    def save_model(self, path):
        raise NotImplementedError()

    def load_model(self, path):
        raise NotImplementedError()
