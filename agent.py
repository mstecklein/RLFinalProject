class Agent:
    def __init__(self, env):
        self.env = env

    def get_name(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def run_episode(self):
        raise NotImplementedError()

    def save_model(self, path: str):
        raise NotImplementedError()

    def load_model(self, path: str):
        raise NotImplementedError()
