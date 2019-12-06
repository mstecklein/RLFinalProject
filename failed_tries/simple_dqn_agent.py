



class SimpleDQNAgent(DQNAgentAbstract):
    """
        Only looks at first two components of the state space:
         - SENSOR_STATUSES
         - LOCATED_SOURCES
        And uses a regular NN for the action value function Q.
    """
    
    def __init__(self, env, **args):
        args["name"] = "SimpleDQN"
        super().__init__(env, **args)
        
        self._input_len = env.observation_space.spaces[SENSOR_STATUSES].n + \
                          env.observation_space.spaces[LOCATED_SOURCES].n
        
        self.model = nn.Sequential(
                    nn.Linear(self._input_len, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, self.env.action_space.n)) 
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                     lr=0.0001,
                                     betas=(0.9, 0.999))
    
    def Q(self, s):
        self.model.eval()
        input = torch.FloatTensor(self._flatten_state(s))
        output = self.model(input)
        return output.detach().numpy()
    
    
    def _flatten_state(self, s):
        return np.concatenate((
            s[SENSOR_STATUSES].flatten(),
            s[LOCATED_SOURCES].flatten()
        ))
    
    
    def update(self, state, target, action):
        self.optimizer.zero_grad()
        target_all_actions = self.Q(state)
        target_all_actions[action] = target
        target_tnsr = torch.Tensor([target_all_actions]).squeeze()
        predicted_tnsr = self.model(torch.FloatTensor(self._flatten_state(state)))
        loss = self.criterion(predicted_tnsr, target_tnsr)
        loss.backward()
        self.optimizer.step()
    
    
    def save(self):
        torch.save({
            'episode_num' : self.episode_num,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'epsilon' : self.epsilon
        }, self.get_model_filename())
    
    
    def load(self):
        try:
            checkpoint = torch.load(self.get_model_filename())
            self.episode_num = checkpoint['episode_num']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
        except FileNotFoundError:
            pass
    
    
    def summarize_history(self, raw_state, info=None):
        del raw_state[SENSOR_COVERAGES]
        return raw_state
