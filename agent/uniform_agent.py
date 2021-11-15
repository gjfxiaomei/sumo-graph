from itertools import cycle
class UniformAgent():
    def __init__(self,action_size):
        self.action_size = action_size
        self.phase_cycle = cycle(list(range(self.action_size)))
        
    def get_action(self,state):
        return next(self.phase_cycle)
    
    def predict(self,state):
        return next(self.phase_cycle)