from itertools import cycle
class PFAgent():
    
class UniformAgent():
    def __init__(self):
        self.phase_cycle = cycle([0,2])
        
    def get_action(self,state,mode):
        return next(self.phase_cycle)