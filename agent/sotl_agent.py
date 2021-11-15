from itertools import cycle
class SotlAgent():
    def __init__(self, action_size):
        self.action_size = action_size
        # g_min
        self.phi = 15
        self.min_green_vehicle = 20
        self.max_red_vehicle = 1

        self.phase_cycle = cycle(list(range(self.action_size)))
        self.action = next(self.phase_cycle)

    def choose_action(self, state):
        # stay in green phase for minimum amout of time
        if state['phase_time'] >= self.phi:
            if state['ngv'] <= self.min_green_vehicle and state['nrv'] >= self.max_red_vehicle:
                self.action = next(self.phase_cycle)
        return self.action
