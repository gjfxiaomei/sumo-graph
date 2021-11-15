import os, sys
import numpy as np
import math
import queue

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class TrafficMetrics:
    def __init__(self, _id, incoming_lanes, netdata, metric_args, mode):
        self.metrics = {}
        if 'delay' in metric_args:
            lane_lengths = {lane:netdata['lane'][lane]['length'] for lane in incoming_lanes}
            lane_speeds = {lane:netdata['lane'][lane]['speed'] for lane in incoming_lanes}
            self.metrics['delay'] = DelayMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds )

        if 'queue' in metric_args:
            self.metrics['queue'] = QueueMetric(_id, incoming_lanes, mode)
        
        if 'log_queue' in metric_args:
            self.metrics['log_queue'] = LogQueueMetric(_id, incoming_lanes, mode)

        if 'virtual_queue' in metric_args:
            self.metrics['virtual_queue'] = VirtualQueueMetric(_id, incoming_lanes, mode)
        
        if 'throughput' in metric_args:
            self.metrics['throughput'] = ThroughputMetric(_id, incoming_lanes, mode)

    def update(self, previous_action_lanes, v_data):
        for m in self.metrics:
            self.metrics[m].update(previous_action_lanes, v_data)

    def get_metric(self, metric, previous_action_lanes):
        return self.metrics[metric].get_metric(previous_action_lanes)

    def get_history(self, metric):
        return self.metrics[metric].get_history()
    
    def get_ave_throughput(self):
        return self.metrics['throughput'].get_ave_throughput()

class TrafficMetric:
    def __init__(self, _id, incoming_lanes, mode):
        self.id = _id
        self.incoming_lanes = incoming_lanes
        self.history = []
        self.ave_lane_throughput = {}
        self.mode = mode
    def get_metric(self, previous_action_lanes):
        pass

    def update(self, previous_action_lanes):
        pass

    def get_history(self):
        return self.history
    
    def get_ave_throughput(self):
        return self.ave_lane_throughput

class DelayMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        super().__init__( _id, incoming_lanes, mode)
        self.lane_travel_times = {lane:lane_lengths[lane]/float(lane_speeds[lane]) for lane in incoming_lanes}
        self.old_v = set()
        self.v_info = {}
        self.t = 0

    def get_v_delay(self, v):
        return ( self.t - self.v_info[v]['t'] ) - self.lane_travel_times[self.v_info[v]['lane']]

    def get_metric(self, previous_action_lanes):
        #calculate delay of vehicles on incoming lanes
        delay = 0
        delay_vec = []
        for v in self.old_v:
            #calculate individual vehicle delay
            v_delay = self.get_v_delay(v)
            delay_vec.append(v_delay)
            if v_delay > 0:
                delay += v_delay
        return delay

    def update(self, previous_action_lanes, v_data):
        new_v = set()

        #record start time and lane of new_vehicles
        for lane in self.incoming_lanes:
            for v in v_data[lane]:
                if v not in self.old_v:
                    self.v_info[v] = {}
                    self.v_info[v]['t'] = self.t
                    self.v_info[v]['lane'] = lane
            new_v.update( set(v_data[lane].keys()) )

        if self.mode == 'test':
            self.history.append(self.get_metric())

        #remove vehicles that have left incoming lanes
        remove_vehicles = self.old_v - new_v
        delay = 0
        for v in remove_vehicles:
            del self.v_info[v]
        
        self.old_v = new_v
        self.t += 1

class ThroughputMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.phase_lanes = {'gGGgrrgrrgrr': ['N2TL_1', 'N2TL_2'], 'gGrgrrgGrgrr': ['S2TL_1', 'N2TL_1'], 'grGgrrgrGgrr': ['S2TL_2', 'N2TL_2'], 'grrgGGgrrgrr': ['E2TL_1', 'E2TL_2'], 'grrgGrgrrgGr': ['E2TL_1', 'W2TL_1'], 'grrgrGgrrgrG': ['W2TL_2', 'E2TL_2'], 'grrgrrgGGgrr': ['S2TL_1', 'S2TL_2'], 'grrgrrgrrgGG': ['W2TL_2', 'W2TL_1']}
        self.alpha = 0.1
        self.stop_speed = 0.3
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}
        self.previous_phase = 'r'*12
        self.lane_throughput = {lane : 0 for lane in self.incoming_lanes}
        self.history_lane_vehicles = {lane : set() for lane in self.incoming_lanes}
        self.ave_lane_throughput = {lane:0 for lane in self.incoming_lanes}
        self.his_lane_throughput = {lane:[] for lane in self.incoming_lanes}
        self.window_size = 4
        self.msr = 0.5

    def indicator(self, lane, phase_lanes):
        if lane in phase_lanes:
            return 1
        else:
            return 0


    # get_metric means this phase is over, we can get the throughput of this phase.
    def get_metric(self, previous_action_lanes):
        # self.his_lane_throughput.put(self.lane_throughput)
        phase_lanes = list(previous_action_lanes.values())[0]
        # print(self.lane_queues)
        for lane in self.incoming_lanes:
            # self.lane_queues[lane] = np.max([0,self.lane_queues[lane] + self.msr - self.indicator(lane,previous_action_lanes)])
            self.his_lane_throughput[lane].append(self.lane_throughput[lane])
            # self.ave_lane_throughput[lane] = (1-1.0/self.window_size)*self.ave_lane_throughput[lane] + (1.0/self.window_size)*self.lane_throughput[lane]
            self.ave_lane_throughput[lane] = sum(self.his_lane_throughput[lane][-self.window_size:])
        # print(self.lane_queues)

        weighted_sum_queues = sum([self.lane_queues[lane]/(self.ave_lane_throughput[lane] + 1) for lane in self.incoming_lanes])
        self.lane_throughput = {lane:0 for lane in self.incoming_lanes}
        
        return weighted_sum_queues

    def update(self, previous_action_lanes, v_data):
        for lane in self.incoming_lanes:
            new_lane_vehicles = set()
            new_lane_vehicles.update(v_data[lane].keys())
            leave_set = self.history_lane_vehicles[lane].difference(new_lane_vehicles)
            self.lane_throughput[lane] += len(leave_set)
            self.history_lane_vehicles[lane] = new_lane_vehicles

        
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1
        self.lane_queues = lane_queues

        
class QueueMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.3 
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}

    def get_metric(self, previous_action_lanes):
        # print(self.phase_lane_queues)
        # return sum([self.phase_lane_queues[phase] for phase in self.phase_lanes.keys()])
        return sum([self.lane_queues[lane] for lane in self.lane_queues])
        
    def update(self, previous_action_lanes, v_data):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1
        self.lane_queues = lane_queues
        if self.mode == 'test':
            self.history.append(self.get_metric())

class VirtualQueueMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.3 
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}
        
    def indictor(self, lane):
        if lane in self.previous_action_lanes:
            return 1
        else:
            return 0

    def get_metric(self, previous_action_lanes):
        # print(self.lane_queues)
        #Record that which lanes were served in last process.
        # print(self.lane_queues)
        # Record that which lanes were served in last process.
        previous_action_lanes = list(previous_action_lanes.values())[0]
        self.previous_action_lanes = previous_action_lanes
        for lane in self.lane_queues:
            self.lane_queues[lane] = max(self.lane_queues[lane]  - self.indictor(lane),0)
        print(previous_action_lanes)
        return sum([self.lane_queues[lane] for lane in self.lane_queues])
        
    def update(self, previous_action_lanes, v_data):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1

        # for lane in self.lane_queues:
        #     self.lane_queues[lane] = max(,0)

        self.lane_queues = lane_queues
        if self.mode == 'test':
            self.history.append(self.get_metric())

class LogQueueMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.3 
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}

    def get_metric(self, previous_action_lanes):
        # print(self.lane_queues)
        return sum([np.log(self.lane_queues[lane] + 1) for lane in self.lane_queues])
        
    def update(self, previous_action_lanes, v_data):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1

        self.lane_queues = lane_queues
        if self.mode == 'test':
            self.history.append(self.get_metric())