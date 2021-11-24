import os, sys
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from utils import list_of_groups
from numpy.lib import utils
from trafficmetrics import TrafficMetrics
from roadnet_reader import RoadnetReader
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import dgl
import traci
import torch as th


#intract with sumo via traci
#only consider one intersection
class SumoAgent:
    def __init__(self, args, roadnet, mode, red_t, yellow_t):
        self.args = args
        self.netdata_reader = RoadnetReader(roadnet)
        self.netdata = self.netdata_reader.get_net_data()
        # print(self.netdata['inter'])
        # graph_road_set = set()
        # for inter in self.netdata['inter']:
        #     for value in self.netdata['inter'][inter]['incoming']:
        #         graph_road_set.add(value)
        #     for value in self.netdata['inter'][inter]['outgoing']:
        #         graph_road_set.add(value)
        # graph_road_list = list(graph_road_set)
        # # print(graph_road_list)
        # u = []
        # v = []
        # for inter in self.netdata['inter']:
        #     for lane in self.netdata['inter'][inter]['tlsindex'].values():
        #         s = lane.split('_')[0]
        #         e = list(self.netdata['lane'][lane]['outgoing'].keys())[0].split('_')[0]
        #         u.append(graph_road_list.index(s))
        #         v.append(graph_road_list.index(e))
        # print(u)
        # print(v)
        # g = dgl.graph((th.tensor(u), th.tensor(v)))

        self.mode = mode
        self.tl_id = "TL"
        self.red_t = red_t
        self.yellow_t = yellow_t
        self.green_t = args.green_duration
        self.initilize = False
        self.sumoCmd = self.get_sumoCmd(args.gui, args.roadnet, args.max_steps)
        print(self.sumoCmd)
        # print(self.netdata['lane'])
        self.start()
        
    #add incoming
    def update_netdata(self):
        self.netdata['inter'][self.tl_id]['incoming_lanes'] = self.incoming_lanes
        self.netdata['inter'][self.tl_id]['green_phases'] = self.green_phases

    def get_subscription_data(self):
        tl_data = traci.junction.getContextSubscriptionResults(self.tl_id)
        lane_vehicles = {l:{} for l in self.incoming_lanes}
        if tl_data is not None:
            for v in tl_data:
                lane = tl_data[v][traci.constants.VAR_LANE_ID]
                if lane not in lane_vehicles:
                    lane_vehicles[lane] = {}
                lane_vehicles[lane][v] = tl_data[v]
        return lane_vehicles
    
    def get_phase_representation(self, phase):
        w = []
        for s in phase:
            #TODO: g和G的权重可能不同
            if s=='g' or s=='G':
                w.append(1)
            else:
                w.append(0)
        return w
    
    def get_state_size(self):
        if self.args.metric == "queue":
            return len(self.incoming_lanes)*2 + len(self.netdata['inter'][self.tl_id]['green_phases'])
        elif self.args.metric == "throughput":
            # return self.get_action_size()*2 + len(self.netdata['inter'][self.tl_id]['green_phases'])
            return len(self.incoming_lanes)*2 + len(self.netdata['inter'][self.tl_id]['green_phases'])
    
    def get_action_size(self):
        return len(self.netdata['inter'][self.tl_id]['green_phases'])

    def get_tl_green_phases(self):
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_id)[0]
        #get only the green phases
        green_phases = [ p.state for p in logic.getPhases() 
                         if 'y' not in p.state 
                         and ('G' in p.state or 'g' in p.state) ]

        #sort to ensure parity between sims (for RL actions)
        return sorted(green_phases)
    
    def get_phase_lanes(self, actions):
        phase_lanes = {a:[] for a in actions}
        for a in actions:
            green_lanes = set()
            red_lanes = set()
            for s in range(len(a)):
                # if a[s] == 'g' or a[s] == 'G': not consider the right-turn lane.
                if a[s] == 'G' or a[s]=='g':
                    green_lanes.add(self.netdata['inter'][self.tl_id]['tlsindex'][s])
                elif a[s] == 'r':
                    red_lanes.add(self.netdata['inter'][self.tl_id]['tlsindex'][s])

            ###some movements are on the same lane, removes duplicate lanes
            pure_green = [l for l in green_lanes if l not in red_lanes]
            if len(pure_green) == 0:
                phase_lanes[a] = list(set(green_lanes))
            else:
                phase_lanes[a] = list(set(pure_green))
        # print(phase_lanes)
        return phase_lanes
    
    def input_to_one_hot(self, phases):
        identity = np.identity(len(phases))                                 
        one_hots = { phases[i]:identity[i,:]  for i in range(len(phases)) }
        return one_hots

    def int_to_input(self, phases):
        return { p:phases[p] for p in range(len(phases)) }

    def get_state(self):
        if self.args.tsc == "sotl":
            # current_phase
            # current_phase_time
            gv,rv = self.get_green_red_vehicles()
            state = {'phase_time':self.phase_time,'ngv':gv,'nrv':rv}
            return state
        if self.args.metric == "queue":
            # return np.concatenate( [np.concatenate([self.get_queue(), self.get_volume()])] ), self.current_phase
            return np.concatenate( [np.concatenate([self.get_incoming_queue(), self.get_incoming_volume()]), self.phase_to_one_hot[self.current_phase]] ), self.current_phase
        elif self.args.metric == "throughput":
            return np.concatenate( [np.concatenate([self.get_ave_throughput(),self.get_incoming_volume()]), self.phase_to_one_hot[self.current_phase]] ), self.current_phase
        
    def generate_graph(self, state, phase):
        #incoming_lanes = ['E2TL_1', 'E2TL_2', 'N2TL_1', 'N2TL_2', 'S2TL_1', 'S2TL_2', 'W2TL_1', 'W2TL_2']
        #对于一个phase: gGGgrrgrrgrr
        #按照 N-E-S-W 的顺序
        # N2TL_(0,1,2)-E2TL_(0,1,2)-S2TL(0,1,2)-W2TL_(0,1,2) 
        #右车道是0，中间是1，左车道是2
        #state也是根据这个顺序
        u, v = th.tensor([N_IN, N_IN, N_IN, E_IN, E_IN, E_IN, S_IN, S_IN, S_IN, W_IN, W_IN, W_IN]), th.tensor([W_OUT, S_OUT, E_OUT, N_OUT, W_OUT, S_OUT, E_OUT, N_OUT, W_OUT, S_OUT, E_OUT, N_OUT])
        connected_edges_from = []
        connected_edges_to = []
        stuck_edges_from = []
        stuck_edges_to = []
        
        for i, s in enumerate(phase):
            #TODO: g和G的权重可能不同
            if s=='g' or s=='G':
                connected_edges_from.append(u[i])
                connected_edges_to.append(v[i])
            else:
                stuck_edges_from.append(u[i])
                stuck_edges_to.append(v[i])
        g = dgl.heterograph({
            ('road', 'connected', 'road'): (connected_edges_from, connected_edges_to),
            ('road', 'stuck', 'road'): (stuck_edges_from, stuck_edges_to)
        })
        return g
        

    def get_intermediate_phases(self, phase, next_phase):
        if phase == next_phase or phase == self.all_red:
            return []
        else:
            # turn green light to yellow(yellow phase)
            yellow_phase = ''.join([ p if p == 'r' else 'y' for p in phase ])
            return [yellow_phase, self.all_red]

    def phase_duration(self):
        if self.current_phase in self.green_phases:
            return self.green_t
        elif 'y' in self.current_phase:
            return self.yellow_t
        else:
            return self.red_t
    
    def get_phase_volume(self):
        if self.data != None:
            phase_volume = []
            for phase in self.phase_lanes.keys():
                v = 0
                for lane in self.phase_lanes[phase]:
                    v += len(self.data[lane])
                phase_volume.append(v)
            return np.array(phase_volume)
        else:
            return np.array([0]*len(self.phase_lanes.keys()))
    
    def get_incoming_volume(self):
        #number of vehicles in each incoming lane
        if self.data != None:
            return np.array([len(self.data[lane]) for lane in self.incoming_lanes])
        else:
            return np.array([0]*len(self.incoming_lanes))
    
    def get_outgoing_volume(self):
        #number of vehicles in each outgoing lane
        if self.data != None:
            volume = []
            for lane in self.outgoing_lanes:
                if lane in self.data:
                    volume.append(len(self.data[lane]))
                else:
                    volume.append(0)
            return np.array(volume)
        else:
            return np.array([0]*len(self.outgoing_lanes))
        
    
    def get_ave_throughput(self):
        ave_throughput = np.array(list(self.trafficmetrics.get_ave_throughput().values()))
        # print(ave_throughput)
        return ave_throughput

    def get_phase_queue(self):
        if self.data != None:
            phase_lane_queues = []
            for phase in self.phase_lanes.keys():
                q = 0
                for lane in self.phase_lanes[phase]:
                    for v in self.data[lane]:
                        if self.data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                            q+=1
                phase_lane_queues.append(q)
        else:
            phase_lane_queues = [0]*len(self.phase_lanes.keys())
        return np.array(phase_lane_queues)
    
    def get_green_red_vehicles(self):
        if self.data != None:
            lane_queues = {lane:0 for lane in self.incoming_lanes}
            for lane in self.incoming_lanes:
                q = 0
                for v in self.data[lane]:
                    if self.data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                        q+=1
                lane_queues[lane] = q
            phase_lanes = self.phase_lanes[self.current_phase]
            # print(phase_lanes)
            green_vehicles = sum([lane_queues[lane] for lane in phase_lanes])
            # print(lane_queues)
            red_vehicles = sum(list(lane_queues.values())) - green_vehicles
            return green_vehicles, red_vehicles
        else:
            return 0,0

    def get_incoming_queue(self):
        if self.data != None:
            lane_queues = []
            for lane in self.incoming_lanes:
                q = 0
                for v in self.data[lane]:
                    if self.data[lane][v][traci.constants.VAR_SPEED] < 0.3:
                        q+=1
                lane_queues.append(q)
        else:
            lane_queues = [0]*len(self.incoming_lanes)
        return np.array(lane_queues)
    
    
    def get_volume_node(self):
        volume = []
        volume.extend(self.get_incoming_volume())
        volume.extend(self.get_outgoing_volume())
        volume = list_of_groups(volume, per_list_len=3)
        return np.array(volume)
    
    def simulate_action(self,action):
        if self.args.tsc == "sotl":
            next_phase = self.int_to_phase[action]
            if self.current_phase == next_phase:
                self.phase_time += 1
            else:
                self.current_phase = next_phase
                self.phase_time = 1
            traci.trafficlight.setRedYellowGreenState( self.tl_id, self.current_phase)
            self.sim()

        else:
            # print("new phase")
            next_phase = self.int_to_phase[action]
            phases = self.get_intermediate_phases( self.current_phase, next_phase)
            self.phase_deque.extend(phases + [next_phase])    
            #get reward before simulate.
            # self.phase_time = 0
            while len(self.phase_deque) != 0 :
                self.current_phase = self.phase_deque.popleft()
                traci.trafficlight.setRedYellowGreenState( self.tl_id, self.current_phase )
                # set current phase,and sim t seconds.
                for _ in range(self.phase_duration()):
                    self.sim()
            #get reward after simulate.
            reward = self.get_reward()
            done = False
            next_state, _ = self.get_state()
            return reward, next_state, done

    def get_reward(self):
        self.previous_action_phase = self.current_phase
        self.previous_action_lanes = self.get_phase_lanes([self.previous_action_phase])
        #return negative delay as reward
        if self.args.metric == 'queue':
            q = self.trafficmetrics.get_metric('queue', self.previous_action_lanes)
            r = -q
            return r
        elif self.args.metric == 'log_queue':
            log_q = self.trafficmetrics.get_metric('log_queue', self.previous_action_lanes)
            r = -log_q
            return r
        elif self.args.metric == 'throughput':
            weighted_sum_queues = self.trafficmetrics.get_metric('throughput', self.previous_action_lanes)
            # print(weighted_sum_queues)
            return -weighted_sum_queues

    def get_lane_delay(self):
        lane_delay = {lane : [] for lane in self.incoming_lanes}
        ave_lane_delay = {lane : 0 for lane in self.incoming_lanes}
        # print(self.incoming_lanes)
        for lane in self.incoming_lanes:
            lt = list(self.lane_travel_times[lane].values())
            ld = [l - self.lane_lengths[lane]/float(self.lane_seppds[lane]) for l in lt]
            lane_delay[lane] = ld
            if len(ld) > 0:
                ave_lane_delay[lane] = np.mean(ld)
        return lane_delay, ave_lane_delay

    def get_road_delay(self):
        road_delay = {road : 0 for road in self.incoming_roads}
        for road in self.incoming_roads:
            rt = list(self.road_travel_times[road].values())
            rd = [r - self.road_lengths[road]/float(self.road_speeds[road]) for r in rt]
            if len(rd) > 0:
                road_delay[road] = np.mean(rd)
        return road_delay

    def get_total_travel_times(self):
        # vt = [self.v_travel_times[v] for v in self.v_travel_times]
        vt = list(self.v_travel_times.values())
        # print(vt)
        if len(vt) > 0:
            average_travel_time = np.mean(vt)
        else:
            average_travel_time = 0
        return average_travel_time

    def get_lane_travel_times(self):
        lane_mean_travel_times = {lane: 0 for lane in self.incoming_lanes}
        for lane in self.incoming_lanes:
            lt = list(self.lane_travel_times[lane].values())
            if len(lt) > 0:
                lane_mean_travel_times[lane] = np.mean(lt)
        return lane_mean_travel_times

    def get_road_travel_times(self):
        road_mean_travel_times = {road : 0 for road in self.incoming_roads}
        for road in self.incoming_roads:
            # rt = [self.road_travel_times[road][v] for v in self.road_travel_times[road]]
            rt = list(self.road_travel_times[road].values())
            # print(rt)
            if len(rt) > 0:
                road_mean_travel_times[road] = np.mean(rt)
        return road_mean_travel_times
    
    def update_travel_times(self):
        # record vehicles depart and arrive time to calc the travel time.
        for v in traci.simulation.getDepartedIDList():
            lane = traci.vehicle.getLaneID(v)
            if lane not in self.incoming_lanes:
                continue
            road = str(lane).split('_')[0]
            self.v_start_roads[v] = road
            self.v_start_lanes[v] = lane
            self.v_start_times[v] = self.t
            self.road_travel_times[road][v] = 0
            self.lane_travel_times[lane][v] = 0
            self.v_travel_times[v] = 0
        
        for v in self.v_start_times:
            road = self.v_start_roads[v]
            lane = self.v_start_lanes[v]
            self.road_travel_times[road][v] += 1
            self.lane_travel_times[lane][v] += 1
            self.v_travel_times[v] += 1

        for v in traci.simulation.getArrivedIDList():
            # road = self.v_start_roads[v]
            # self.road_travel_times[road][v] = self.t - self.v_start_times[v]
            # self.v_travel_times[v] = self.t -self.v_start_times[v]
            if v not in self.v_start_times.keys():
                continue
            del self.v_start_times[v]
            del self.v_start_roads[v]
            del self.v_start_lanes[v]
        
        
    def empty_intersection(self):
        for lane in self.incoming_lanes:
            if len(self.data[lane]) > 0:
                return False
        return True

    def get_timestep(self):
        return self.t


    def sim(self):
        traci.simulationStep()
        self.data = self.get_subscription_data()
        self.trafficmetrics.update(self.previous_action_lanes, self.data)
        self.update_travel_times()
        self.t += 1

    def get_sumoCmd(self, gui, roadnet, max_steps):
        if gui == False:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')
        sumoCmd = [sumoBinary, "-c", os.path.join('roadnet',roadnet+'.sumocfg'),"--no-step-log", "true","--waiting-time-memory", str(max_steps), "--no-warnings", "true"]
        return sumoCmd
    
    def start(self):
        self.t = 0
        traci.start(self.sumoCmd)
        # traci.load(['-r','roadnet/single.rou.xml'])
        traci.junction.subscribeContext(self.tl_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 300, 
                                        [traci.constants.VAR_LANEPOSITION, 
                                        traci.constants.VAR_POSITION,
                                        traci.constants.VAR_SPEED, 
                                        traci.constants.VAR_LANE_ID])
        if not self.initilize:
            self.green_phases = self.get_tl_green_phases()
            self.phase_lanes = self.get_phase_lanes(self.green_phases)
            self.all_red = len((self.green_phases[0]))*'r'
            self.incoming_lanes = set()
            for p in self.phase_lanes:
                for l in self.phase_lanes[p]:
                    self.incoming_lanes.add(l)
            self.incoming_lanes = sorted(list(self.incoming_lanes))
            print(self.incoming_lanes)
            self.outgoing_lanes = set()
            for lane in self.incoming_lanes:
                self.outgoing_lanes.add(list(self.netdata['lane'][lane]['outgoing'].keys())[0])
            self.outgoing_lanes = sorted(list(self.outgoing_lanes))
            print(self.outgoing_lanes)

            self.incoming_roads = set()
            for l in self.incoming_lanes:
                road = str(l).split('_')[0]
                if road not in self.incoming_roads:
                    self.incoming_roads.add(road)
            self.incoming_roads = sorted(list(self.incoming_roads))


            self.update_netdata()

            self.road_lengths = {road:self.netdata['edge'][road]['length'] for road in self.incoming_roads}
            self.road_speeds = {road:self.netdata['edge'][road]['speed'] for road in self.incoming_roads}
            self.lane_lengths = {lane:self.netdata['lane'][lane]['length'] for lane in self.incoming_lanes}
            self.lane_seppds = {lane:self.netdata['lane'][lane]['speed'] for lane in self.incoming_lanes}

            self.phase_to_one_hot = self.input_to_one_hot(self.green_phases)
            self.int_to_phase = self.int_to_input(self.green_phases)
            self.initilize =True

        self.v_start_times = {}
        self.v_travel_times = {}
        self.v_start_roads = {}
        self.v_start_lanes = {}
        self.road_travel_times = {road:{} for road in self.incoming_roads}
        self.lane_travel_times = {lane:{} for lane in self.incoming_lanes}

        self.phase_deque = deque()
        self.current_phase = self.green_phases[0]
        self.previous_action_phase = self.all_red
        self.previous_action_lanes = {}
        self.phase_time = 0
        
        if self.mode == 'train':
            self.metric_args = ['delay','queue','log_queue','throughput']

        self.trafficmetrics = TrafficMetrics(self.tl_id, self.incoming_lanes, self.netdata, self.metric_args, self.mode)
        self.data = None

    def close(self):
        traci.close()

        
