
##
# How long does it take for a car on non-arterial road
# to be served.
##

from sumo_agent import SumoAgent
from dqn_agent import DqnAgent
from uniform_agent import UniformAgent
from roadnet_reader import RoadnetReader
from utils import set_save_path
from saver import Saver
from generator import TrafficGenerator
import timeit

import os, sys
from parseargs import parse_cl_args
from logger import Logger

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class Controller:
    def __init__(self,args):
        self.args = args
        
        self.traffic_gen = TrafficGenerator(self.args.max_steps, self.args.n_cars_generated)
        self.tl_id = "TL"
        self.time_step = 0
        self.mode = 'test'
        self.netdata_reader = RoadnetReader(args.roadnet)
        self.netdata = self.netdata_reader.get_net_data()  
        # print(self.netdata['lane'])
        self.episode_road_dely = []
        self.sumo_agent = SumoAgent(args, self.args.roadnet, self.args.mode, self.args.red_duration, self.args.yellow_duration)     
        self.sumo_agent.close()
        
        self.save_path = set_save_path(args.roadnet,args.tsc, self.mode, self.args.metric)
        self.saver = Saver(self.save_path)
        if self.args.tsc == 'dqn':
            self.model_path = set_save_path(args.roadnet,args.tsc,'train', self.args.metric)
            state_size = self.sumo_agent.get_state_size()
            action_size = self.sumo_agent.get_action_size()
            self.alg_agent = DqnAgent(self.args.batch_size,state_size,action_size)
            self.load_model()
        elif self.args.tsc == 'uniform':
            self.alg_agent = UniformAgent()
    

    def run(self,episode):
        #start sumo
        # self.sumo_agent.set_sumo(self.args.gui, self.args.roadnet, self.args.max_steps)
        start_time = timeit.default_timer()
        self.traffic_gen.generate_specific(seed=episode)
        
        self.phase_list = []
        self.time_step = 0
        # if episode != 0:
        self.sumo_agent.start()

        while self.time_step < self.args.max_steps:
            #to get a new phase when the phase deque is empty.
            # print("step:",self.time_step)
            if len(self.sumo_agent.phase_deque) == 0 :
                #get state
                state = self.sumo_agent.get_state() 
                # print("get_action")
                action = self.alg_agent.get_action(state,self.mode)
                self.phase_list.append(action)
                #simulate a phase
                reward, next_state = self.sumo_agent.simulate_action(action)
                self.time_step =self.sumo_agent.get_timestep()
            else:
                self.sumo_agent.sim()
                self.time_step = self.sumo_agent.get_timestep()
        
        self.road_travel_times, self.average_travel_times = self.sumo_agent.get_travel_times()
        self.road_delay = self.sumo_agent.get_delay()
        print(self.road_delay)
        self.episode_road_dely.append(self.road_delay)

        self.sumo_agent.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time
    


    def save_results(self):
        save_path = os.path.join(os.getcwd(),'save', "response", self.args.tsc, self.args.metric,'')
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(os.path.join(save_path, 'response.txt'), "w") as file:
            for road_dely in self.episode_road_dely:
                file.write("%s %s\n"%(road_dely['E2TL'],road_dely['S2TL']))

    def load_model(self):
        self.alg_agent.load_model(self.model_path)
        print("load model success")


if __name__ == "__main__":
    args = parse_cl_args()
    # log = Logger('episode_info.log',level='info')
    ctl = Controller(args)
    episode = 0
    for episode in range(10):
        simulation_time = ctl.run(episode)
        # log.logger.info('episode:{} - Simulation time:{}s'.format(episode+1, simulation_time))
    
    ctl.save_results()
    
    
