from sumo_agent import SumoAgent
from agent.sotl_agent import SotlAgent
from generator import TrafficGenerator
from parseargs import parse_cl_args
from logger import Logger
import timeit
import numpy as np
import os, sys
from saver import Saver

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
        self.sumo_agent = SumoAgent(args, self.args.roadnet, self.args.mode, self.args.red_duration, self.args.yellow_duration)   
        self.sumo_agent.close()
        action_size = self.sumo_agent.get_action_size()
        self.sotl_agent = SotlAgent(action_size)  
        self.episode_lane_delay = {lane:[] for lane in self.sumo_agent.incoming_lanes}
        self.episode_average_travel_times = []

        self.save_path = os.path.join(os.getcwd(), "save", "sotl", args.roadnet, '')
        os.makedirs(os.path.dirname(self.save_path),exist_ok=True)

        self.saver = Saver(self.save_path)

    def run(self, episode, bias):
        start_time = timeit.default_timer()

        if args.roadnet == 'imbalance':
            self.traffic_gen.generate_imbalance(seed=episode)
        else:
            self.traffic_gen.generate_biased(seed=episode, bias=bias)
 
        self.time_step = 0
        self.sumo_agent.start()
        last_action = self.sotl_agent.choose_action(self.sumo_agent.get_state())
        while self.time_step < self.args.max_steps:
            state = self.sumo_agent.get_state() 
            action = self.sotl_agent.choose_action(state)
            if action == last_action:
                self.sumo_agent.simulate_action(action)
            else:
                # first simulate yellow and red
                phases = self.sumo_agent.get_intermediate_phases( self.sumo_agent.int_to_phase[last_action], self.sumo_agent.int_to_phase[action])
                yellow, red = phases[0], phases[1]
                for _ in range(self.args.yellow_duration):                
                    traci.trafficlight.setRedYellowGreenState( self.sumo_agent.tl_id, yellow)
                    self.sumo_agent.sim()
                    self.time_step += 1
                for _ in range(self.args.red_duration):
                    traci.trafficlight.setRedYellowGreenState( self.sumo_agent.tl_id, red)
                    self.sumo_agent.sim()
                    self.time_step += 1
                self.sumo_agent.simulate_action(action)

            last_action = action
            self.time_step += 1 

        # self.saver.save_data(data=self.phase_list,filename="phase-list-of-episode"+str(episode))
        travel_times = self.sumo_agent.get_total_travel_times()
        lane_delay, ave_lane_delay = self.sumo_agent.get_lane_delay()
        self.sumo_agent.close()

        for lane in self.sumo_agent.incoming_lanes:
            self.episode_lane_delay[lane].append(ave_lane_delay[lane])
        self.episode_average_travel_times.append(travel_times)
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time
    
    def save_results(self):
        mode = "flush"
        for lane in self.sumo_agent.incoming_lanes:
            self.saver.save_data(data=self.episode_lane_delay[lane], filename='Delay-of-'+lane, mode=mode)
        self.saver.save_data(data=self.episode_average_travel_times,filename='Average-travel-time',mode=mode)
        # self.saver.save_data_and_plot(data=self.episode_travel_times, filename='test-Travel-time', xlabel='Episode', ylabel='episode mean travel time (s)')

if __name__ == "__main__":
    args = parse_cl_args()
    log = Logger('episode_info.log',level='info')
    ctl = Controller(args)
    episode = 0
    for episode in range(args.test_episodes):
        bias = episode * args.maxbias / args.test_episodes
        simulation_time = ctl.run(episode, bias)
        log.logger.info('episode:{} - Simulation time:{}s'.format(episode+1, simulation_time))
    
    ctl.save_results()