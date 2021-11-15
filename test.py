#set sumo and get state
#feed state to dqn_agent and get action
#feed the action recived from dqn_agent to sumo_agent ,simulate and get new_state and reward
# train dqn_agent
from sumo_agent import SumoAgent

from agent.uniform_agent import UniformAgent
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from roadnet_reader import RoadnetReader
from utils import set_train_path, set_test_path
from saver import Saver
from generator import TrafficGenerator
import timeit
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from parseargs import parse_cl_args
from logger import Logger

class Controller:
    def __init__(self,args):
        self.args = args
        
        self.traffic_gen = TrafficGenerator(self.args.max_steps, self.args.n_cars_generated)
        self.tl_id = "TL"
        self.time_step = 0
        self.mode = 'test'
        self.sumo_agent = SumoAgent(args, self.args.roadnet, self.args.mode, self.args.red_duration, self.args.yellow_duration)   
        self.sumo_agent.close()  
        self.episode_lane_travel_times = {lane:[] for lane in self.sumo_agent.incoming_lanes}
        self.episode_lane_delay = {lane:[] for lane in self.sumo_agent.incoming_lanes}
        self.episode_average_travel_times = []
        
        if self.args.tsc == 'dqn':
            self.model_path, self.save_path = set_test_path(args.roadnet, args.tsc, self.args.metric, args.mln)
            state_size = self.sumo_agent.get_state_size()
            action_size = self.sumo_agent.get_action_size()
            self.alg_agent = DQNAgent(state_size,action_size)
            self.load_model()

        elif self.args.tsc == 'uniform':
            action_size = self.sumo_agent.get_action_size()
            self.save_path = os.path.join(os.getcwd(), "save", "uniform", args.roadnet, '')
            os.makedirs(os.path.dirname(self.save_path),exist_ok=True)
            self.alg_agent = UniformAgent(action_size)

        elif self.args.tsc == 'ppo':
            self.save_path = set_save_path(args.roadnet,args.tsc, self.mode, self.args.metric)
            self.model_path = set_save_path(args.roadnet,args.tsc,'train', self.args.metric)
            state_size = self.sumo_agent.get_state_size()
            action_size = self.sumo_agent.get_action_size()

            self.alg_agent = PPOAgent(state_size,action_size)
            self.load_model()

        self.saver = Saver(self.save_path)

    def run(self, episode, bias):
        start_time = timeit.default_timer()
        test_batch_ave_daley = {lane:[] for lane in self.sumo_agent.incoming_lanes}
        test_batch_ave_travel = []

        for i in range(1):
            self.traffic_gen.generate_biased(seed=episode*10+i, bias=bias)
            # self.phase_list = []
            self.time_step = 0
            # if episode != 0:
            self.sumo_agent.start()

            while self.time_step < self.args.max_steps:
                #to get a new phase when the phase deque is empty.
                # print("step:",self.time_step)
                if len(self.sumo_agent.phase_deque) == 0 :
                    state = self.sumo_agent.get_state() 
                    action = self.alg_agent.predict(state)
                    reward, next_state, _ = self.sumo_agent.simulate_action(action)
                    self.time_step =self.sumo_agent.get_timestep()
                else:
                    self.sumo_agent.sim()
                    self.time_step = self.sumo_agent.get_timestep()
            # self.saver.save_data(data=self.phase_list,filename="phase-list-of-episode"+str(episode))
            travel_times = self.sumo_agent.get_total_travel_times()
            lane_delay, ave_lane_delay = self.sumo_agent.get_lane_delay()
            self.sumo_agent.close()
            for lane in self.sumo_agent.incoming_lanes:
                test_batch_ave_daley[lane].append(ave_lane_delay[lane])
            test_batch_ave_travel.append(travel_times)
        
        for lane in self.sumo_agent.incoming_lanes:
            self.episode_lane_delay[lane].append(np.mean(test_batch_ave_daley[lane]))
        self.episode_average_travel_times.append(np.mean(test_batch_ave_travel))
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time
    
    def save_results(self):
        mode = "flush"
           
        for lane in self.sumo_agent.incoming_lanes:
            self.saver.save_data(data=self.episode_lane_delay[lane], filename='Delay-of-'+lane, mode=mode)
        
        self.saver.save_data(data=self.episode_average_travel_times,filename='Average-travel-time',mode=mode)
        # self.saver.save_data_and_plot(data=self.episode_travel_times, filename='test-Travel-time', xlabel='Episode', ylabel='episode mean travel time (s)')

    def load_model(self):
        print("load model from",self.model_path)
        self.alg_agent.load_model(self.model_path)
    

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
