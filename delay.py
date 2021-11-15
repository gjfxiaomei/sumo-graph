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
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from parseargs import parse_cl_args
from logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.episode_major_delay = []
        self.episode_minor_delay = []
        
        if self.args.tsc == 'dqn':
            self.model_path, _ = set_test_path(args.roadnet, args.tsc, self.args.metric, args.mln)
            self.save_path = os.path.join(os.getcwd(), "Delay", self.args.metric,'')
            state_size = self.sumo_agent.get_state_size()
            action_size = self.sumo_agent.get_action_size()
            self.alg_agent = DQNAgent(state_size,action_size)
            self.load_model()

        elif self.args.tsc == 'uniform':
            action_size = self.sumo_agent.get_action_size()
            self.save_path = os.path.join(os.getcwd(), "Delay/uniform/")
            os.makedirs(os.path.dirname(self.save_path),exist_ok=True)
            self.alg_agent = UniformAgent(action_size)


        self.saver = Saver(self.save_path)

    def run(self, episode, bias):
        start_time = timeit.default_timer()

        if args.roadnet == 'imbalance':
            self.traffic_gen.generate_imbalance(seed=episode)
        else:
            self.traffic_gen.generate_biased(seed=episode, bias=bias)
            # self.traffic_gen.generate_uniform(episode)
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
        lane_delay, _ = self.sumo_agent.get_lane_delay()
        self.sumo_agent.close()

        major_delay = []
        minor_delay = []

        major_delay.extend(lane_delay['N2TL_1'])
        major_delay.extend(lane_delay['N2TL_2'])
        major_delay.extend(lane_delay['S2TL_1'])
        major_delay.extend(lane_delay['S2TL_2'])

        minor_delay.extend(lane_delay['W2TL_1'])
        minor_delay.extend(lane_delay['W2TL_2'])
        minor_delay.extend(lane_delay['E2TL_1'])
        minor_delay.extend(lane_delay['E2TL_2'])

        major_delay = pd.DataFrame(major_delay, columns=['delay'])
        major_delay['bias'] = bias+0.5
        major_delay.to_csv(os.path.join(self.save_path, 'major'+str(bias)+'.csv'))
        
        minor_delay = pd.DataFrame(minor_delay,columns=['delay'])
        minor_delay['bias'] = bias+0.5
        minor_delay.to_csv(os.path.join(self.save_path, 'minor'+str(bias)+'.csv'))

        self.episode_major_delay.append(major_delay)
        self.episode_minor_delay.append(minor_delay)

        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time
    
    def save_results(self):
        total_major_delay = pd.concat(self.episode_major_delay)
        total_minor_delay = pd.concat(self.episode_minor_delay)
        # sns.set(style='whitegrid')
        # ax = sns.boxplot(x='bias',y='delay',data=total_minor_delay)
        # # plt.show()
        # plt.savefig(os.path.join(self.save_path,'delay.pdf'),dpi=800, format='pdf')
        total_major_delay.to_csv(os.path.join(self.save_path,'major_delay.csv'))
        total_minor_delay.to_csv(os.path.join(self.save_path,'minor_delay.csv'))


    def load_model(self):
        print("load model from",self.model_path)
        self.alg_agent.load_model(self.model_path)
    

if __name__ == "__main__":
    args = parse_cl_args()
    log = Logger('episode_info.log',level='info')
    ctl = Controller(args)
    episode = 0
    bias_list = [0.0, 0.25, 0.4]
    for episode in range(3):
        bias = bias_list[episode]
        simulation_time = ctl.run(episode, bias)
        log.logger.info('episode:{} - Simulation time:{}s'.format(episode+1, simulation_time))
    
    ctl.save_results()
