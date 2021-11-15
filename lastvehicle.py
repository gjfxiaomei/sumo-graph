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
        
        # self.traffic_gen = TrafficGenerator(self.args.max_steps, self.args.n_cars_generated)
        self.tl_id = "TL"
        self.time_step = 0
        self.mode = 'test'
        self.sumo_agent = SumoAgent(args, self.args.roadnet, self.args.mode, self.args.red_duration, self.args.yellow_duration)   
        self.sumo_agent.close()
        
        if self.args.tsc == 'dqn':
            self.model_path, _ = set_test_path("single4", args.tsc, self.args.metric, args.mln)
            self.save_path = os.path.join(os.getcwd(), "LastVehicle", self.args.metric,'')
            state_size = self.sumo_agent.get_state_size()
            action_size = self.sumo_agent.get_action_size()
            self.alg_agent = DQNAgent(state_size,action_size)
            self.load_model()

        elif self.args.tsc == 'uniform':
            action_size = self.sumo_agent.get_action_size()
            self.save_path = os.path.join(os.getcwd(), "LastVehicle/uniform/")
            self.alg_agent = UniformAgent(action_size)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.saver = Saver(self.save_path)

    def run(self):
        start_time = timeit.default_timer()
        self.time_step = 0
        self.sumo_agent.start()

        while self.time_step < self.args.max_steps:
            if len(self.sumo_agent.phase_deque) == 0:
                state = self.sumo_agent.get_state()
                # print(state)
                action = self.alg_agent.predict(state)
                reward, next_state, _ = self.sumo_agent.simulate_action(action)
                self.time_step =self.sumo_agent.get_timestep()
            else:
                self.sumo_agent.sim()
                self.time_step = self.sumo_agent.get_timestep()
        # self.saver.save_data(data=self.phase_list,filename="phase-list-of-episode"+str(episode))
        travel_times = self.sumo_agent.get_total_travel_times()
        lane_delay, _ = self.sumo_agent.get_lane_delay()
        print("average travel time:", travel_times," delay of last vehilce:", lane_delay["E2TL_1"])
        self.sumo_agent.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time

    def load_model(self):
        print("load model from",self.model_path)
        self.alg_agent.load_model(self.model_path)

    def save_result(self):
        pass

if __name__ == "__main__":
    args = parse_cl_args()
    log = Logger('episode_info.log',level='info')
    ctl = Controller(args)
    simulation_time = ctl.run()
    # log.logger.info('episode:{} - Simulation time:{}s'.format(episode+1, simulation_time))