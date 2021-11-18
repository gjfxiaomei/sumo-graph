import os
import numpy

from numpy.core.numeric import normalize_axis_tuple
from graph_network import GraphAgent
from memory import Memory
from parseargs import parse_cl_args
from logger import Logger
from tqdm import tqdm
import matplotlib.pyplot as plt
from sumo_agent import SumoAgent
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from roadnet_reader import RoadnetReader
from utils import set_train_path
from saver import Saver
from generator import TrafficGenerator
import os
import timeit
import pandas as pd
import math
from torch.utils.tensorboard import SummaryWriter

class Controller:
    def __init__(self,args):
        self.args = args
        
        self.traffic_gen = TrafficGenerator(self.args.max_steps, self.args.n_cars_generated)
        self.tl_id = "TL"
        self.time_step = 0
        self.mode = 'train'
        self.sumo_agent = SumoAgent(args, self.args.roadnet, self.mode, self.args.red_duration, self.args.yellow_duration)     
        self.episode_road_travel_times = {road:[] for road in self.sumo_agent.incoming_roads}
        self.phase_list = self.sumo_agent.get_tl_green_phases()
        self.graph_agent = GraphAgent(self.phase_list, self.args.graph_in_dim, self.args.graph_hidden_dim, self.args.graph_out_dim, self.args.graph_num_heads, self.args.graph_lr,  ('road', 'connected', 'road'))

        self.episode_average_travel_times = []
        self.save_path = set_train_path(args.roadnet, args.tsc, self.mode, self.args.metric, self.args.cmt)
        self.saver = Saver(self.save_path)
        self.graph_memory = {}
        for phase in self.phase_list:
            self.graph_memory[phase] = Memory(size_max=500000, size_min=50)
        state_size = self.sumo_agent.get_state_size()
        action_size = self.sumo_agent.get_action_size()
        if self.args.tsc == 'dqn':
            self.alg_agent = DQNAgent(state_size, action_size)
        elif self.args.tsc == 'ppo':
            self.alg_agent = PPOAgent(state_size, action_size)

        if self.args.conTrain == True:
            self.load_model()
        
        self.sumo_agent.close()

    def run(self,episode, epsilon, writer):

        start_time = timeit.default_timer()
        if self.args.tsc == 'dqn':
            self.alg_agent.set_epsilon(epsilon)

        if self.args.roadnet != "lastvehicle":
            # self.traffic_gen.generate_uniform(seed=episode)
            bias = (episode % args.test_episodes) * args.maxbias / args.test_episodes
            self.traffic_gen.generate_biased(episode,bias)

        self.time_step = 0
        self.episode_reward = 0
        # if episode != 0:
        self.sumo_agent.start()
        while self.time_step < self.args.max_steps:
            #to get a new phase when the phase deque is empty.
            # print("step:",self.time_step)
            if len(self.sumo_agent.phase_deque) == 0 :
                #get state
                # print("get_newphase")
                state, phase = self.sumo_agent.get_state()
                current_volume_node = self.sumo_agent.get_volume_node()
                #get action
                action = self.alg_agent.get_action(state)
                #simulate a phase
                reward, next_state, _ = self.sumo_agent.simulate_action(action)
                next_volume_node = self.sumo_agent.get_volume_node()
                self.graph_memory[phase].add_sample((current_volume_node, next_volume_node))
                # print(reward)
                self.episode_reward += reward
                self.time_step =self.sumo_agent.get_timestep()
                #store experience
                # print(reward)
                if self.args.tsc == 'dqn':
                    self.alg_agent.store_experience(state,action,reward,next_state,done = False)
                elif self.args.tsc == 'ppo':
                    self.alg_agent.store_experience(reward, done = False)
            else:
                self.sumo_agent.sim()
                self.time_step = self.sumo_agent.get_timestep()

        # road_travel_times, average_travel_times = self.sumo_agent.get_travel_times()
        average_travel_times = self.sumo_agent.get_total_travel_times()
        # for road in self.sumo_agent.incoming_roads:
        #     self.episode_road_travel_times[road].append(road_travel_times[road])
        self.episode_average_travel_times.append(average_travel_times)
        self.sumo_agent.close()
        
        simulation_time = round(timeit.default_timer() - start_time, 1)
        start_time = timeit.default_timer()
        # self.alg_agent.train()
        for phase in self.phase_list:
            samples = self.graph_memory[phase].get_samples(128)
            if len(samples) !=0:
                state = []
                next_state = []
                for i in range(self.graph_agent.agent_dict[phase].g.num_nodes()):
                    current_node_state = []
                    next_node_state = []
                    for item in samples:
                        current_node_state.append(item[0][i])
                        next_node_state.append(item[1][i])
                    state.append(current_node_state)
                    next_state.append(next_node_state)
                state = numpy.array(state)
                next_state = numpy.array(next_state)
                loss = self.graph_agent.train(state, next_state, phase)
                print(loss)
                writer.add_scalar(f'{phase}-loss', loss, episode+1)
        training_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time, training_time, average_travel_times, self.episode_reward
    
    def save_results(self):
        if self.args.conTrain == True:
            mode = "append"
        else:
            mode = "flush"
        # self.saver.set_path(self.save_path)
        # for road in self.sumo_agent.incoming_roads:
        #     self.saver.save_data(data=self.episode_road_travel_times[road], filename='Travel-time-of-'+road, mode=mode)
        self.saver.save_data(data=self.episode_average_travel_times,filename="Average-travel-time", mode=mode)
        # self.saver.save_data_and_plot(data=self.episode_travel_times, filename='train-Travel-time',xlabel='Episode', ylabel='episode mean travel time (s)')

    def load_model(self):
        self.alg_agent.load_model(self.save_path)

    def save_model(self):
        self.alg_agent.save_model(self.save_path)


if __name__ == "__main__":
    args = parse_cl_args()
    writer = SummaryWriter(comment="--"+args.tsc+"--"+args.metric+"--"+args.cmt)
    log = Logger('episode_info.log',level='info')
    ctl = Controller(args)
    episode = 0
    EPS_START = 0.9
    EPS_END = 0.05
    DECAY_EPI = int(args.train_episodes*0.5)
    for episode in tqdm(range(args.train_episodes)):
        if episode < DECAY_EPI:
            epsilon = EPS_END + (EPS_START-EPS_END)*math.exp(-1. * episode/DECAY_EPI)
        else:
            epsilon = 0.05
        simulation_time, training_time, average_travel_times, reward = ctl.run(episode, epsilon, writer)
        writer.add_scalar('train/ave_travel_time', average_travel_times, episode + 1)
        writer.add_scalar('train/reward', reward, episode + 1)
        log.logger.info('episode:{} - Simulation time:{}s - Training time:{}s - Total:{}s'.format(episode+1, simulation_time, training_time,round(simulation_time+training_time, 1)))
        if episode % 50 == 0:
            ctl.save_model()
    ctl.save_results()
    writer.close()