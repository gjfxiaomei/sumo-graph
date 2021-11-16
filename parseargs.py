import argparse,os

def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=2, dest='n', help='number of sim procs (parallel simulations) generating experiences, default: os.cpu_count()-1')
    parser.add_argument("-tsc",type=str,default='dqn',dest='tsc',help='traffic singal control. options:uniform,dqn,ppo')
    parser.add_argument("-mln", type=int, default=1, dest='mln', help='model number')
    parser.add_argument("-cmt", type=str, default='',dest='cmt',help='add a comment about result')
    parser.add_argument("-ln",default=True,action='store_true',dest='ln',help='learning method?')
    
    parser.add_argument("-maxbias", type=float, default=0.4, dest="maxbias", help="max bias of traffic flow in [0, 0.5]")
    ##
    parser.add_argument("-m",type=str,default='queue',dest='metric',help='metrric of the reward')
    parser.add_argument("-s",type=str,default='lane',dest='state',help='state type: lane or phase lane')
    parser.add_argument("-conTrain",default=False,action='store_true',dest='conTrain',help='continue train if last train is not ideal')
    ##sumo params
    parser.add_argument("-gui",default=False,action='store_true',dest='gui',help='use gui, default:False')
    parser.add_argument("-sumocfg",type=str,default='single4',dest='sumocfg',help='path to desired simulation sumo configuration file')
    parser.add_argument("-roadnet",type=str,default='single4',dest='roadnet',help = 'roadnet flie, options:[single4,single8]')

    parser.add_argument("-train_episodes",type=int,default=100,dest='train_episodes',help='total train episodes')
    parser.add_argument("-test_episodes",type=int,default=10,dest='test_episodes',help='total test episodes')
    parser.add_argument("-max_steps",type=int,default=3600,dest='max_steps',help='max steps in each simulaiton episode')
    parser.add_argument("-n_cars_generated",type=int,default=800,dest='n_cars_generated',help='number of cars generated in each episode')
    parser.add_argument("-green_duration",type=int,default=20,dest='green_duration',help="duration of green light")
    parser.add_argument("-yellow_duration",type=int,default=3,dest='yellow_duration',help="duration of yellow light")
    parser.add_argument("-red_duration",type=int,default=2,dest='red_duration',help="duration of red light")
    parser.add_argument("-port", type=int,default=9000, dest='port', help='port to connect self.conn.server, default: 9000')
    parser.add_argument("-mode",type=str,default='train',dest='mode',help='option:trian,test')
    parser.add_argument("-scale",type=float,default=1.4,dest='scale',help='vehicle generation scale parameter, higher values generates more vehicles')
    
    #graph params

    parser.add_argument("-in_dim", type=int, default=8)
    parser.add_argument('-hidden_dim', type=int, default=32)
    parser.add_argument('-out_dim', type=int, default=8)
    parser.add_argument('-num_heads', type=int, default=2)


    #rl parmas
    parser.add_argument("-batch_size", type=int, default=32, dest='batch_size', help='batch size to sample from replay to train neural net, default: 32')
    #neural net params
    parser.add_argument("-lr", type=float, default=0.0001, dest='lr', help='ddpg actor/dqn neural network learning rate, default: 0.0001')
    parser.add_argument("-lrc", type=float, default=0.001, dest='lrc', help='ddpg critic neural network learning rate, default: 0.001')
    parser.add_argument("-lre", type=float, default=0.00000001, dest='lre', help='neural network optimizer epsilon, default: 0.00000001')
    parser.add_argument("-hidden_act", type=str, default='elu', dest='hidden_act', help='neural network hidden layer activation, default: elu')
    parser.add_argument("-n_hidden", type=int, default=3, dest='n_hidden', help='neural network hidden layer scaling factor, default: 3')
    
    # parser.add_argument("-save_path", type=str, default='saved_models', dest='save_path', help='dir to save neural network weights, default: saved_models')
    # parser.add_argument("-save_replay", type=str, default='saved_replays', dest='save_replay', help='dir to save experience replays, default: saved_replays')
    # parser.add_argument("-load_replay", default=False, action='store_true', dest='load_replay', help='load experience replays if they exist')

    # parser.add_argument("-save_t", type=int, default=120, dest='save_t', help='interval in seconds between saving neural networks on learners, default: 120 (s)')
    # parser.add_argument("-save", default=False, action='store_true', dest='save', help='use argument to save neural network weights')
    # parser.add_argument("-load", default=False, action='store_true', dest='load', help='use argument to load neural network weights assuming they exist')
    
    args= parser.parse_args()
    return args
