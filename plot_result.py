import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import pandas as pd
import numpy as np
# from utils import set_save_path
from torch.utils.tensorboard import SummaryWriter

def JFI(dis):
    return np.square(sum(dis)) /( len(dis)* sum(np.square(dis)) )

def smooth(data, weight=0.9):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_tail(bias):
    fig = plt.figure(1)
    queue_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/queue/%s.csv'%(str(bias))),index_col=None)
    data = queue_delay['Delay'].values
    data = np.array(data)
    threhold = range(int(np.max(data)+1))
    dis = [np.sum(data>item)/ data.size for item in threhold]
    plt.scatter(x=threhold, y=dis, label='LIT')

    thp_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/throughput/%s.csv'%(str(bias))),index_col=None)
    data = thp_delay['delay'].values
    data = np.array(data)
    threhold = range(int(np.max(data)+1))
    dis = [np.sum(data>item)/ data.size for item in threhold]
    plt.scatter(x=threhold, y=dis, label='FIT')

    uniform_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/uniform/%s.csv'%(str(bias))),index_col=None)
    data = uniform_delay['delay'].values
    data = np.array(data)
    threhold = range(int(np.max(data)+1))
    dis = [np.sum(data>item)/ data.size for item in threhold]
    plt.scatter(x=threhold, y=dis, label='FT')
    plt.xlabel('t(s)')
    plt.ylabel('P(delay > t)')
    plt.title("ratio = "+str(bias+0.5))
    plt.legend()
    fig.savefig('tail_%s.png'%(bias),dpi=800,format='png')
    plt.clf()

def plot_delay(road):
    queue_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/LIT/%s_delay.csv'%(road)),index_col=None)
    thp_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/FIT/%s_delay.csv'%(road)),index_col=None)
    uniform_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/FT/%s_delay.csv'%(road)),index_col=None)
    sotl_delay = pd.read_csv(os.path.join(os.getcwd(),'Delay/SOTL/%s_delay.csv'%(road)),index_col=None)
    sotl_delay = sotl_delay[sotl_delay['delay'].astype(float) < 1000]
    queue_delay['method'] = 'LIT'
    thp_delay['method'] = 'FIT'
    uniform_delay['method'] = 'FT'
    sotl_delay['method'] = 'SOTL'
    total_delay = pd.concat([uniform_delay,sotl_delay,queue_delay,thp_delay])
    fig = plt.figure()
    ax = sns.boxplot(x='bias',y='delay',hue='method',fliersize=5.0, data=total_delay)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$', font)
    plt.ylabel('Delay(s)',font)
    plt.legend(title=None)
    # plt.show()
    fig.savefig(os.path.join(os.getcwd(),'pics','%s_delay.pdf'%(road)),dpi=800,format='pdf')

def get_score(data):
    score_list = []
    for i in range(0,len(data)):
        delay = data.iloc[i]['delay']
        if delay < 40:
            score = 5
        elif delay < 80 and delay > 40:
            score = 4
        elif delay < 120 and delay > 80:
            score = 3
        elif delay < 160 and delay > 120:
            score = 2
        elif delay > 160:
            score = 1
        score_list.append(score)
    return score_list

def get_des(method):
    major = pd.read_csv(os.path.join(os.getcwd(),'Delay/%s/major0.25.csv'%(method)), index_col=None)
    minor = pd.read_csv(os.path.join(os.getcwd(),'Delay/%s/minor0.25.csv'%(method)), index_col=None)
    score = [] 
    score.extend(get_score(major))
    score.extend(get_score(minor))
    return score

def plot_des():
    FT = get_des("FT")
    SOTL = get_des("SOTL")
    LIT = get_des("LIT")
    FIT = get_des("FIT")
    
    labels = '5', '4', '3', '2', '1'
    fig, axs = plt.subplots(1,4,figsize=(10, 5),subplot_kw=dict(aspect="equal"))
    fig.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace = 0.4, hspace = 0.1)#调整子图间距
    # fig.tight_layout()
    ft_score = [FT.count(5),FT.count(4),FT.count(3),FT.count(2),FT.count(1)]
    sotl_score = [SOTL.count(5),SOTL.count(4),SOTL.count(3),SOTL.count(2),SOTL.count(1)]
    lit_score = [LIT.count(5),LIT.count(4),LIT.count(3),LIT.count(2),LIT.count(1)]
    fit_score = [FIT.count(5),FIT.count(4),FIT.count(3),FIT.count(2),FIT.count(1)]
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    print("FT:",np.mean(FT),np.var(FT))
    print("SOTL:",np.mean(SOTL),np.var(SOTL))
    print("LIT:",np.mean(LIT),np.var(LIT))
    print("FIT:",np.mean(FIT),np.var(FIT))
    axs[0].pie(x=ft_score, radius= 1.4, labels=labels, autopct='%.1f%%')
    axs[0].text(-1.2, -2.1, "Mean:%.2f  Var:%.2f"%(np.mean(FT),np.var(FT)),fontsize=12)
    axs[0].set_title("FT", pad=30,font=font)
    axs[1].pie(x=sotl_score, radius= 1.4, labels=labels, autopct='%.1f%%')
    axs[1].text(-1.2, -2.1, "Mean:%.2f  Var:%.2f"%(np.mean(SOTL),np.var(SOTL)),fontsize=12)
    axs[1].set_title("SOTL", pad=30,font=font)
    axs[2].pie(x=lit_score, radius= 1.4, labels=labels, autopct='%.1f%%')
    axs[2].text(-1.2, -2.1, "Mean:%.2f  Var:%.2f"%(np.mean(LIT),np.var(LIT)),fontsize=12)
    axs[2].set_title("LIT", pad=30,font=font)
    axs[3].pie(x=fit_score, radius= 1.4, labels=labels, autopct='%.1f%%')
    axs[3].text(-1.2, -2.1, "Mean:%.2f  Var:%.2f"%(np.mean(FIT),np.var(FIT)),fontsize=12)
    axs[3].set_title("FIT", pad=30,font=font)
    
    fig.legend(labels,loc='lower left', ncol=5, title="DES:",fontsize=12)
    plt.show()
    fig.savefig('./pics/des.pdf', dpi=800, format='pdf',bbox_inches='tight')

def plot_window():
    thp_10 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_7_duration=20_exp_w=10','Average-travel-time.txt'),header=None).values.reshape(-1)
    thp_10 = smooth(thp_10)
    thp_1 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_8_duration=20_exp_w=1','Average-travel-time.txt'),header=None).values.reshape(-1)
    thp_1 = smooth(thp_1)
    thp_50 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_14_duration=20_exp_w=50','Average-travel-time.txt'),header=None).values.reshape(-1)
    thp_50 = smooth(thp_50)
    thp_100 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_10_duration=20_exp_w=100','Average-travel-time.txt'),header=None).values.reshape(-1)
    thp_100 = smooth(thp_100)
    queue = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/queue/result_1_duration=20_queue_500','Average-travel-time.txt'),header=None).values.reshape(-1)
    queue = smooth(queue)
    
    x = [round(0.5+i*0.4/10, 2) for i in range(10)]
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot(x, queue, "-+", label='LIT', color='red')
    plt.plot(x, thp_1, "-d", label='FIT:'+r'$W=1$', color='blue')
    plt.plot(x, thp_50, "-x", label='FIT:'+r'$W=10$', color='green')
    plt.plot(x, thp_10, "-D", label='FIT:'+r'$W=50$', color='orange')
    plt.plot(x, thp_100, "-_", label='FIT:'+r'$W=100$', color='black')
    bottom, top = plt.ylim()
    plt.ylim(bottom*0.8,top*1.2)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$', font)
    plt.ylabel("Average travel time(s)", font)
    plt.legend(fontsize=12)
    fig.savefig('./pics/window_efficiency.pdf', dpi=800, format='pdf')
    plt.clf()

    rl_path = './save/single4/dqn/test/'
    thp_1_delay, thp_10_delay, thp_50_delay, thp_100_delay = [],[],[],[]
    thp_1_JFI, thp_10_JFI, thp_50_JFI, thp_100_JFI  = [],[],[],[]
    queue_delay = []
    queue_JFI = []
    lanes= ['E2TL_1','E2TL_2','N2TL_1','N2TL_2','S2TL_1','S2TL_2','W2TL_1','W2TL_2']
    for l in lanes:
        thp_1_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_8_duration=20_exp_w=1/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        thp_10_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_7_duration=20_exp_w=10/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        thp_50_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_14_duration=20_exp_w=50/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        thp_100_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_10_duration=20_exp_w=100/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        queue_delay.append(pd.read_csv(os.path.join(rl_path,"queue/result_1_duration=20_queue_500/Delay-of-%s.txt"%(l)), header=None).values.reshape(-1))
    queue_delay = np.array(queue_delay)
    thp_1_delay = np.array(thp_1_delay)
    thp_10_delay = np.array(thp_10_delay)
    thp_50_delay = np.array(thp_50_delay)
    thp_100_delay = np.array(thp_100_delay)

    #calc JFI
    for i in range(thp_1_delay.shape[1]):
        queue_JFI.append(JFI(queue_delay[:,i]))
        thp_1_JFI.append(JFI(thp_1_delay[:,i]))
        thp_10_JFI.append(JFI(thp_10_delay[:,i]))
        thp_50_JFI.append(JFI(thp_50_delay[:,i]))
        thp_100_JFI.append(JFI(thp_100_delay[:,i]))

    ratios = [round(0.5+i*0.4/10, 2) for i in range(10)]
    queue_JFI = smooth(queue_JFI, weight=0.9)
    thp_1_JFI = smooth(thp_1_JFI, weight=0.9)
    thp_10_JFI = smooth(thp_10_JFI, weight=0.9)
    thp_50_JFI = smooth(thp_50_JFI, weight=0.9)
    thp_100_JFI = smooth(thp_100_JFI, weight=0.9)
    fig, ax = plt.subplots()
    plt.plot(ratios,queue_JFI,  "-+", label='LIT', color='red')
    plt.plot(ratios,thp_1_JFI,  "-d", label='FIT:'+r'$W=1$', color='blue')
    plt.plot(ratios,thp_50_JFI, "-x", label='FIT:'+r'$W=10$', color='green')
    plt.plot(ratios,thp_10_JFI, "-D", label='FIT:'+r'$W=50$', color='orange')
    plt.plot(ratios,thp_100_JFI, "-_", label='FIT:'+r'$W=100$', color='black')
    x = np.linspace(0.5,0.875)
    y = [1.0]*len(x)
    plt.plot(x,y,'--')
    bottom, top = plt.ylim()
    plt.ylim(bottom*0.9,top)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$',font)
    plt.ylabel("Jain fairness index",font)
    plt.legend(fontsize=12)
    # plt.show() 
    fig.savefig('./pics/window_fairness.pdf',dpi=800,format='pdf')
    plt.clf()

def plot_duration():
    d10 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_19_duration=10_exp','Average-travel-time.txt'),header=None).values.reshape(-1)
    d10 = smooth(d10, weight=0.9)
    d15 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_15_duration=15_exp','Average-travel-time.txt'),header=None).values.reshape(-1)
    d15 = smooth(d15, weight=0.9)
    d20 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_16_duration=20_exp','Average-travel-time.txt'),header=None).values.reshape(-1)
    d20 = smooth(d20, weight=0.9)
    d25 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_17_duration=25_exp','Average-travel-time.txt'),header=None).values.reshape(-1)
    d25 = smooth(d25, weight=0.9)
    d30 = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_18_duration=30_exp','Average-travel-time.txt'),header=None).values.reshape(-1)
    d30 = smooth(d30, weight=0.9)
    # queue = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/queue/result_1_duration=20_queue_500','Average-travel-time.txt'),header=None).values.reshape(-1)
    # queue = smooth(queue)
    
    x = [round(0.5+i*0.4/10, 2) for i in range(10)]
    fig, ax = plt.subplots()
    plt.figure(1)
    # plt.plot(x, queue, label='LIT', color='red')
    plt.plot(x, d10, "-+", label='FIT:'+r'$\Delta t=10$', color='green')
    plt.plot(x, d15, "-d", label='FIT:'+r'$\Delta t=15$', color='blue')
    plt.plot(x, d20, "-x", label='FIT:'+r'$\Delta t=20$', color='orange')
    plt.plot(x, d25, "-D", label='FIT:'+r'$\Delta t=25$', color='red')
    plt.plot(x, d30, "-_", label='FIT:'+r'$\Delta t=30$', color='black')
    bottom, top = plt.ylim()
    plt.ylim(bottom*0.8,top*1.1)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$', font)
    plt.ylabel("Average travel time(s)", font)
    plt.legend(fontsize=12)
    fig.savefig('./pics/duration_efficiency.pdf', dpi=800, format='pdf')
    plt.clf()

    rl_path = './save/single4/dqn/test/'
    d10_delay, d15_delay, d20_delay, d25_delay, d30_delay = [],[],[],[],[]
    d10_JFI, d15_JFI, d20_JFI, d25_JFI, d30_JFI  = [],[],[],[],[]
    queue_delay = []
    queue_JFI = []
    lanes= ['E2TL_1','E2TL_2','N2TL_1','N2TL_2','S2TL_1','S2TL_2','W2TL_1','W2TL_2']
    for l in lanes:
        d10_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_19_duration=10_exp/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        d15_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_15_duration=15_exp/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        d20_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_16_duration=20_exp/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        d25_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_17_duration=25_exp/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        d30_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_18_duration=30_exp/Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))
        # queue_delay.append(pd.read_csv(os.path.join(rl_path,"queue/result_1_duration=20_queue_500/Delay-of-%s.txt"%(l)), header=None).values.reshape(-1))
    # queue_delay = np.array(queue_delay)
    d10_delay = np.array(d10_delay)
    d15_delay = np.array(d15_delay)
    d20_delay = np.array(d20_delay)
    d25_delay = np.array(d25_delay)
    d30_delay = np.array(d30_delay)

    #calc JFI
    for i in range(d15_delay.shape[1]):
        # queue_JFI.append(JFI(queue_delay[:,i]))
        d10_JFI.append(JFI(d10_delay[:,i]))
        d15_JFI.append(JFI(d15_delay[:,i]))
        d20_JFI.append(JFI(d20_delay[:,i]))
        d25_JFI.append(JFI(d25_delay[:,i]))
        d30_JFI.append(JFI(d30_delay[:,i]))

    ratios = [round(0.5+i*0.4/10, 2) for i in range(10)]
    # queue_JFI = smooth(queue_JFI, weight=0.9)
    d10_JFI = smooth(d10_JFI, weight=0.94)
    d15_JFI = smooth(d15_JFI, weight=0.94)
    d20_JFI = smooth(d20_JFI, weight=0.94)
    d25_JFI = smooth(d25_JFI, weight=0.94)
    d30_JFI = smooth(d30_JFI, weight=0.94)
    fig, ax = plt.subplots()
    # plt.plot(ratios,queue_JFI,label='LIT', color='red')
    plt.plot(ratios,d10_JFI,label='FIT:'+r'$\Delta t=10$', color='green')
    plt.plot(ratios,d15_JFI,label='FIT:'+r'$\Delta t=15$', color='blue')
    plt.plot(ratios,d20_JFI,label='FIT:'+r'$\Delta t=20$', color='orange')
    # plt.plot(ratios,d25_JFI,label='FAT:'+r'$\Delta t=25$', color='green')
    plt.plot(ratios,d30_JFI,label='FIT:'+r'$\Delta t=30$', color='black')
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$',font)
    plt.ylabel("Jain fairness index",font)
    plt.legend(fontsize=12)
    plt.show() 
    fig.savefig('./pics/duration_fairness.pdf',dpi=800,format='pdf')
    plt.clf()

def plot_throughput():
    thp = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/throughput/result_7_duration=20__exp_biased_500','Average-travel-time.txt'),header=None).values.reshape(-1)
    thp = smooth(thp)
    uniform = pd.read_csv(os.path.join(os.getcwd(),'save/uniform/single4','Average-travel-time.txt'),header=None).values.reshape(-1)
    uniform = smooth(uniform)
    queue = pd.read_csv(os.path.join(os.getcwd(),'save/single4/dqn/test/queue/result_1_duration=20_queue_500','Average-travel-time.txt'),header=None).values.reshape(-1)
    queue = smooth(queue)
    sotl = pd.read_csv(os.path.join(os.getcwd(),'save/sotl/single4','Average-travel-time.txt'),header=None).values.reshape(-1)
    sotl = smooth(sotl)
    x = [round(0.5+i*0.4/10, 2) for i in range(10)]
    fig, ax = plt.subplots()
    plt.figure(1)
    plt.plot(x,uniform, "-s", label="FT")
    plt.plot(x,sotl, "-p", label='SOTL')
    plt.plot(x,queue, "-*", label='LIT')
    plt.plot(x,thp, "-h", label="FIT")
    bottom, top = plt.ylim()
    plt.ylim(bottom*0.8,top*1.2)
    # print(bottom, top)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$', font)
    plt.ylabel("Average travel time(s)", font)
    plt.legend(fontsize=12)
    # plt.show()
    fig.savefig('./pics/efficiency.pdf',dpi=800,format='pdf')
    plt.clf()
    

def plot_imbalance():
    imbalance = {}
    dataE = pd.read_csv(os.path.join(os.getcwd(),'save/imbalance/dqn/test/queue/result_1_imbalance/Detail-Delay-of-E2TL_0.txt'),header=None)
    dataN = pd.read_csv(os.path.join(os.getcwd(),'save/imbalance/dqn/test/queue/result_1_imbalance/Detail-Delay-of-N2TL_0.txt'),header=None)
    dataS = pd.read_csv(os.path.join(os.getcwd(),'save/imbalance/dqn/test/queue/result_1_imbalance/Detail-Delay-of-S2TL_0.txt'),header=None)
    dataW = pd.read_csv(os.path.join(os.getcwd(),'save/imbalance/dqn/test/queue/result_1_imbalance/Detail-Delay-of-W2TL_0.txt'),header=None)
    imbalance = pd.concat([dataE,dataW,dataN,dataS],axis=1,ignore_index=True)
    imbalance.columns = ['E','W','N','S']
    fig, ax = plt.subplots()
    font = {'weight':'normal','size':20}

    plt.figure(1)
    sns.boxplot(data=imbalance)
    plt.ylabel('delay',font)
    plt.xlabel(fot)
    # plt.show()
    fig.savefig('boxplot.pdf',dpi=800,format='pdf')
    plt.clf()


def plot_fairness():
    test_episodes = 20
    rl_path = './save/single4/dqn/test/'
    uniform_path = './save/uniform/single4/'
    sotl_path = './save/sotl/single4'
    uniform_delay = []
    uniform_JFI = []
    queue_delay = []
    queue_JFI = []
    thp_delay = []
    thp_JFI = []
    sotl_delay = []
    sotl_JFI = []
    lanes= ['E2TL_1','E2TL_2','N2TL_1','N2TL_2','S2TL_1','S2TL_2','W2TL_1','W2TL_2']
    #load data
    for l in lanes:
        uniform_delay.append(pd.read_csv(os.path.join(uniform_path,"Delay-of-%s.txt"%(l)), header=None).values.reshape(-1))
        queue_delay.append(pd.read_csv(os.path.join(rl_path,"queue/result_1_duration=20_queue_500/Delay-of-%s.txt"%(l)), header=None).values.reshape(-1))
        thp_delay.append(pd.read_csv(os.path.join(rl_path,"throughput/result_7_duration=20__exp_biased_500/Delay-of-%s.txt"%(l)), header=None).values.reshape(-1))
        sotl_delay.append(pd.read_csv(os.path.join(sotl_path,"Delay-of-%s.txt"%(l)),header=None).values.reshape(-1))

    uniform_delay = np.array(uniform_delay)
    queue_delay = np.array(queue_delay)
    thp_delay = np.array(thp_delay)
    sotl_delay = np.array(sotl_delay)
    #calc JFI
    for i in range(thp_delay.shape[1]):
        uniform_JFI.append(JFI(uniform_delay[:,i]))
        queue_JFI.append(JFI(queue_delay[:,i]))
        thp_JFI.append(JFI(thp_delay[:,i]))
        sotl_JFI.append(JFI(sotl_delay[:,i]))

    ratios = [round(0.5+i*0.4/10, 2) for i in range(10)]
    queue_JFI = smooth(queue_JFI, weight=0.8)
    uniform_JFI = smooth(uniform_JFI, weight=0.8)
    thp_JFI = smooth(thp_JFI, weight=0.8)
    sotl_JFI = smooth(sotl_JFI, weight=0.8)
    fig, ax = plt.subplots()
    # plt.plot(ratios,queue_JFI,label="queue")
    plt.plot(ratios,uniform_JFI, "-s", label="FT")
    plt.plot(ratios,sotl_JFI, "-p", label="SOTL")
    plt.plot(ratios,queue_JFI, "-*", label='LIT')
    plt.plot(ratios,thp_JFI, "-h", label="FIT")
    x = np.linspace(0.5,0.875)
    y = [1.0]*len(x)
    plt.plot(x,y,'--')
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel(r'$\rho$', font)
    plt.ylabel("Jain fairness index", font)
    plt.legend(loc="lower left",fontsize=12)
    plt.show() 
    fig.savefig('./pics/fairness.pdf',dpi=800,format='pdf')
    plt.clf()

def plot_lastvehicle():
    x = [1,2,3,4]
    y = [456.94,164.73,56.80,70.62]
    fig =plt.figure()
    plt.bar(x,y,align='center', width= 0.4, tick_label=['FT','SOTL','LIT','FIT'])
    bottom, top = plt.ylim()
    plt.ylim(bottom, top*1.1)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel("Methods", font)
    plt.ylabel("Average travel time(s)", font)

    for a, b in zip(x,y):
        plt.text(a,b,'%.00f'%b,ha='center',va='bottom',fontsize=12)
    plt.show()
    fig.savefig('./pics/lastvehicle_ave.pdf',dpi=800,format='pdf')

def plot_lastvehilce_delay():
    x = [1,2,3,4]
    y = [55,60,280,70]
    fig =plt.figure()
    plt.bar(x,y,align='center', width= 0.4, tick_label=['FT','SOTL','LIT','FIT'])
    bottom, top = plt.ylim()
    plt.ylim(bottom, top*1.1)
    font = {
    'weight' : 'normal',
    'size'   : 15,
    }
    plt.xlabel("Methods", font)
    plt.ylabel("Response time(s)", font)
    for a, b in zip(x,y):
        plt.text(a,b,'%.00f'%b,ha='center',va='bottom',fontsize=12)

    # plt.legend("")
    plt.show()
    fig.savefig('./pics/lastvehicle_delay.pdf',dpi=800,format='pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tsc",type=str,default='dqn',dest='tsc',help='traffic singal control. options:uniform,dqn')
    parser.add_argument("-roadnet",type=str,default='single',dest='roadnet',help = 'raodnet flie name')
    parser.add_argument("-mode",type=str,default='train',dest='mode')
    parser.add_argument("-m",type=str,default='queue',dest='metric',help='options:[queue,log_queue,throughput]')
    args = parser.parse_args()
    # save_path = set_save_path(args.roadnet, args.tsc, args.mode, args.metric)
    # plot_convergence(save_path)
    # road_list = ['major','minor']
    # for road in road_list:
    #     plot_delay(road)
    # bias = [0.0, 0.25, 0.4]
    # for b in bias:
    #     plot_tail(b)
    # plot_imbalance()
    # plot_fairness()
    # plot_window()
    # plot_des()
    # plot_duration()
    # plot_throughput()
    # plot()
    # plot_window()
    # plot_lastvehicle()
    plot_lastvehilce_delay()
    # plot_duration()