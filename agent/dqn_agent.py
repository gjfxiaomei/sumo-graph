import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
import numpy as np
from replay_buffer import BasicBuffer

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,action_dim)
        )
    
    def forward(self, state):
        qvals = self.fc(state)
        return qvals

class DQNAgent:
    def __init__(self,state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = BasicBuffer(max_size=100000)
        self.target_update_freq = 50
        self.batch_size = 32
        self.train_epoch = 800
        self.gamma = 0.75
        self.update_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()
    
    def set_epsilon(self, epsilon):
        self.eps = epsilon
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state,action,reward,next_state,done)

    def predict(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        return action
    
    def get_action(self, state):
        u = np.random.uniform()
        # print(u)
        if u < self.eps:
            action = np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().to(self.device)
            qvals = self.model.forward(state)
            action = np.argmax(qvals.cpu().detach().numpy())
        return action

    def compute_loss(self, batch):
        if len(batch) > 0:
            states, actions, rewards, next_states, dones = batch
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            Q_S = self.model.forward(states)
            Q_S_A = Q_S.gather(1,actions.unsqueeze(1)).squeeze(1)

            Q_S2 = self.model.forward(next_states)
            # next_Q = self.model.forward(next_states)
            target_Q_S2 = self.target_model.forward(next_states)
            target_Q_S2_A = target_Q_S2.gather(1, torch.max(Q_S2,1)[1].unsqueeze(1)).squeeze(1)

            expected_Q = rewards.squeeze(1) + self.gamma * target_Q_S2_A

            loss = self.MSE_loss(Q_S_A, expected_Q)
            return loss

    def update(self):
        batch = self.replay_buffer.sample(self.batch_size)

        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def train(self):
        for _ in range(self.train_epoch):
            self.update()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path,"parameter.pkl"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path,"parameter.pkl")))
        self.model.eval()
