import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from memory import Memory

class DQN:
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = 400
        self.n_layers = 4
        self.batch_size = 100
        self.lr = 0.001

        self.model = self.build_model()
    
    def build_model(self):
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(self.width,activation='relu')(inputs)
        
        for _ in range(self.n_layers):
            x = layers.Dense(self.width,activation='relu')(x)
        outputs = layers.Dense(self.output_dim,activation='linear')(x)

        model = keras.Model(inputs=inputs,outputs=outputs,name='dqn')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self.lr))
        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self.input_dim])
        # state = np.expand_dims(state,-1)
        return self.model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        # print(states)
        # print(np.shape(states))
        # states = np.reshape(states,[self.input_dim,])
        return self.model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self.model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self.model.save(os.path.join(path, 'trained_model.h5'))
    
    def load_model(self,path):
        model_file_path = os.path.join(path,'trained_model.h5')
        if os.path.isfile(model_file_path):
            self.model = load_model(model_file_path)
        else:
            sys.exit("Model not found")

class DqnAgent():
    def __init__(self,batch_size,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size,action_size)
        self.batch_size = batch_size
        self.gamma = 0.75
        self.replay = Memory(size_max=100000,size_min=600)
    
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon
    
    def predict(self,state):
        q_state = self.model.predict_one(state)
        action = np.argmax(q_state)
        return action
    
    def get_action(self,state):
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            q_state = self.model.predict_one(state)
            action = np.argmax(q_state)
        return action


    def store_experience(self,state,action,next_state,reward,terminal):
        experience = {'s':state,'a':action,'s_':next_state,'r':reward,'t':terminal}
        self.replay.add_sample(experience)
    
    def save_model(self,path):
        self.model.save_model(path)
    
    def load_model(self,path):
        self.model.load_model(path)
    
    def train_batch(self):
        samples = self.replay.get_samples(self.batch_size)
        if len(samples) > 0:
            states = np.array([sample['s'] for sample in samples])
            next_states = np.array([sample['s_'] for sample in samples])
            
            #prediction
            q_s = self.model.predict_batch(states)
            q_s_ = self.model.predict_batch(next_states)

            x = np.zeros((len(samples),self.state_size))
            y = np.zeros((len(samples),self.action_size))

            for i, sample in enumerate(samples):
                s,a,s_,r,t = sample['s'],sample['a'],sample['s_'],sample['r'],sample['t']
                current_q = q_s[i]
                current_q[a] = r + self.gamma*np.amax(q_s_[i])
                x[i] = s
                y[i] = current_q
            self.model.train_batch(x,y)


        


