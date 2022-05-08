import tensorflow as tf
import keras
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)

import gym
from keras import models
from keras import layers
# from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from Algorithms.Algorithm import algorithm 


# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))


class DQN(algorithm):
    def __init__(self,env):
        # parameters for RL
        self.env=env
        self.gamma=0.99
        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.episode_num=400
        self.steps_over_one_training = []
        self.steps_over_episodes = []

        # parameters for Neural Network training
        self.training_epochs = 5
        self.timeout_steps=200 #max is 200
        self.replay_buffer_size=deque(maxlen=20000)
        self.num_selected_from_replay_buffer=32
        self.trainNetwork=self.createNetwork()
        self.targetNetwork=self.createNetwork()
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())
        

    # create the Keras neural network
    def createNetwork(self):
      # with tf.device('/device:GPU:0'):

        # construct fully-connected neural network layer      
        model = models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(48, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n,activation='linear'))
        # model.compile(optimizer=optimizers.RMSprop(lr=self.learning_rate), loss=losses.mean_squared_error)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # choose action at a certain state based on greedy 
    def chooseAction(self,state):

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action=np.argmax(self.trainNetwork.predict(state)[0])

        return action

    
    # use matrix to speed-up training
    def trainUsingReplayMemory_Boost(self):
        # with tf.device('/device:GPU:0'):
          if len(self.replay_buffer_size) < self.num_selected_from_replay_buffer:
              return
          samples = random.sample(self.replay_buffer_size,self.num_selected_from_replay_buffer)
          npsamples = np.array(samples)
          states_temp, actions_temp, rewards_temp, nextstates_temp, dones_temp = np.hsplit(npsamples, 5)
          states = np.concatenate((np.squeeze(states_temp[:])), axis = 0)
          rewards = rewards_temp.reshape(self.num_selected_from_replay_buffer,).astype(float)
          targets = self.trainNetwork.predict(states)
          newstates = np.concatenate(np.concatenate(nextstates_temp))
          dones = np.concatenate(dones_temp).astype(bool)
          notdones = ~dones
          notdones = notdones.astype(float)
          dones = dones.astype(float)
          Q_futures = self.targetNetwork.predict(newstates).max(axis = 1)
          targets[(np.arange(self.num_selected_from_replay_buffer), actions_temp.reshape(self.num_selected_from_replay_buffer,).astype(int))] = rewards * dones + (rewards + Q_futures * self.gamma)*notdones
          self.trainNetwork.fit(states, targets, epochs=1, verbose=0)


    # using replay memory for training, to avoid overshot
    def trainUsingReplayMemory(self):

        # check if we have got enough data to train in the buffer
        # we need to get at least "self.num_selected_from_replay_buffer" number of data
        if len(self.replay_buffer_size) < self.num_selected_from_replay_buffer:
            return

        # do the sampling
        samples = random.sample(self.replay_buffer_size,self.num_selected_from_replay_buffer)

        # do the prediction for the target network
        states_list = []
        newStates_list=[]
        for sample in samples:
            state, action, reward, new_state, done = sample
            states_list.append(state)
            newStates_list.append(new_state)

        newArray = np.array(states_list)
        states_list = newArray.reshape(self.num_selected_from_replay_buffer, 2)

        newArray2 = np.array(newStates_list)
        newStates_list = newArray2.reshape(self.num_selected_from_replay_buffer, 2)

        targets_for_train = self.trainNetwork.predict(states_list)
        new_state_targets=self.targetNetwork.predict(newStates_list)

        i=0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets_for_train[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i+=1

        self.trainNetwork.fit(states_list, targets_for_train, epochs=1, verbose=0)


    def trainNN(self,currentState,eps):
        rewardSum = 0
        max_position=-99

        for i in range(self.timeout_steps):
            bestAction = self.chooseAction(currentState)

            new_state, reward, done, _ = env.step(bestAction)

            new_state = new_state.reshape(1, 2)

            # # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]


            # # Adjust reward for task completion
            if new_state[0][0] >= 0.5:
                reward += 10

            self.replay_buffer_size.append([currentState, bestAction, reward, new_state, done])

            # this should be work when training on GPU
            self.trainUsingReplayMemory_Boost()

            rewardSum += reward

            currentState = new_state

            if done:
                break
        if i >= 199:
            print("Failed to finish task in epsoide {}".format(eps))
            
        else:
            print("Success in epsoide {}, used {} iterations!".format(eps, i))
            self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(eps))
        self.steps_over_episodes.append(i)

        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

        print("now epsilon is {}, the reward is {} maxPosition is {}".format(max(self.epsilon_min, self.epsilon), rewardSum,max_position))
        self.epsilon -= self.epsilon_decay
    
    def testNN(self, N):
      model=models.load_model('./trainNetworkInEPS390.h5')

      # repeat the model with  times
      for i in range(N):
          curr_state = env.reset().reshape(1, 2)
          rewardSum=0
          steps_for_one_episode = []

          # 200 steps maximum for one episode
          for t in range(self.timeout_steps):
              env.render()
              action = np.argmax(model.predict(curr_state)[0])

              next_state, reward, done, info = env.step(action)

              next_state = next_state.reshape(1, 2)

              curr_state=next_state

              rewardSum+=reward

              steps += 1
              if done:
                  steps_for_one_episode.append(steps)
                  print("Episode finished after {} timesteps reward is {}".format(t+1,rewardSum))
                  break
        

    # draw the result of N training times
    def plot_graphs(self, env, episodes, runs, params):
        with open("./result_from_colab.txt", "r") as f:
            all_lines = f.readlines()
            all_epoch_result = []
            all_episodes_result = []
            count = 0
            for line in all_lines:
                words_list = line.split(" ")
                count += 1
                if words_list[0] == "now":
                    word = words_list[7]
                    num = -float(word)
                    if  len(all_episodes_result) < 400:
                        all_episodes_result.append(num)
                    elif len(all_episodes_result) >= 400:
                        # print("current length is: ", len(all_episodes_result))
                        all_epoch_result.append(all_episodes_result.copy())
                        all_episodes_result = []
                        all_episodes_result.append(num)
            all_epoch_result.append(all_episodes_result)
           
            all_epochs_result_array = np.array(all_epoch_result)
            data_mean = np.mean(all_epochs_result_array, axis = 0)
            data_std = np.std(all_epochs_result_array, axis= 0)
            plt.plot(data_mean, 'b-')
            plt.fill_between(np.arange(len(data_mean)), data_mean - data_std, data_mean + data_std, color='red')
            plt.xlabel("Number of Episodes")
            plt.ylabel("Number of Steps to Goal")
            plt.title("Deep Q Learning between episodes and step to goal")
            plt.show()
        return 

    def train(self, env, N, params):
      # Train the network for 5 times, each time with 500 episodes
      for i in range(N):
        self.steps_over_episodes = []
        for eps in range(self.episode_num):
            currentState=env.reset().reshape(1,2)
            self.trainNN(currentState, eps)
        self.steps_over_one_training.append(self.steps_over_episodes)

      with open("./result_from_colab.txt", 'a') as f:
        f.write(self.steps_over_one_training)

    def run_game(self, final_params):
       self.gamma, self.learning_rate, self.epsilon, self.training_epochs = final_params
       self.testNN(self.training_epochs)
       return

    # some getters and setters
    def setGamma(self, gamma):
        self.gamma = gamma
        return

    def setLearningrate(self, learning_rate):
        self.learning_rate = learning_rate
        return

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
        return

    def setEpisodeNum(self, episode_num):
        self.episode_num = episode_num
        return

    def getGamma(self):
        return self.gamma

    def getLearningrate(self):
        return self.learning_rate

    def getEpsilon(self):
        return self.epsilon

    def getEpisodeNum(self):
        return self.episode_num

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    dqn = DQN(env)
    params = [0.9, 0.001, 0.1]
    dqn.train(env, 5, params)
    episodes = dqn.getEpisodeNum()
    dqn.plot_graphs(env, episodes, 5, params)