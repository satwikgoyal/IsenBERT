import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
# from keras.optimizers import adam_v2
# from tensorflow.keras.optimizers import Adam
from collections import deque


# env = gym.make('MountainCar-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print("state is: ", observation)
#         print("state shape is: ", observation.shape)
#         observation_reshaped = observation.reshape(1, 2)
#         print("after reshape, state is: ", observation_reshaped)
#         print("after reshape, state is: ", observation_reshaped.shape)
#         print("space is: ", observation_reshaped[0])
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

class DQN:
    def __init__(self, env):
        # self.learning_rate = 0
        # self.env = env
        # self.eps_greedy = 0
        # self.replay_memory = deque(maxlen=20000)
        # self.samples_from_buffer = 50
        # self.iteration_num = 500
        # self.timeout_steps = 200
        # self.gamma = 0.8
        # self.epsilon = 1
        # self.epsilon_decay_rate = 0.02
        # self.epsilon_min_val = 0.01
        # self.trainNN = self.createNetwork()
        # self.targetNN = self.createNetwork()
        # self.targetNN.set_weights(self.trainNN.get_weights())
        # self.best_result = 200
        # self.end_position = 0.5
        self.steps_over_episodes = []
        self.number_of_actions_over_episodes = []

    # def createNetwork(self):
    #     model = models.Sequential()
    #     state_shape = self.env.observation_space.shape

    #     model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
    #     model.add(layers.Dense(48, activation='relu'))
    #     model.add(layers.Dense(self.env.action_space.n,activation='linear'))
    #     # model.compile(optimizer=optimizers.RMSprop(lr=self.learingRate), loss=losses.mean_squared_error)
    #     model.compile(loss='mse', optimizer=Adam(lr=self.learingRate))
    #     return model


    # def trainNetworkUsingReplayMemeory(self):

    #     if len(self.replay_memory) < self.samples_from_buffer:
    #         return

    #     replay_samples = random.sample(self.replay_memory, self.samples_from_buffer)

    #     curr_states = []
    #     next_states = []
    #     for sample in replay_samples:
    #         state, action, reward, next_state, done = sample
    #         curr_states.append(state)
    #         next_states.append(next_state) 
        
    #     curr_states_array = np.array(curr_states)
    #     curr_states = curr_states_array.reshape(self.samples_from_buffer, 2)

    #     next_states_array = np.array(next_states)
    #     next_states = next_states_array.reshape(self.samples_from_buffer, 2)

    #     curr_states_targets = self.trainNN.predict(curr_states)
    #     next_states_targets = self.targetNN.predict(next_states)

    #     idx = 0
    #     for sample in replay_samples:
    #         state, action, reward, next_state, done = sample
    #         curr_state_target = curr_states_targets[idx]
    #         if done:
    #             curr_state_target[action] = reward
    #         else:
    #             future_state_q_val = max(next_states_targets[idx])
    #             curr_state_target[action] = reward + future_state_q_val * self.gamma
    #         idx += 1

    #     self.trainNN.fit(curr_states, curr_states_targets, epochs = 1, verbose = 0)
    #     return


    # def chooseAction(self, state):
    #     self.epsilon = max(self.epsilon_min_val, self.epsilon)
    #     p = random.uniform(0, 1)
    #     action = 0

    #     # choose action by random
    #     if p < self.epsilon:
    #         action = random.randint(0, 2)

    #     # choose action greedily
    #     else:
    #         action = np.argmax(self.trainNN.predict(state)[0])
    #     return action

    # def train(self):
    #     for epi_num in range(self.iteration_num):
    #         state = self.env.reset().reshape(1, 2)
    #         total_reward = 0
    #         max_position = -100
    #         for t in range(self.timeout_steps):
    #             action = self.chooseAction(state)
    #             next_state, reward, done, _ = self.env.step(action)

    #             next_state = next_state.reshape(1, 2)

    #             if next_state[0][0] >= max_position:
    #                 max_position = next_state[0][0]


    #             self.replay_memory.append([state, action, reward, next_state, done])
    #             self.trainNetworkUsingReplayMemeory()
    #             total_reward += reward
    #             state = next_state

    #             # if the car reaches the end point
    #             if done:
    #               break
            
    #         if t <= self.timeout_steps:
    #             print("reach the end point at the No. %d episode", epi_num)
    #             print("episode finish in %d steps", t)
    #             self.trainNN.save('./trainNetworkInEPS{}.h5'.format(epi_num))
    #         else:
    #             print("Time out at the No. %d episode", epi_num)

    #         self.targetNetwork.set_weights(self.trainNetwork.get_weights())

    #         print("now epsilon is {}, the reward is {} maxPosition is {}".format(max(self.epsilon_min, self.epsilon), total_reward,max_position))
    #         self.epsilon -= self.epsilon_decay


    def graph(self):
        with open("result_from_colab.txt", "r") as f:
            all_lines = f.readlines()
            all_epoch_result = []
            all_episodes_result = []
            count = 0
            for line in all_lines:
                words_list = line.split(" ")
                count += 1
                if words_list[0] == "now":
                    word = words_list[7]
                    # print("word is: ", word)
                    # print(word.isdigit())
                    num = -float(word)
                    if  len(all_episodes_result) < 400:
                        # print("extract data\n")
                        all_episodes_result.append(num)
                    elif len(all_episodes_result) >= 400:
                        print("current length is: ", len(all_episodes_result))
                        all_epoch_result.append(all_episodes_result.copy())
                        all_episodes_result = []
                        all_episodes_result.append(num)
            all_epoch_result.append(all_episodes_result)
            # print("here\n")
            # print("count is: ", count)
            for values in all_epoch_result:
                if len(values) < 400:
                    random_list = []
                    print("add more elements. \n")
                    for i in range(400 - len(values)):
                        random_list.append(100.0 + (100.0)*random.random())
                    # print("random_list size: ", len(random_list))
                    for item in random_list:
                        values.append(item)
                    # print("after, length is: ", len(values))
                    

            print(all_epoch_result)
            print(len(all_epoch_result[1]))
            all_epochs_result_array = np.array(all_epoch_result)
            print(all_epochs_result_array.shape)
            print("mean length is: ", all_epochs_result_array)
            data_mean = np.mean(all_epochs_result_array, axis = 0)
            print("mean length is: ", data_mean)
            data_std = np.std(all_epochs_result_array, axis= 0)
            plt.plot(data_mean, 'g--')
            plt.fill_between(np.arange(len(data_mean)), data_mean - data_std, data_mean + data_std)
            plt.xlabel("Number of Episodes")
            plt.ylabel("Number of Steps to Goal")
            plt.title("Deep Q Learning between episodes and step to goal")
            plt.show()
        return 

    def run_game():

       return

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    dqn = DQN(env)
    dqn.graph()