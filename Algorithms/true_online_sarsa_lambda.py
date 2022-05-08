from Algorithms.Algorithm import algorithm
import numpy as np
import gym
import matplotlib.pyplot as plt

class true_online(algorithm):
    def train(self, env, N, parameters):
        alpha, gamma, l, ep, b_x, b_v = parameters

        q = np.zeros((b_x, b_v, 3))
        steps_list = []

        for i in range(1, N+1):
            steps = 0
            observation = env.reset()

            if ep>0.05:
                ep-=0.01

            x = observation[0]
            v = observation[1]

            scaled_x = int(((((x - (-1.2))/(0.61 - (-1.2)))*(1-(0))) + (0))//(1/b_x))
            scaled_v = int(((((v - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0))//(1/b_v))

            if np.random.rand()<=ep:
                action = int(np.random.rand()//(1/3))
            else:
                action = np.argmax(q[scaled_x][scaled_v])
            
            q_old = 0
            e = np.zeros((b_x, b_v, 3))

            #episode
            done = False
            total_reward = 0
            while(not done and steps<200):
                # env.render()
                observation, reward, done, info = env.step(action)
                x_next, v_next = observation

                scaled_x_next = int(((((x_next - (-1.2))/(0.61 - (-1.2)))*(1-(0))) + (0))//(1/b_x))
                scaled_v_next = int(((((v_next - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0))//(1/b_v))
                next_action = None

                if np.random.rand()<=ep:
                    next_action = int(np.random.rand()//(1/3))
                else:
                    next_action = np.argmax(q[scaled_x_next][scaled_v_next])
                    
                delta_q = q[scaled_x][scaled_v][action]-q_old
                q_old = q[scaled_x_next][scaled_v_next][next_action]
                delta = reward + gamma*q[scaled_x_next][scaled_v_next][next_action]-q[scaled_x][scaled_v][action]
                e[scaled_x][scaled_v][action] = (1-alpha)*e[scaled_x][scaled_v][action] + 1
                q += alpha*(delta+delta_q)*e
                e = gamma*l*e
                q[scaled_x][scaled_v][action] = q[scaled_x][scaled_v][action] - alpha*delta_q
                
                action = next_action
                scaled_x = scaled_x_next
                scaled_v = scaled_v_next

                steps+=1
                total_reward += reward
            
            steps_list.append(steps)

        return steps_list, q
    
    def plot_graphs(self, env, eps, runs, parameters):
        steps_lists = []
        
        for run in range(runs):
            steps_list, _ = self.train(env, eps, parameters)
            steps_lists.append(steps_list)

        steps_lists_mean = np.mean(steps_lists, axis=0)
        steps_lists_std = np.std(steps_lists, axis=0)

        plt.title("True Online Sarsa Lambda between Episodes and Steps to Goal")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Number of Steps to Goal")
        plt.errorbar(range(eps), steps_lists_mean, steps_lists_std, ecolor="red")
        plt.show()
    
    def run_game(self, env, q, b_x, b_v):
        observation = env.reset()
        x, v = observation
        scaled_x = int(((((x - (-1.2))/(0.61 - (-1.2)))*(1-(0))) + (0))//(1/b_x))
        scaled_v = int(((((v - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0))//(1/b_v))

        action = np.argmax(q[scaled_x][scaled_v])
        done = False
        while(not done):
            env.render()
            observation, reward, done, info = env.step(action)
                
            x, v = observation
            scaled_x = int(((((x - (-1.2))/(0.61 - (-1.2)))*(1-(0))) + (0))//(1/b_x))
            scaled_v = int(((((v - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0))//(1/b_v))
            action = np.argmax(q[scaled_x][scaled_v])


            