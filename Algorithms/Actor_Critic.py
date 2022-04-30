from Algorithms.Algorithm import algorithm
import numpy as np
import gym
import matplotlib.pyplot as plt

class actor_critic(algorithm):

    ### HELPER METHODS ###
            
    ## normalize functions between 0 and 1 cosine
    def normalize(self,state):

        s0, s1 = state

        # s0 = x, s1 = v
        new_s0 = (s0 + 1.2) / (0.5 + 1.2)
        new_s1 = (s1 + 0.7) / (0.7 + 0.7)

        return [new_s0, new_s1]

    ## phi full version # size (M+1)^2 where 2 is dim of state
    def compute_phi(self,M, s):

        # normalize
        state = self.normalize(s)

        s0, s1 = state

        # full fourier basis function using cosine
        phi = [1.0]

        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi *s0))

        for j in range(1, M+1):
            phi.append(np.cos(j*np.pi *s1))

        for a in range(1, M+1):
            for b in range(1, M+1):
                phi.append(np.cos(np.pi *(a*s0)*(b*s1)))

        return np.array(phi)


    # softmax to get the next action, probability of action, and gradient for actor critic
    def softmax(self,epsilon, theta, phi):

        policy = np.zeros(theta.shape[0])

        # compute theta(s,a) full
        theta_sa = theta.dot(phi)

        # compute softmax
        for i in range(theta.shape[0]):

            policy[i] = np.exp(epsilon*theta_sa[i]) / np.sum(np.exp(epsilon*theta_sa))

        actions = np.array([0,1,2])

        # sample next action
        action = np.random.choice(actions, p =policy)

        prob = policy[np.argwhere(actions== action)[0][0]]

        # gradient of softmax
        grad = np.empty((0,len(phi)))

        # compute gradient
        for j in range(len(policy)):

            if action == actions[j]:
                grad = np.vstack((grad,(1-prob) * phi))
            else:
                grad = np.vstack((grad,(-1)*(policy[j] * phi)))
    
        return action, prob, grad


    ### train method ###
    def train(self,env,N,parameters):

        # parameters for actor critic
        alpha_theta, alpha_w, M1, M2, epsilon = parameters
       
        curve_list = []
        step_list = []
        
        theta = np.zeros((3, (M1+1)**2))
        w = np.zeros(((M2+1)**2))
        
        ac_counter = 0

        for i in range(N):

            # get initial state
            state = env.reset()

            gamma = 1.0

            step_counter = 0
            
            finish = False

            # loop until reach goal state or terminate after 200 actions
            while finish != True and step_counter <200:

                # phi for softmax
                phi_soft = self.compute_phi(M1,state)

                # get action
                action, prob, grad = self.softmax(epsilon, theta, phi_soft)
                
                # next state, reward, boolean for goal state
                next_state, reward, finish, _ = env.step(action)

                # phi function approx
                phi_approx = self.compute_phi(M2, state)

                # phi for next state
                phi_approx_p = self.compute_phi(M2, next_state) 

                # delta                                 
                delta =  reward + (gamma*(w.dot(phi_approx_p))) - (w.dot(phi_approx))

                # update w
                w = w + (alpha_w * delta * phi_approx)

                # update theta
                theta = theta + (alpha_theta *delta * grad) 

                state = next_state

                ac_counter += 1
                step_counter += 1

            curve_list.append(ac_counter)
            step_list.append(step_counter)
            
        return curve_list, step_list, theta


    def plot_graphs(self,env, eps, runs, parameters):

        curve_list = []
        step_list = []

        for i in range(runs):
            print(i)
            curve, step, theta = self.train(env, eps, parameters)
            curve_list.append(curve)
            step_list.append(step)

        # compute mean and standard deviation
        curve = np.mean(curve_list, axis = 0)
        step = np.mean(step_list, axis = 0)
        std_step = np.std(step_list, axis = 0)

        # plot total number of actions vs episodes
        plt.plot(curve, range(eps))
        plt.xlabel("Number of Actions")
        plt.ylabel("Number of Episodes")
        plt.title("Actor-Critic Curve between actions and episodes")
        plt.show()

        # plot episodes vs steps to the goal state
        plt.errorbar(range(eps), step, std_step, ecolor="red")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Number of steps to goal")
        plt.ylim(75, 250)
        plt.yticks(range(100,250,50))
        plt.title("Actor-Critic Curve between episodes and step to goal")
        plt.show()


    def run_game(self,final_parameters):

        theta, M1, epsilon = final_parameters
    
        env = gym.make('MountainCar-v0')
        state = env.reset()
        
        done = False
        
        counter = 0

        while done != True:

            print(counter)

            # phi softmax
            phi_soft = self.compute_phi(M1,state) # change to full fourier basis

            # get action
            action, _, _ = self.softmax(epsilon, theta, phi_soft)

            env.render()

            # next state, boolean for goal state
            new_state, _, done, _ = env.step(action)

            #update state
            state = new_state
            
            counter += 1


        env.close()