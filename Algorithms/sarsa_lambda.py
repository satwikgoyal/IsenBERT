
import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

class sarsa_lambda(algorithm):
    def train():
      alpha = 0.37
      gamma = 0.975
      q_len = 17
      lamb = 0.88
      epsilon = 0.99
      epsilon_start = 1
      episodes = 500
      epsilon_end = episodes // 2
      high = env.observation_space.high
      low = env.observation_space.low
      discrete_size_list = [q_len for i in range(0, len(high))]
      discrete_os_win_size = (high - low) / discrete_size_list
      q_table = np.zeros((discrete_size_list + [env.action_space.n]))
      e_table = np.zeros((discrete_size_list + [env.action_space.n]))
      epsilon_decay_value = epsilon / (epsilon_end - epsilon_start)
      scores = []
      curve_list = []
      step_list = []
      ac_counter = 0
      env._max_episode_steps = 200
      low_state = env.observation_space.low
      step_list = []
      for episode in range(episodes):
        # initiate reward every episode
        step_counter = 0
        score = 0

        state = env.reset()
        size = (high - low) / discrete_size_list
        state_new = (state - low) // size
        discrete_state = tuple(int(i) for i in state_new)

        if np.random.random() < epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[discrete_state])
        # action = take_action(state, epsilon)

        #   reset e_table
        e_table = np.zeros(discrete_size_list + [env.action_space.n])

        done = False

        while not done:

            next_state, reward, done, _ = env.step(action)
            score += reward
            size = (high - low) / discrete_size_list
            state_new = (state - low) // size
            discrete_state = tuple(int(i) for i in state_new)

            if np.random.random() < epsilon:
                next_action = np.random.randint(0, env.action_space.n)
            else:
                next_action = np.argmax(q_table[discrete_state])

            if not done:
                ac_counter += 1
                step_counter += 1
                size = (high - low) / discrete_size_list
                state_new = (state - low) // size
                state2 = tuple(int(i) for i in state_new)
                next_state_new = (next_state - low) // size
                next_state2 = tuple(int(i) for i in next_state_new)

                target = reward + gamma * q_table[next_state2][next_action]
                delta = target - q_table[state2][action]

                e_table[state2][action] += 1
                q_table += alpha * delta * e_table
                e_table = gamma * lamb * e_table

            elif next_state[0] >= 0.436:
                q_table[state2][action] = 0


            state = next_state
            action = next_action

        curve_list.append(ac_counter)
        step_list.append(step_counter)

        if epsilon_end >= episode >= epsilon_start:
            epsilon -= epsilon_decay_value

        scores.append(score)
      return step_list, curve_list, q_table




    def graph():
      step_list, curve_list = train()
      curve = np.mean(curve_list, axis = 0)
      step = np.mean(step_list, axis = 0)
      std_step = np.std(step_list, axis = 0)
      x = [i for i in range(len(step_list))]
      y = step_list
      plt.plot(x, y)

      start_x = 0
      for i in range(len(step_list)):
        if step_list[i] < 199:
          start_x = i
          break

      y = y[start_x:]

      plt.fill_between(x[start_x:],y - std_step, y + std_step, alpha=0.4,edgecolor='#1B2ACC', linewidth=0.3, linestyle='dashdot',  antialiased=True)
      plt.title("Sarsa Lambda between episodes and step to achive")
      plt.xlabel('Episodes')
      plt.ylabel('step_to_achive')
      plt.show()


    def game_run():
      done = False
      state = env.reset()
      _,_,q_table = train()
      while not done:
          q_len = 20
          discrete_size_list = [q_len for i in range(0, len(env.observation_space.high))]
          size = (env.observation_space.high - env.observation_space.low) / discrete_size_list
          state_new = (state - env.observation_space.low) // size
          discrete_state = tuple(int(i) for i in state_new)
          action = np.argmax(q_table[discrete_state])
          next_state, _, done, _ = env.step(action)
          state = next_state
          env.render()

      env.close()
