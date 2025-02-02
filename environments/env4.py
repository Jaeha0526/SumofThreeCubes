import math
import numpy as np
from .env1 import SumofCubesEnv
from collections import Counter

class SumofCubesEnv4(SumofCubesEnv):
    def __init__(self, args):
        self.num_envs = args.num_envs
        self.k = args.target_k
        self.reward = args.reward
        self.max_n = args.max_n
        self.min_n = args.min_n
        self.i_exp = args.i_exp
        self.ep_length = args.ep_length
        self.num_state = 13

        self.single_observation_space = self.single_observation(np.array([1]*self.num_state))
        self.single_action_space = self.single_action(0)
        # self.num_actions = 3**self.num_state
        self.num_actions = 2 * self.num_state

        self.states = np.zeros((self.num_envs, 6), dtype = np.int32)
        self.observations = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype = np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        # for infos
        self.last_episode_length = 0
        self.max_sum = 0
        self.last_reward = 0

        # initialize
        self.reset()

        # all the records
        self.records = []
        self.solutions = set()
        self.solutionstates = set()


    def reset(self, dones=None):
        if dones is not None :
            for i in range(self.num_envs):
                if dones[i]:
                    # record episode of the most recent terminated one
                    self.last_episode_length = self.episode_lengths[i]
                    self.last_reward = self.single_reward(self.states[i])
                    self.records.append(np.array(self.states[i]))
                    # reset done env
                    self.states[i] = np.random.randint(-3, -2, (self.num_state,))
                    self.observations[i] = self.single_observation(self.states[i])
                    self.episode_lengths[i] = 0
                else:
                    self.episode_lengths[i] += 1

        else :
            self.states = np.random.randint(-3, -1, (self.num_envs, self.num_state))
            self.observations = self.observations_from_states(self.states)
            self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        return self.observations


    def single_observation(self,state):

        return np.array(state)


    def single_reward(self, state):
        a0, a1, a2, a3, a4, b0, b1, b2, b3, c0, c1, c2, c3 = state

        reward = 0

        for i in range(self.min_n,self.max_n+1):

            x = a0 + a1*i + a2*i**2 + a3*i**3 + a4*i**4
            y = b0 + b1*i + b2*i**2 + b3*i**3 - a4*i**4
            z = c0 + c1*i + c2*i**2 + c3*i**3

            k = x**3 + y**3 + z**3
            if abs(x+y+z) == 1 :
                continue

            if abs(k) == self.k:
                reward += self.reward[0]
                sign = int(k/abs(k))
                pair = (sign*x,sign*y,sign*z)

                self.solutions.add(pair)
                self.solutionstates.add(((a0,a1,a2,a3,a4), (b0,b1,b2,b3), (c0,c1,c2,c3), sign, i))

            else :
                reward += self.reward[1] * 2**( - self.reward[2]*abs(k)/ i**5 )
                reward += self.reward[3] * 2**( -1/(1+abs(x)+abs(y)+abs(z)) )

        return reward



    def single_action(self, action):
        # each of number -1, 0, 1   ->   3**6
        # return np.array([
        #     action % 3 - 1,
        #     (action // 3) % 3 - 1,
        #     (action // 9) % 3 - 1,
        #     (action // 27) % 3 - 1,
        #     (action // 81) % 3 - 1,
        #     (action // 243) % 3 - 1
        # ])

        # independently adjust numbers   ->  2 * 13
        action_space = np.array([0]*self.num_state)
        action_space[action % self.num_state] = 2*(action//self.num_state) - 1
        return action_space



    def step(self, actions):
        # take actions and get informations
        zd_actions = [ self.single_action(action) for action in actions]
        self.states = self.states + zd_actions

        # restrict x or y or z is constant
        for i in range(len(self.states)):
            if self.states[i][1] == 0 and self.states[i][2] == 0 and self.states[i][3] == 0 and self.states[i][4] == 0:
                self.states[i][1] = 1
                self.states[i][2] = 1
                self.states[i][3] = 1
                self.states[i][4] = 1

            if self.states[i][6] == 0 and self.states[i][7] == 0 and self.states[i][8] == 0 and self.states[i][4] == 0:
                self.states[i][6] = 1
                self.states[i][7] = 1
                self.states[i][8] = 1
                self.states[i][9] = 1

            if self.states[i][10] == 0 and self.states[i][11] == 0 and self.states[i][12] == 0:
                self.states[i][10] = 1
                self.states[i][11] = 1
                self.states[i][12] = 1


        # get observations from states
        self.observations = self.observations_from_states(self.states)

        # get dones from states
        next_dones = np.array([ 0 if self.episode_lengths[i] < self.ep_length else 1 for i in range(self.num_envs) ])

        # get rewards from states
        rewards = self.rewards_from_states(self.states)

        # reset done envs
        self.reset(next_dones)

        # additional infos
        infos = [
        {"episode" : {'history' : self.episode_lengths, 'l' : self.last_episode_length}},
        {"reward" : self.last_reward },
        {"states" : self.states[0] }
                  ]

        return self.observations, rewards, next_dones, infos

    def observations_from_states(self, states):
        return np.array([ self.single_observation(state) for state in states ])

    def rewards_from_states(self,states):
        return np.array([ self.single_reward(state) for state in states ])