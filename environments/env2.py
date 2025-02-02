import math
import numpy as np
from .env1 import SumofCubesEnv
from collections import Counter

class SumofCubesEnv2(SumofCubesEnv):
    def __init__(self, args):
        self.num_envs = args.num_envs
        self.k = args.target_k
        self.reward = args.reward
        self.max_z = args.max_z
        self.mode = args.mode
        self.action_bias = args.action_bias

        self.single_observation_space = self.single_observation(np.array([1,1]))
        self.single_action_space = self.single_action(0)
        self.num_actions = 8*5

        self.states = np.zeros((self.num_envs, 2), dtype = np.int32)
        self.observations = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype = np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        # for infos
        self.last_episode_length = 0
        self.max_sum = 0
        self.last_reward = 0

        # initialize
        self.reset()

        # all the records
        self.k_records = {}
        self.solutions = Counter()


    def reset(self, dones=None):
        if dones is not None :
            for i in range(self.num_envs):
                if dones[i]:
                    # record episode of the most recent terminated one
                    self.last_episode_length = self.episode_lengths[i]
                    self.last_reward = self.single_reward(self.states[i])[0]
                    # reset done env
                    self.states[i] = np.random.randint(1, 3, (2,))
                    self.observations[i] = self.single_observation(self.states[i])
                    self.episode_lengths[i] = 0
                else:
                    self.episode_lengths[i] += 1

        else :
            self.states = np.random.randint(1, 3, (self.num_envs, 2))
            self.observations = self.observations_from_states(self.states)
            self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        return self.observations


    def single_observation(self,state):
        z, d = state
        return np.array([z, d, ((z**3 - self.k)%d)/d, 4*(z**3-self.k)/(3*d) - d**2/3 ])


    def single_reward(self, state):
        z, d = state
        z3mk = (z**3 - self.k)*self.mode
        disc = 12*z3mk/d - 3*d**2
        reward = 0
        pair = None
        if z3mk % d == 0 :
            reward += self.reward[0]

            if disc > 0 :
                reward += self.reward[1]

                sqrt_n = round(math.sqrt(disc)/3, 5) # Round to 5 decimal places
                if abs(sqrt_n - round(sqrt_n)) < 1e-5:
                    reward += self.reward[2]

                    xmy = int(round(sqrt_n))
                    if (xmy-d)%2 == 0 :
                        pair = (-int((d+xmy)/2),-int((d-xmy)/2),z)
                        if pair not in self.solutions and z-d != 1:
                            reward += self.reward[3]
                        self.solutions[pair] += 1

        return reward, pair



    def single_action(self, action): # z : -1 - +6   d : -2 - 2
        return np.array([(action // 8) + self.action_bias[0], (action % 5) + self.action_bias[1]])



    def step(self, actions):
        # take actions and get informations
        zd_actions = [ self.single_action(action) for action in actions]
        self.states = self.states + zd_actions

        # restrict z, d larger than 0
        for i in range(len(self.states)):
            if self.states[i][1] == 0 :
                self.states[i][1] = 1
            else :
                self.states[i][1] = abs(self.states[i][1])

            if self.states[i][0] == 0 :
                self.states[i][0] = 1
            else :
                self.states[i][0] = abs(self.states[i][0])


        # get observations from states
        self.observations = self.observations_from_states(self.states)

        # get dones from states
        next_dones = np.array([ 0 if 4*(state[0]**3-self.k) - state[1]**3 > -10 and state[0] < self.max_z else 1 for state in self.states ])

        # get rewards from states
        rps = self.rewards_from_states(self.states)
        rewards = np.array([ reward for reward, pair in rps ])

        # reset done envs
        self.reset(next_dones)

        # additional infos
        infos = [
        {"episode" : {'history' : self.episode_lengths, 'l' : self.last_episode_length}},
        {"reward" : self.last_reward}
                  ]

        return self.observations, rewards, next_dones, infos

    def observations_from_states(self, states):
        return np.array([ self.single_observation(state) for state in states ])

    def rewards_from_states(self,states):
        return [ self.single_reward(state) for state in states ]