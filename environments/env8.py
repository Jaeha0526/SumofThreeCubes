import math
from collections import Counter
import numpy as np
from .env1 import SumofCubesEnv

class SumofCubesEnv8(SumofCubesEnv):
    def __init__(self, args):
        self.num_envs = args.num_envs
        self.max_k = args.max_k        # boundary for max k

        self.single_observation_space = self.single_observation(np.array([1,1,1]))
        self.single_action_space = self.single_action(0)
        self.num_actions = 8**3

        self.states = np.zeros((self.num_envs, 3), dtype = np.int32)
        self.observations = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype = np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)
        self.reward_sums = np.zeros(self.num_envs, dtype = np.float32)

        # for infos
        self.last_episode_length = 0   # length of recently ended episode
        self.max_sum = 0               # sum of recently ended episode
        self.last_reward = 0           # reward of recently ended episode
        self.last_reward_list = []      # reward list of recently ended episode
        self.last_reward_sum = 0        # sum of reward list of recently ended episode

        # initialize
        self.reset()

        # all the records
        self.k_records = {}

        # p list we want to train
        self.plist = args.plist


    def reset(self, dones=None):
        if dones is not None :
            for i in range(self.num_envs):
                if dones[i]:
                    # record episode of the most recent terminated one
                    self.last_episode_length = self.episode_lengths[i]
                    self.last_reward = self.single_reward(self.states[i])[0]
                    self.last_reward_list = self.single_reward(self.states[i])[1]
                    self.last_reward_sum = self.reward_sums[i]
                    # reset done env
                    self.states[i] = np.random.randint(1, 3, (3,))
                    self.observations[i] = self.single_observation(self.states[i])
                    self.episode_lengths[i] = 0
                    self.reward_sums[i] = 0
                else:
                    self.episode_lengths[i] += 1
                    self.reward_sums[i] += self.single_reward(self.states[i])[0]

        else :
            self.states = np.random.randint(1, 3, (self.num_envs, 3))
            self.observations = self.observations_from_states(self.states)
            self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        return self.observations


    def single_observation(self,state):
        x, y, z = state
        # return np.array([x, y, z, abs(x**3 + y**3 - z**3), x%5, y%5, z%5, x**2, y**2, z**2, x**3, y**3, z**3, x**3-z**3, y**3-z**3,
        #                  x/z, y/z, x/y, x-z, y-z, x-y, x**2 + x*z + z**2, y**2 + y*z + z**2])
        return np.array([x, y, z, abs(x**3 + y**3 - z**3), x%37, y%37, z%37, x%7, y%7, z%7, x%11, y%11, z%11,
                         x/z, y/z, x/y, x-z, y-z, x-y, x**2, y**2, z**2, x**3, y**3, z**3, x**3-z**3, y**3-z**3, x**2 + x*z + z**2, y**2 + y*z + z**2])

    def modp_reward(self, k, p):
        k = k % p

        return 2**(-abs(k-1)/p * 50)

# plist = [11,13,19 ...]
# reward = sum ( reward_11 + reward 13 + ... )

    def single_reward(self,state):
        x, y, z = state
        k = abs(x**3 + y**3 - z**3)
        plist = self.plist
        reward_list = [ self.modp_reward(k,p) for p in plist ]

        return sum(reward_list), np.array(reward_list)


    def single_action(self, action): # +1 ~ +8    k = abs( x^3 + y^3 - z^3 )
        return np.array([(action // 64) + 1, (action % 8) + 1, (action % 64)//8 + 1])



    def step(self, actions):
        # take actions
        xyz_actions = [ self.single_action(action) for action in actions]
        self.states = self.states + xyz_actions
        self.observations = self.observations_from_states(self.states)

        # test dones
        # next_dones = np.array([ 1 if
        #                        abs(state[0]**3 + state[1]**3 - state[2]**3) > self.max_k or self.episode_lengths[i] > 7
        #                         else 0 for i, state in enumerate(self.states)])
        next_dones = np.array([ 1 if self.episode_lengths[i] > 13
                                 else 0 for i, state in enumerate(self.states)])

        # record max sum
        if 0 in next_dones :
            self.max_sum = max([ abs(s[0])+abs(s[1])+abs(s[2]) for s,b in zip(self.states,next_dones) if b == 0])
        rewards = self.rewards_from_states(self.states)

        # write on record board
        for i in range(self.num_envs):
            k = abs(self.states[i][0]**3 + self.states[i][1]**3 - self.states[i][2]**3)
            if k < self.max_k :
                if k not in self.k_records :
                    self.k_records[k] = [{tuple(self.states[i])},1]
                else :
                    self.k_records[k][0].add(tuple(self.states[i]))
                    self.k_records[k][1] += 1

        # reset done envs
        self.reset(next_dones)

        # additional infos
        infos = [ {"k": [abs(state[0]**3 + state[1]**3 - state[2]**3) for state in self.states]},
        {"episode" : {'history' : self.episode_lengths, 'l' : self.last_episode_length}},
        {"max_sum" : self.max_sum},
        {"reward" : self.last_reward, "reward_list": self.last_reward_list, "reward_sum": self.last_reward_sum },
        {"states" : self.states },
                  ]

        return self.observations, rewards, next_dones, infos

    def observations_from_states(self, states):
        return np.array([ self.single_observation(state) for state in states ])

    def rewards_from_states(self,states):
        return np.array([ self.single_reward(state)[0] for state in states ])