import math
from collections import Counter
import numpy as np
from .env1 import SumofCubesEnv

class SumofCubesEnv6(SumofCubesEnv):
    def __init__(self, args):
        self.num_envs = args.num_envs
        self.max_k = args.max_k
        self.rewards = args.reward
        self.depreciation = args.depreciation

        self.single_observation_space = self.single_observation(np.array([1,1,1,1]))
        self.single_action_space = self.single_action(0)
        self.num_actions = 4**4

        self.states = np.zeros((self.num_envs, 4), dtype = np.int32)
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
        self.solutions = {}


    def reset(self, dones=None):
        if dones is not None :
            for i in range(self.num_envs):
                if dones[i]:
                    # record episode of the most recent terminated one
                    self.last_episode_length = self.episode_lengths[i]
                    self.last_reward = self.single_reward(self.states[i])
                    # reset done env
                    self.states[i] = np.random.randint(1, 4, (4,))
                    self.observations[i] = self.single_observation(self.states[i])
                    self.episode_lengths[i] = 0
                else:
                    self.episode_lengths[i] += 1

        else :
            self.states = np.random.randint(1, 4, (self.num_envs, 4))
            self.observations = self.observations_from_states(self.states)
            self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        return self.observations


    def single_observation(self,state):
        x, y, z, w = state
        normalization = 10000
        return np.array([x, y, z, w, np.sign(x), np.sign(y), np.sign(z), np.sign(w), abs(x**3 + y**3 + z**3 + w**3)/normalization, x**2/normalization, y**2/normalization, z**2/normalization, w**2/normalization
                         , x**3/normalization, y**3/normalization, z**3/normalization, w**3/normalization,
                         x**3-z**3, y**3-z**3, x**3-w**3, y**3-w**3,
                         x/(1+abs(z)), y/(1+abs(z)), x/(1+abs(y)), z/(1+abs(w)), x/(1+abs(w)), y/(1+abs(w)), x-z, y-z, x-y, z-w,
                         x**2 + x*z + z**2, y**2 + y*z + z**2, z**2 + z*w + w**2])


    def single_reward(self,state):
        x, y, z, w = state
        k = abs(x**3 + y**3 + z**3 + w**3)
        sorted_state = sorted([x,y,z,w])
        fs = frozenset([sorted_state.append(k)])

        # end reward
        if k > self.max_k :
            return 0

        # if it was visited
        if fs in self.solutions:
            self.solutions[fs] += 1
            return self.rewards[0]*(1 - k/self.max_k) * 2**(-self.solutions[fs]/self.depreciation)

        # check for having same number with different sign or 0
        a = set([x,y,z,w])
        b = set([-x,-y,-z,-w])
        if len(a.intersection(b)) > 0:
            return self.rewards[1]

        # new solution
        self.solutions[fs] = 1
        reward = self.rewards[2]*(1 - k/self.max_k) + self.rewards[3]*(x**2 + y**2 + z**2)

        return reward


    def single_action(self, action): # -1 ~ +2
        return np.array([(action // 64) - 1 , (action % 64)//16 - 1, (action % 16)//4 - 1, (action % 4) - 1])

    def single_signs(self, state):
        x, y, z, w = state
        return np.array([np.sign(x), np.sign(y), np.sign(z), np.sign(w)])


    def step(self, actions):
        # take actions and get informations
        xyzw_signs = np.array([ self.single_signs(state) for state in self.states ])
        xyzw_actions = np.array([ self.single_action(action) for action in actions ])
        self.states = self.states + xyzw_actions * xyzw_signs
        self.observations = self.observations_from_states(self.states)
        next_dones = np.array([ 1 if abs(state[0]**3 + state[1]**3 + state[2]**3 + state[3]**3) > self.max_k else 0 for state in self.states ])

        if 0 in next_dones :
            self.max_sum = max([ abs(s[0])+abs(s[1])+abs(s[2])+abs(s[3]) for s,b in zip(self.states,next_dones) if b == 0])
        rewards = self.rewards_from_states(self.states)

        # write on record board
        for i in range(self.num_envs):
            k = abs(self.states[i][0]**3 + self.states[i][1]**3 + self.states[i][2]**3 + self.states[i][3]**3)
            if k < self.max_k :
                if k not in self.k_records :
                    self.k_records[k] = [{tuple(self.states[i])},1]
                else :
                    self.k_records[k][0].add(tuple(self.states[i]))
                    self.k_records[k][1] += 1

        # reset done envs
        self.reset(next_dones)

        # additional infos
        infos = [ {"k": [abs(state[0]**3 + state[1]**3 + state[2]**3 + state[3]**3) for state in self.states]},
        {"episode" : {'history' : self.episode_lengths, 'l' : self.last_episode_length}},
        {"max_sum" : self.max_sum},
        {"reward" : self.last_reward },
                  ]

        return self.observations, rewards, next_dones, infos

    def observations_from_states(self, states):
        return np.array([ self.single_observation(state) for state in states ])

    def rewards_from_states(self,states):
        return np.array([ self.single_reward(state) for state in states ])