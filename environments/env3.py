import math
import numpy as np
from .env1 import SumofCubesEnv
from collections import Counter

class SumofCubesEnv3(SumofCubesEnv):
    def __init__(self, args):
        self.num_envs = args.num_envs
        self.k = args.target_k
        self.reward = args.reward
        self.max_n = args.max_n
        self.min_n = args.min_n
        self.ep_length = args.ep_length

        self.single_observation_space = self.single_observation(np.array([1,1,1,1,1,1]))
        self.single_action_space = self.single_action(0)
        # self.num_actions = 3**6
        self.num_actions = 2 * 6

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
                    self.states[i] = np.random.randint(-3, -1, (6,))
                    self.observations[i] = self.single_observation(self.states[i])
                    self.episode_lengths[i] = 0
                else:
                    self.episode_lengths[i] += 1

        else :
            self.states = np.random.randint(-3, -1, (self.num_envs, 6))
            self.observations = self.observations_from_states(self.states)
            self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        return self.observations


    def single_observation(self,state):
        a0, a1, a2, a3, b0, b1 = state
        return np.array([ a0, a1, a2, a3, b0, b1 ])


    def single_reward(self, state):
        a0, a1, a2, a3, b0, b1 = state

        reward = 0

        for i in range(self.min_n,self.max_n+1):

            z = a0 + a1*i + a2*i**2 + a3*i**3
            d = b0 + b1*i

            if d == 0 or z == 0 :
                continue

            z3mk = - z**3 + self.k
            disc = 12*z3mk/d - 3*d**2

            if z3mk % d == 0 :
                reward += self.reward[0]

                if disc > 0 :
                    reward += self.reward[1]

                    sqrt_n = round(math.sqrt(disc)/3, 20) # Round to 5 decimal places
                    if abs(sqrt_n - round(sqrt_n)) < 1e-20:
                        reward += self.reward[2]

                        xmy = int(round(sqrt_n))
                        if (xmy-d)%2 == 0 :
                            pair = (int((d+xmy)/2),int((d-xmy)/2),z)
                            reward += self.reward[3]

                            if pair[0] + pair[1] + pair[2] == 1 :
                                reward -= (self.reward[0] + self.reward[1] + self.reward[2] + self.reward[3])
                            else :
                                self.solutions.add(pair)
                                self.solutionstates.add((a0,a1,a2,a3,b0,b1,i))

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

        # independently adjust numbers   ->  2 * 6
        action_space = np.array([0,0,0,0,0,0])
        action_space[action % 6] = 2*(action//6) - 1
        return action_space



    def step(self, actions):
        # take actions and get informations
        zd_actions = [ self.single_action(action) for action in actions]
        self.states = self.states + zd_actions

        # restrict a1, a2, and a3 are all zero or b1 is zero
        for i in range(len(self.states)):
            if self.states[i][5] == 0 :
                self.states[i][5] = 1

            if self.states[i][1] == 0 and self.states[i][2] == 0 and self.states[i][3] == 0:
                self.states[i][1] = 1
                self.states[i][2] = 1
                self.states[i][3] = 1


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