import numpy as np

class SumofCubesEnv():
    def __init__(self, args):
        self.num_envs = args.num_envs
        self.max_k = args.max_k

        self.single_observation_space = self.single_observation(np.array([1,1,1]))
        self.single_action_space = self.single_action(0)
        self.num_actions = 8**3

        self.states = np.zeros((self.num_envs, 3), dtype = np.int32)
        self.observations = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype = np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        # for infos
        self.last_episode_length = 0
        self.max_sum = 0

        # initialize
        self.reset()

        # all the records
        self.k_records = {}


    def reset(self, dones=None):
        if dones is not None :
            for i in range(self.num_envs):
                if dones[i]:
                    # record episode of the most recent terminated one
                    self.last_episode_length = self.episode_lengths[i]
                    # reset done env
                    self.states[i] = np.random.randint(1, 3, (3,))
                    self.observations[i] = self.single_observation(self.states[i])
                    self.episode_lengths[i] = 0
                else:
                    self.episode_lengths[i] += 1

        else :
            self.states = np.random.randint(1, 3, (self.num_envs, 3))
            self.observations = self.observations_from_states(self.states)
            self.episode_lengths = np.zeros(self.num_envs, dtype = np.int32)

        return self.observations


    def single_observation(self,state):
        x, y, z = state
        return np.array([x, y, z, abs(x**3 + y**3 - z**3), x**2, y**2, z**2, x**3, y**3, z**3, x**3-z**3, y**3-z**3,
                         x/z, y/z, x/y, x-z, y-z, x-y, x**2 + x*z + z**2, y**2 + y*z + z**2])


    def single_reward(self,state):
        x, y, z = state
        k = abs(x**3 + y**3 - z**3)
        if k > self.max_k :
            reward = -1
        elif k == 2 :
            reward = 10
        else :
            reward = 1 - k/self.max_k

        return reward


    def single_action(self, action): # +1 ~ +8    k = abs( x^3 + y^3 - z^3 )
        return np.array([(action // 64) + 1, (action % 8) + 1, (action % 64)//8 + 1])



    def step(self, actions):
        # take actions and get informations
        xyz_actions = [ self.single_action(action) for action in actions]
        self.states = self.states + xyz_actions
        self.observations = self.observations_from_states(self.states)
        next_dones = np.array([ 1 if abs(state[0]**3 + state[1]**3 - state[2]**3) > self.max_k else 0 for state in self.states ])
        if 0 in next_dones :
            self.max_sum = max([ abs(s[0])+abs(s[1])+abs(s[2]) for s,b in zip(self.states,next_dones) if b == 0])
        rewards = self.rewards_from_states(self.states)

        # write on record board
        for i in range(self.num_envs):
            k = abs(self.states[i][0]**3 + self.states[i][1]**3 - self.states[i][2]**3)
            if k < 100 :
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
        {"max_sum" : self.max_sum}
                  ]

        return self.observations, rewards, next_dones, infos

    def observations_from_states(self, states):
        return np.array([ self.single_observation(state) for state in states ])

    def rewards_from_states(self,states):
        return np.array([ self.single_reward(state) for state in states ])