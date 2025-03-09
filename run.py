from PPO import (
    PPOArgs,
    ReplayMinibatch,
    ReplayMemory,
    PPOAgent,
    PPOScheduler,
    PPOTrainer,
    plot_rewards
)
from environments.env1 import SumofCubesEnv
from environments.env2 import SumofCubesEnv2
from environments.env3 import SumofCubesEnv3
from environments.env4 import SumofCubesEnv4
from environments.env5 import SumofCubesEnv5
from environments.env6 import SumofCubesEnv6
from environments.env7 import SumofCubesEnv7
from environments.env8 import SumofCubesEnv8
from environments.env9 import SumofCubesEnv9



if __name__ == '__main__':
    
    # # environment test code
    # args = PPOArgs()
    # env = SumofCubesEnv3(args)
    # print(env.reset())
    # print(env.step([0,1,2,3]))
    # print(env.step([10,11,12,13]))
    # print(env.step([20,21,22,23]))
    # print(env.step([30,31,32,33]))
    
    # # PPOAgent test code
    # args = PPOArgs()
    # agent = PPOAgent(args, SumofCubesEnv8(args))
    # print(agent.play_step())
    
    # # Trainer test code
    # args = PPOArgs(
    #     num_steps = 6, 
    #     total_timesteps = 24, 
    #     target_k = 1, 
    #     reward = [1, 0.01, 1, 0.1, 0.1, 0.1], 
    #     ep_length=3
    # )
    # trainer = PPOTrainer(args, environment=SumofCubesEnv3, test_mode = True)
    # trainer.train()
    
    #env8 run
    args = PPOArgs(   total_timesteps = 100000,
        num_envs = 8,
        num_steps = 128,
        target_k = 1,
        max_k = 1000000,
        gamma = 0.99,
        ent_coef=0.01,
        plist = [37],
        reward = [1000, 100, 1, 0.1, 0.1], 
        depreciation=10,
    )

    trainer = PPOTrainer(args, SumofCubesEnv9)
    trainer.train()
    print(trainer.envs.good_solutions)
    # print(trainer.envs.solutions)
    plot_rewards([trainer.last_episode_len_records[:]])
    
