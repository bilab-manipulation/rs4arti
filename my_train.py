"""

"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

from my_utils import make_env

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hinge', type=str)
    parser.add_argument('--latch', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    print(f'hinge: {args.hinge} {type(args.hinge)}')
    print(f'latch: {args.latch} {type(args.latch)}')
    print(f'seed: {args.seed} {type(args.seed)}')

    hinge = args.hinge
    if hinge == 'yes':
        hinge = True
    elif hinge == 'no':
        hinge = False
    else:
        print('undefined hinge')
    latch = args.latch
    if latch == 'yes':
        latch = True
    elif latch == 'no':
        latch = False
    else:
        print('undefined latch')
    seed = args.seed# 4
    env_id = f'nodist_door_panda_osc_dense_hinge{hinge}_latch{latch}_{seed}'
    n_cpu = 20 # dale3: 72, biomen: 24
    sac_policy = 'MlpPolicy' # 입력이 구조 없는 vector라서 cnn보다 mlp가 맞음
    tot_timesteps = 1000000
    
    from stable_baselines3.common.logger import configure
    log_dir = f'./{env_id}_tensorboard/'
    os.makedirs(log_dir, exist_ok = True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    vec_env = SubprocVecEnv([make_env(env_id, i, False, hinge, latch, seed) for i in range(n_cpu)])

    model = SAC(sac_policy, 
                vec_env, 
                learning_rate = 0.00075,
                buffer_size = 1000000,
                learning_starts = 3300,
                batch_size = 128,
                tau = 0.005,
                gamma = 0.99,
                train_freq=1, 
                gradient_steps=1, 
                action_noise = None,
                replay_buffer_class = None,
                #replay_buffer_kwargs
                #optimize_memory_usage
                ent_coef= 'auto',
                target_update_interval = 5,#1,
                target_entropy = 'auto',
                use_sde = False,
                sde_sample_freq = -1,
                use_sde_at_warmup = False,
                stats_window_size = 100,
                seed = seed,
                #tensorboard_log = log_dir,
                verbose=1, 
                )
    
    #eval_env = SubprocVecEnv([make_env(env_id, 100, False)])
    #eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                         log_path=log_dir, eval_freq=5000, deterministic=True)


    #model.learn(total_timesteps=tot_timesteps, log_interval=4, callback = eval_callback, progress_bar = True, )
    model.set_logger(new_logger)
    model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar = True)
    model.save(env_id)

    del model