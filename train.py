"""

"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

from my_utils import make_env


if __name__ == '__main__':
    env_id = 'door'
    n_cpu = 70 # dale3 기준 총 72개
    sac_policy = 'MlpPolicy' # 입력이 구조 없는 vector라서 cnn보다 mlp가 맞음
    tot_timesteps = 1000000
    
    from stable_baselines3.common.logger import configure
    log_dir = f'./{env_id}_tensorboard/'
    os.makedirs(log_dir, exist_ok = True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    vec_env = SubprocVecEnv([make_env(env_id, i, False) for i in range(n_cpu)])

    model = SAC(sac_policy, 
                vec_env, 
                batch_size = 128,
                learning_rate = 0.00075, # stable-baseline에서는 policy랑 qfunc에 동일한 learning rate 써야 해서 중간값 씀
                buffer_size = 1000000,
                learning_starts = 3300,
                target_update_interval = 5,

                train_freq=1, 
                gradient_steps=-1, 
                verbose=1, 
                #tensorboard_log = log_dir
                )
    
    #eval_env = SubprocVecEnv([make_env(env_id, 100, False)])
    #eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                         log_path=log_dir, eval_freq=5000, deterministic=True)


    #model.learn(total_timesteps=tot_timesteps, log_interval=4, callback = eval_callback, progress_bar = True, )
    model.set_logger(new_logger)
    model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar = True)
    model.save(env_id)

    del model