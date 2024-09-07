"""

"""
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from my_utils import make_env


if __name__ == '__main__':
    env_id = 'door_sac'
    n_cpu = 60
    sac_policy = 'MlpPolicy'
    tot_timesteps = 1000000

    vec_env = SubprocVecEnv([make_env(env_id, i, False) for i in range(n_cpu)])

    model = SAC(sac_policy, vec_env, train_freq=1, gradient_steps=2, verbose=1, tensorboard_log = f'./{env_id}_tensorboard/')
    model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar = True)
    model.save(env_id)

    del model