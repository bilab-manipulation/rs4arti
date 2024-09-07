import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import robosuite as suite
from robosuite.wrappers import GymWrapper

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = GymWrapper(
            suite.make(
                "MyDoor",
                robots="UR5e",  # use Sawyer robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=False,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth,
                use_latch=False,
            )
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = 'door_sac'
    n_cpu = 1
    sac_policy = 'MlpPolicy'
    tot_timesteps = 1000000

    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(n_cpu)])
    model = SAC.load(env_id, env=vec_env)

    obs = vec_env.reset()
    for t in range(1000):
    #while True:
        action, _states = model.predict(obs)
        #print(action)
        obs, rewards, dones, info = vec_env.step(action)
        #vec_env.render()