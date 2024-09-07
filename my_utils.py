import robosuite as suite
from robosuite.wrappers import GymWrapper

from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, render: bool, seed: int = 0):
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
                has_renderer=render,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth,
                use_latch=False,
            )
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init