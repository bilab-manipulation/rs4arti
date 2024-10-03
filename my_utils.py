import robosuite as suite
from robosuite.wrappers import MyGymWrapper

from robosuite.controllers import load_controller_config

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

def make_env(env_id: str, rank: int, render: bool, hinge: bool, latch: bool, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():

        rsenv = suite.make(
                "MyDoor",
                robots="Panda", # robot type
                controller_configs=load_controller_config(default_controller="OSC_POSE"), # OSC_POSE, JOINT_POSE, etc.
                use_latch=latch, # for easy
                use_camera_obs=False,  # use pixel observations
                reward_shaping=False,  # use dense rewards
                has_renderer=render,  # make sure we can render to the screen
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                control_freq=20,  # control should happen fast enough so that simulation looks smooth,
                horizon=500,
                camera_depths=False,
                camera_names='agentview',

                hard_reset = True,
                ignore_done = False,
            )
        
        # print('enabled observables: ', len(rsenv.enabled_observables), rsenv.enabled_observables)
        # print('active observables: ', len(rsenv.active_observables), rsenv.active_observables)

        full_observable_list = [        # enabled, active
            'robot0_joint_pos',         # True, False
            'robot0_joint_pos_cos',     # True, True
            'robot0_joint_pos_sin',     # True, True
            'robot0_joint_vel',         # True, True
            'robot0_eef_pos',           # True, True
            'robot0_eef_quat',          # True, True
            'robot0_eef_vel_lin',       # True, False
            'robot0_eef_vel_ang',       # True, False
            'robot0_gripper_qpos',      # True, True
            'robot0_gripper_qvel',      # True, True
            # 'agentview_image',        # 
            # 'agentview_depth',        # 
            'door_pos',                 # True, True
            'handle_pos',               # True, True
            'door_to_eef_pos',          # True, True
            'handle_to_eef_pos',        # True, True
            'hinge_qpos',               # True, True
        ]
        if latch == True:
            full_observable_list.append('handle_qpos')
        for observable in full_observable_list:
            rsenv.modify_observable(observable, 'enabled', True)
            rsenv.modify_observable(observable, 'active', True)
        
        useless_observable_list = [
            # 'robot0_joint_pos',
            # 'robot0_joint_pos_cos',
            # 'robot0_joint_pos_sin',
            # 'robot0_joint_vel',
            # 'robot0_eef_pos',
            # 'robot0_eef_quat',
            # 'robot0_eef_vel_lin',
            # 'robot0_eef_vel_ang',
            # 'robot0_gripper_qpos',
            # 'robot0_gripper_qvel',
            # 'agentview_image',
            # 'agentview_depth',
            # 'door_pos',
            # 'handle_pos',
            # 'door_to_eef_pos',
            # 'handle_to_eef_pos',
            # 'hinge_qpos',
        ]
        for observable in useless_observable_list:
            rsenv.modify_observable(observable, 'enabled', False)
            rsenv.modify_observable(observable, 'active', False)

        # print('Robosuite environment maked:',type(rsenv) , rsenv, dir(rsenv))
        # print(len(rsenv._observables.keys()))
        # print(rsenv._observables.keys())
        
        env = MyGymWrapper(
            rsenv, hinge, latch
        )

        env.reset(seed=seed + rank)

        return Monitor(env)
        #return env
    set_random_seed(seed)
    return _init