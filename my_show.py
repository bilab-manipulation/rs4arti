from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from my_utils import make_env

if __name__ == '__main__':
    env_id = 'door'
    
    vec_env = DummyVecEnv([make_env(env_id, 0, True)])
    model = SAC.load(env_id, env=vec_env)

    obs = vec_env.reset()
    for t in range(500):
        action, _states = model.predict(obs)
        #print(action)
        obs, rewards, dones, info = vec_env.step(action)
        print(t, obs)
        vec_env.envs[0].render()