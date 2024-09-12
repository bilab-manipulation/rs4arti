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

        #from robosuite.utils.camera_utils import grab_next_frame, transform_from_pixels_to_world

        #print(action)
        obs, rewards, dones, info = vec_env.step(action)
        print(t, len(obs), obs)
        import open3d as o3d
        color_raw = o3d.geometry.Image(vec_env.envs[0].env.env._observables['agentview_image'].obs)
        depth_raw = o3d.geometry.Image(vec_env.envs[0].env.env._observables['agentview_depth'].obs)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        print(rgbd_image)

        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.title('Redwood grayscale image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Redwood depth image')
        plt.imshow(rgbd_image.depth)
        #plt.show()

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        
        print('here1')
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print('here2')
        print(type(pcd), pcd)
        print('here3')
        #o3d.visualization.draw_geometries([pcd], zoom=0.5)
        o3d.visualization.draw_geometries([pcd])

        
        import pdb; pdb.set_trace()

        vec_env.envs[0].render()