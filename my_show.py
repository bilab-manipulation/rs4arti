from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from my_utils import make_env

if __name__ == '__main__':
    env_id = 'door'
    
    vec_env = DummyVecEnv([make_env(env_id, 0, True)])
    model = SAC.load(env_id, env=vec_env)

    print('here 1')

    obs = vec_env.reset()
    for t in range(500):
        action, _states = model.predict(obs)

        print('here 2')

        #from robosuite.utils.camera_utils import grab_next_frame, transform_from_pixels_to_world

        #print(action)
        obs, rewards, dones, info = vec_env.step(action)
        print(t, len(obs), obs)
        import open3d as o3d
        print('here 3')
        color_raw = o3d.geometry.Image(vec_env.envs[0].env.env._observables['agentview_image'].obs)
        depth_raw = o3d.geometry.Image(vec_env.envs[0].env.env._observables['agentview_depth'].obs)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        print(rgbd_image)
        print('here 4')

        import matplotlib.pyplot as plt
        print('here 5')
        plt.subplot(1, 2, 1)
        plt.title('Redwood grayscale image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Redwood depth image')
        plt.imshow(rgbd_image.depth)
        print('here 6')
        plt.show()
        print('here 7')

        o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        print('here 8')

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        
        print('here 9')
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print('here 10')
        print(type(pcd), pcd)
        print('here 11')
        #o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_plotly([pcd])
        print('here 12')
        #pcd = o3d.io.read_point_cloud("file.pcd", format="pcd")
        #vis = o3d.visualization.Visualizer()
        #vis.create_window()
        #vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        #vis.get_render_option().point_size = 3.0
        #vis.add_geometry(pcd)
        #o3d.capture_screen_image("vis_test.jpg", do_render=False)
        print('here 13')
        vec_env.envs[0].render()