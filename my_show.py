from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import open3d as o3d
import matplotlib.pyplot as plt

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from my_utils import make_env

if __name__ == '__main__':
    env_id = 'door'
    horizon = 500
    
    print('making env')
    vec_env = DummyVecEnv([make_env(env_id, 0, True)])

    print('load model')
    model = SAC.load(env_id, env=vec_env)

    print('reset env')
    obs = vec_env.reset()
    
    print('get cam intrinsic')
    camera_intrinsic_matrix = get_camera_intrinsic_matrix(vec_env.envs[0].sim, 'agentview', 480, 640)
    o3d_pcam_intr = o3d.camera.PinholeCameraIntrinsic(256, 256, camera_intrinsic_matrix)
    print(camera_intrinsic_matrix)
    print(o3d_pcam_intr)

    for t in range(horizon):
        print('predict by model')
        action, _states = model.predict(obs)

        print('step forward')
        obs, rewards, dones, info = vec_env.step(action)
        print(t, len(obs), obs)
        
        print('making rgbd image')
        color_raw = o3d.geometry.Image(vec_env.envs[0].env.env._observables['agentview_image'].obs)
        depth_raw = o3d.geometry.Image(vec_env.envs[0].env.env._observables['agentview_depth'].obs)
        print(vec_env.envs[0].env.env._observables['agentview_depth'].obs.dtype)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
        print(rgbd_image)
        
        print('making rgbd image plot')
        plt.subplot(1, 2, 1)
        plt.title('grayscale image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('depth image')
        plt.imshow(rgbd_image.depth)
        print('plot rgbd image')
        plt.show()
        plt.close()

        print('here')
        cam_intr = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        print(cam_intr)

        print('making point cloud from rgbd image')
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        print('plot pc by using plotly')
        o3d.visualization.draw_plotly([pcd])
        
        print('rendering env')
        vec_env.envs[0].render()

        import pdb;pdb.set_trace()