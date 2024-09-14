'''
Codes mainly from a robocup2023 repository owned by team TIDYBOY
'''


import numpy as np
from open3d import geometry, visualization, camera
import open3d as o3d
# import pyrender
import cv2

class Depth2PC:
    def __init__(self, camera_intrinsic):
        '''
        camera intrinsic matrix k: 3x3 numpy ndarray
        '''
        self.depth = None
        self.stamp = None
        self.rgb = None
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        self.camera_intrinsic = [fx, fy, cx, cy]
        self.camera_info = camera.PinholeCameraIntrinsic()


    def get_pc(self, rgb, depth, camera_extrinsic=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])):
        '''
        rgb image: numpy ndarray of shape (H, W, 3) the value should be 0~255, np.uint8
        depth image: numpy ndarray of shape (H, W)  
        camera_extrinsic: numpy ndarray of shape (4, 4)
            - extrinsic parameter that represents the direction and rotation of your camera based on the global coordinates
        
        
        output:
            pointcloud: open3d.geometry.PointCloud
        '''
        if self.camera_intrinsic is not None and depth is not None:
            if rgb is None:
                rgb = np.zeros((depth.shape[0], depth.shape[1], 3)).astype(np.uint8)
            H, W = depth.shape
            
            if not rgb.shape[:-1] == depth.shape:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            assert rgb.shape[:-1] == depth.shape, "IMAGE SHAPE OF RGB and DEPTH SHOULD BE SAME"
            

            # Handle NaN and Inf values in depth
            depth = np.where(np.isnan(depth), 0, depth)  # Replace NaNs with 0
            depth = np.where(np.isinf(depth), 0, depth)  # Replace Infs with 0
            print("rgb", rgb)

            depth = geometry.Image(depth.astype(np.uint16))
            rgb = geometry.Image(rgb.astype(np.uint8))
            print("camera intrinsic", self.camera_intrinsic)    
            self.camera_info.set_intrinsics(H, W, self.camera_intrinsic[0], self.camera_intrinsic[1],
                                            self.camera_intrinsic[2], self.camera_intrinsic[3])
            rgbd_image = geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False)
            pc = geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic=self.camera_info, extrinsic=camera_extrinsic, project_valid_depth_only=False)
            # pc = geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
            #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
            return pc
        
        else:
            return None
        
    def get_labeled_pc(self, depth, label, camera_extrinsic=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])):
        '''
        x, y, z + color + label을 다 커버하는 모듈은 제공안함... 그냥 따로따로 해서 합치는게 방법임
        현재는 color부분을 대체해서 label+ xyz 포인트 클라우드를 만들어준다.
        color = label_idx * 3 으로 다 인코딩 해줌
        depth: np.ndarray shape (H, W)
        label: np.ndarray shape (H, W) or (H, W, 1)
        output:
            pointcloud: open3d.geometry.PointCloud
        '''
        if self.camera_intrinsic is not None and depth is not None and label is not None:
            if len(depth.shape) == 3:
                assert depth.shape[-1] == 1, depth.shape
                depth = np.squeeze(depth, axis=-1)
            if len(label.shape) == 2:
                label = np.expand_dims(label, axis=-1)
            assert label.shape[-1] == 1 and len(label.shape) == 3, label.shape 
            assert label.max() <= 255, "LABEL OVER 8bit is not supported."
            label = np.tile(label, (1, 1, 3)).astype(np.uint8)
        
            pc = self.get_pc(label, depth, camera_extrinsic)
            print("PC", pc)
            return pc

        else:
            return None
        
        
        
    
    @staticmethod
    def visualize_points(pc):
        '''
        pc: open3d geometry pointcloud
        '''
        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors)
        cloud = pyrender.Mesh.from_points(points, colors=colors)
        scene = pyrender.Scene()
        scene.add(cloud)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
        # o3d.visualization.draw_geometries([pc])
    
    @staticmethod
    def visualize_labeled_points(pc):
        '''
        pc: labeled points by get_labeled_pc
        '''
        np.random.seed(0)
        points = np.asarray(pc.points)
        colors = np.asarray(pc.colors) 
        colors_decode = colors * 255
        uc = np.unique(colors_decode)
        for u in uc:
            colors[colors_decode[:,0] == u] = np.random.rand(3)
        
        pcd = o3d.geometry.PointCloud()
        # NumPy 배열을 PointCloud에 추가
        pcd.points = o3d.utility.Vector3dVector(points)  # xyz 좌표 추가
        pcd.colors = o3d.utility.Vector3dVector(colors)  # rgb 색상 추가

        # 시각화
        o3d.visualization.draw_plotly([pcd])

        # cloud = pyrender.Mesh.from_points(points, colors=colors)
        # scene = pyrender.Scene()


    

if __name__ == '__main__':
    '''
    images are from OCID Dataset (https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/)
    '''
    from pathlib import Path
    # 570.34221865, 570.34225815, 319.50000023, 239.50000413
    K = np.asarray([570.34221865,0,0,0,570.34225815,0,319.50000023,239.50000413,1]).astype(np.float32).reshape(3, 3)
    depth2pc = Depth2PC(K)
    rgb = cv2.cvtColor(cv2.imread('pointcloud_utils/rgb_example.png'), cv2.COLOR_BGR2RGB)
    depth = cv2.imread('pointcloud_utils/depth_example.png', cv2.IMREAD_ANYDEPTH)
    
    print('here 1')
    
    pc = depth2pc.get_pc(rgb, depth)
    print('here 2')
    pc = pc.remove_non_finite_points()
    print('here 3')
    depth2pc.visualize_points(pc)
    print('here 4')