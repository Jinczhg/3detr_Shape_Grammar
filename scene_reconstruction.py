import os
from functools import reduce

import numpy as np
import open3d as o3d
from utils.pc_util import rotx, roty, rotz
from pc_alignment import icp_registration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
demo_dir = os.path.join(BASE_DIR, 'demo_files')
viz_data = []

viz_cls = [0, 1, 2, 3, 4]
viz_color = [[0.6, 0.2, 1], [0.9, 0.5, 0.2], [1, 0.3, 0.9], [0.7, 0.9, 0.2], [1, 0.8, 0]]
object_mesh_path = ["/home/jzhang72/Downloads/PSML3SampleProject(1)/java/test_table.ply",
                    "/home/jzhang72/Downloads/PSML3SampleProject(1)/java/test_chair.ply",
                    "/home/jzhang72/Downloads/PSML3SampleProject(1)/java/test_bookshelf.ply",
                    "/home/jzhang72/Downloads/PSML3SampleProject(1)/java/test_floor.ply",
                    "/home/jzhang72/Downloads/PSML3SampleProject(1)/java/test_backwall.ply"]
obj_pts_lst = []
obj_pts_ext_lst = []
obj_pcd_lst = []

# original point cloud
scene_point_cloud = []
pc_path = os.path.join(demo_dir, '4/pointcloud.txt')
bb3d_gt_path = os.path.join(demo_dir, '4/bb3d.txt')
with open(pc_path, 'r') as f:
    for line in f.readlines():
        point = line.split(' ')
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        point_as_array = [x, y, z]
        scene_point_cloud.append(point_as_array)
scene_point_cloud = np.asarray(scene_point_cloud)
pcd_scene = o3d.geometry.PointCloud()
pcd_scene.points = o3d.utility.Vector3dVector(scene_point_cloud)
# pcd_scene = pcd_scene.uniform_down_sample(5)
# show the original point cloud of the scene
o3d.visualization.draw_geometries([pcd_scene],
                                  zoom=2,
                                  front=[0.5439, 0.2333, -0.8060],
                                  lookat=[2.4615, -2.1331, 1.338],
                                  up=[-0.1781, 0.9708, 0.1608])

# ground truth 3D bbox
bbox = []
with open(bb3d_gt_path, 'r') as f:
    for line in f.readlines():
        point = line.split(' ')
        cls = float(point[0])
        cx = float(point[1])
        cy = float(point[2])
        cz = float(point[3])
        l = float(point[4])
        w = float(point[5])
        h = float(point[6])
        roll = float(point[7])
        pitch = float(point[8])
        yaw = float(point[9])
        point_as_array = [cls, cx, cy, cz, l, w, h, roll, pitch, yaw]
        bbox.append(point_as_array)

        # visualize the bbox
        # if cls in viz_cls:
        center = np.array([cx, cy, cz])
        size = np.array([2 * l, 2 * w, 2 * h])
        xform = np.matmul(rotz(yaw), roty(pitch))
        xform = np.matmul(xform, rotx(roll))
        gt_3d = o3d.geometry.OrientedBoundingBox(center, xform, size)
        gt_3d.color = [0, 1, 0]
        viz_data.append(gt_3d)
bbox = np.asarray(bbox)
viz_data.append(pcd_scene)
o3d.visualization.draw_geometries(viz_data,
                                  zoom=2,
                                  front=[0.5439, 0.2333, -0.8060],
                                  lookat=[2.4615, -2.1331, 1.338],
                                  up=[-0.1781, 0.9708, 0.1608])

# reconstructed model
for obj_pth in object_mesh_path:
    object_mesh = o3d.io.read_point_cloud(obj_pth)
    object_points = object_mesh.points
    object_points = np.asarray(object_points)
    # obj_points = np.vstack((obj_points[:, 0], obj_points[:, 1], obj_points[:, 2])).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points)
    pcd = pcd.uniform_down_sample(50)
    pcd.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=2,
    #                                   front=[0.5439, 0.2333, -0.8060],
    #                                   lookat=[2.4615, -2.1331, 1.338],
    #                                   up=[-0.1781, 0.9708, 0.1608])
    # viz_data.append(pcd)
    obj_pts_lst.append(object_points)
    object_points_ext = np.vstack((object_points[:, 0], object_points[:, 1], object_points[:, 2], np.ones(len(object_points))))
    obj_pts_ext_lst.append(object_points_ext)
    obj_pcd_lst.append(pcd)

# read the pose from the prediction
predPath = os.path.join(demo_dir, 'pred_results.txt')
# predicted object locations in the point cloud
loc = []
with open(predPath, 'r') as f:
    for line in f.readlines():
        pred = line.split(' ')
        class_p = int(pred[0])
        if class_p in viz_cls:
            target_object = obj_pts_ext_lst[class_p]  # retrieve the corresponding pcd model from the candidate models for substitution
            target_pcd = obj_pcd_lst[class_p]
            affine_transform = np.zeros([4, 4])  # transformation generated from the predicted roll, pitch, yaw and cx, cy, cz
            # Flip X-right, Y-forward, Z-up (SUNRGBD coordinate) back to X-right, Y-up, Z-forward (shape grammar coordinate)
            cx_p = float(pred[1])
            cy_p = float(pred[3])
            cz_p = float(pred[2])
            size_x_p = float(pred[4])
            size_y_p = float(pred[6])
            size_z_p = float(pred[5])
            roll_p = float(pred[7])
            yaw_p = -float(pred[8])  # -pitch (SUNRGBD coordinate) -> yaw (shape grammar)
            pitch_p = -float(pred[9])  # -yaw (SUNRGBD coordinate) -> pitch (shape grammar)
            xform_p = np.matmul(rotz(yaw_p), roty(pitch_p))
            xform_p = np.matmul(xform_p, rotx(roll_p))
            affine_transform[0:3, 0:3] = xform_p
            affine_transform[:, 3] = np.asarray([cx_p, cy_p, cz_p, 1]).transpose()

            ICP_align = True
            if ICP_align:
                # align the reconstructed object with the original scene point clouds using ICP
                threshold = 0.2  # max correspondence distance
                trans_init = affine_transform
                reg_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                max_iter = 1000
                # align with the entire scene
                affine_transform = icp_registration(target_pcd, pcd_scene, trans_init, threshold, reg_method, max_iter, False)

            obj_points_trans = np.matmul(affine_transform, target_object)
            obj_points_trans = np.delete(obj_points_trans, 3, axis=0)  # get rid of last row (ones)
            obj_points_trans = obj_points_trans.transpose()
            pcd_p = o3d.geometry.PointCloud()
            pcd_p.points = o3d.utility.Vector3dVector(obj_points_trans)
            pcd_p.paint_uniform_color(viz_color[class_p])
            pcd_p = pcd_p.uniform_down_sample(10)
            viz_data.append(pcd_p)

            # first transform the scene point cloud to align with the 3D bbox (so we have an axis-align coordinate to operate on bounding boxes)
            # then locate the predicted bounding boxes in the scene
            scene_point_cloud_xform = np.matmul(affine_transform[0:3, 0:3], scene_point_cloud.transpose()).transpose()
            center_xform = np.matmul(affine_transform[0:3, 0:3], np.asarray([cx_p, cy_p, cz_p]).transpose())
            condition_x = np.where((scene_point_cloud_xform[:, 0] > (center_xform[0] - 0.55 * size_x_p))
                                   & (scene_point_cloud_xform[:, 0] < (center_xform[0] + 0.55 * size_x_p)))
            condition_y = np.where((scene_point_cloud_xform[:, 1] > (center_xform[1] - 0.55 * size_y_p))
                                   & (scene_point_cloud_xform[:, 1] < (center_xform[1] + 0.55 * size_y_p)))
            condition_z = np.where((scene_point_cloud_xform[:, 2] > (center_xform[2] - 0.55 * size_z_p))
                                   & (scene_point_cloud_xform[:, 2] < (center_xform[2] + 0.55 * size_z_p)))
            location = reduce(np.intersect1d, (condition_x, condition_y, condition_z))
            loc.extend(location)

# remove the scene points within the predicted bounding boxes from the original scene
removal = True
if removal:
    scene_point_cloud_new = np.delete(scene_point_cloud, np.asarray(loc), axis=0)
else:
    scene_point_cloud_new = scene_point_cloud

pcd_scene_new = o3d.geometry.PointCloud()
pcd_scene_new.points = o3d.utility.Vector3dVector(scene_point_cloud_new)
# pcd_scene_new.uniform_down_sample(1000)
viz_data.append(pcd_scene_new)

# show the new scene point cloud with original objects removed
o3d.visualization.draw_geometries([pcd_scene_new],
                                  zoom=2,
                                  front=[0.5439, 0.2333, -0.8060],
                                  lookat=[2.4615, -2.1331, 1.338],
                                  up=[-0.1781, 0.9708, 0.1608])

# show the new scene point cloud with the object replaced with the reconstructed models
o3d.visualization.draw_geometries(viz_data,
                                  zoom=2,
                                  front=[0.5439, 0.2333, -0.8060],
                                  lookat=[2.4615, -2.1331, 1.338],
                                  up=[-0.1781, 0.9708, 0.1608])
# alpha = 0.02
# radii = [0.005, 0.01, 0.02, 0.04]
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_scene_new, alpha)
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_scene_new, o3d.utility.DoubleVector(radii))
# print('run Poisson surface reconstruction')
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_scene_new, depth=9)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# o3d.io.write_triangle_mesh("scene_reconstruction.ply", mesh)
# www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
