import numpy as np
import open3d as o3d
import copy


def icp_registration(source, target, trans_init, threshold, reg_method, max_iter, draw=False):
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-plane ICP")
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        reg_method, o3d.pipelines.registration.ICPConvergenceCriteria(max_iter))
    print(reg_icp)
    print("Initial transformation is: \n", trans_init)
    print("Final transformation is: \n", reg_icp.transformation)
    # print(reg_icp.transformation)
    if draw:
        draw_registration_result(source, target, reg_icp.transformation)

    return reg_icp.transformation


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
