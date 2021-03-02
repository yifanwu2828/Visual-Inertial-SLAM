import time

import numpy as np
from tqdm import tqdm
from numba import jit

from utils import *


def main():
    '''
        data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra
        credit filename = "./data/03.npz"
        t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    '''
    pass


if __name__ == '__main__':
    np.seterr(all='raise')
    '''
    data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part
    t: time stamp 
                with shape 1*t
    features: visual feature point coordinates in stereo images, 
                with shape 4*n*t, where n is number of features
    linear_velocity: velocity measurements in IMU frame
                with shape 3*t
    angular_velocity: angular velocity measurements in IMU frame
                with shape 3*t
    K: (left)camera intrinsic matrix
                with shape 3*3
    b: stereo camera baseline
                with shape 1
    imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
                with shape 4*4
    '''
    start_load = tic("########## Loading Data 1 ##########")
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename, load_features=True)
    '''
    K = [fs_u,  fs_theta    cu]
        [0,     fsv,        cv]
        [0,     0,          1 ]
    '''
    fs_theta = K[0, 1]
    fs_u = K[0, 0]
    fs_v = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]

    toc(start_load, name="Loading Data")
    # (a) IMU Localization via EKF Prediction


    # (c) Landmark Mapping via EKF Update

    # (d) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori=True)
