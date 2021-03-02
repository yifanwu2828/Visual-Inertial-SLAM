import time

import numpy as np
from tqdm import tqdm
from numba import jit

from utils import *


@jit(nopython=True)
def reg2homo(X: np.ndarray) -> np.ndarray:
    """
    Convert Matrix to homogenous coordinate
    :param X: matrix/vector
    :type :numpy array
    return X_ -> [[X]
                  [1]]
    """
    # assert isinstance(X, np.ndarray)
    ones = np.ones((1, X.shape[1]), dtype=np.float64)
    X_ = np.concatenate((X, ones), axis=0)
    return X_


def projection(q: np.ndarray, derivative=True) -> np.ndarray:
    """
    Projection Function
    π(q) := 1/q3 @ q  ∈ R^{4}
    :param q: numpy.array
    :param derivative: calculator dπ(q)/dq
    """
    assert isinstance(q, np.ndarray)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    pi_q = q / q3
    dpi_dq = None
    if derivative:
        dpi_dq = np.array([[1, 0, -q1 / q3, 0],
                           [0, 1, -q2 / q3, 0],
                           [0, 0, 0, 0],
                           [0, 0, -q4 / q3, 1]],
                          dtype=np.float64)

    return pi_q, dpi_dq


@jit(nopython=True)
def get_M(fs_u: float, fs_v: float, cu: float, cv: float, b: float) -> np.ndarray:
    """
    Stereo Camera Calibration Matrix
    :param fs_u: focal length [m],  pixel scaling [pixels/m]
    :param fs_v: focal length [m],  pixel scaling [pixels/m]
    :param cu: principal point [pixels]
    :param cv: principal point [pixels]
    :param b: stereo baseline [m]
    :return 4x4 Intrinsic Matrix
    """
    M = np.array([[fs_u, 0, cu, 0],
                  [0, fs_v, cv, 0],
                  [fs_u, 0, cu, -fs_u * b],
                  [0, fs_v, cv, 0]],
                 dtype=np.float64)
    return M


def main():
    """
        data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra
        credit filename = "./data/03.npz"
        t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    """
    ############################
    '''test pi'''
    q = np.array([1, 2, 3, 1])
    pi_q, dq = projection(q)
    print(q.shape)  # (4,)
    print(q)
    print(pi_q.shape)  # (4,)
    print(pi_q)
    print(dq.shape)  # (4,4)
    print(dq)
    ############################
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
    ###################################################################################################################
    start_load = tic("########## Loading Data 1 ##########")
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename, load_features=True)
    del filename
    '''
    K = [fs_u,  fs_theta=0  cu]
        [0,     fsv,        cv]
        [0,     0,          1 ]
    '''
    # CAM Param
    fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
    fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
    cu = K[0, 2]  # principal point [pixels]
    cv = K[1, 2]  # principal point [pixels]
    b = float(b)  # stereo baseline [m]
    M = get_M(fs_u, fs_v, cu, cv, b)  # intrinsic matrix

    # imu_T_cam and cam_T_imu
    cam_T_imu = np.linalg.inv(imu_T_cam)  # transformation O_T_I from the IMU to camera optical frame (extrinsic param)

    toc(start_load, name="Loading Data")
    ###################################################################################################################
    '''Dead Reckoning'''











    ###################################################################################################################
    # (a) IMU Localization via EKF Prediction

    # (c) Landmark Mapping via EKF Update

    # (d) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori=True)
