
import numpy as np
from scipy import linalg
from tqdm import tqdm
from numba import jit

from utils import *


def skew2vec(x_hat: np.ndarray)-> np.ndarray:
    """
    hat map so3 to vector
    :param x_hat:
    :return:
    """
    x1, x2, x3 = x_hat[2, 1], x_hat[0, 2], x_hat[1, 0]
    return np.vstack((x1, x2, x3))


@jit(nopython=True)
def vec2skew(x: np.ndarray) -> np.ndarray:
    """
    vector to hat map so3
    :param x: vector
    :type x: numpy array vector
    :return: skew symmetric matrix
    """
    x1, x2, x3 = x[0], x[1], x[2]
    x_hat = np.array([[0, -x3, x2],
                      [x3, 0, -x1],
                      [-x2, x1, 0]],
                     dtype=np.float64)
    return x_hat


def vec2twist_hat(x: np.ndarray) -> np.ndarray:
    """
    vector to twist hat map se3
    :param x: 6x1
    :return: 4x4 twist matrices
            [w^(t) v(t)]
            [0,     0]
    """
    assert x.size == 6
    return np.block([[vec2skew(x[3:6, 0]), x[0:3, 0].reshape(3, 1)],
                     [np.zeros((1, 4))]])


def vec2twist_wedge(x):
    """
    vector to twist wedge se3
    :param x: 6x1
    :return: 6x6 twist matrices
    """
    assert x.size == 6
    return np.block([[vec2skew(x[3:6, 0]), vec2skew(x[0:3, 0])],
                     [np.zeros((3, 3)), vec2skew(x[3:6, 0])]])


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


def get_T(Rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Calculate Rigid Body Pose
    :param: R: rotation matrix
    :type: 3x3 numpy array
    :param: p: translation matrix
    :type: (3,) numpy array
    :return: pose T [R P
                     0.T 1 ]
    """
    assert isinstance(Rot, np.ndarray)
    assert isinstance(pos, np.ndarray)
    assert np.size(Rot) == 9
    assert pos.ndim == 1
    x, y, z = pos
    T = np.array([[0, 0, 0, x],
                  [0, 0, 0, y],
                  [0, 0, 0, z],
                  [0, 0, 0, 1]])
    T[0:3, 0:3] = Rot
    return T


@jit(nopython=True)
def projection(q: np.ndarray, derivative=True) -> np.ndarray:
    """
    Projection Function
    π(q) := 1/q3 @ q  ∈ R^{4}
    :param q: numpy.array
    :param derivative:
    """
    # assert isinstance(q, np.ndarray)
    # Prevent Divide by zero error
    q3 = q[2] + 1e-9
    pi_q = q / q3
    return pi_q


@jit(nopython=True)
def projection_derivative(q: np.ndarray) -> np.ndarray:
    """
    Projection Function Derivative
    calculate dπ(q)/dq
    :param q: numpy.array
    return: dπ(q)/dq
    """
    dpi_dq = np.eye(4)
    dpi_dq[2, 2] = 0.0
    dpi_dq[0, 2] = -q[0] / q[2]
    dpi_dq[1, 2] = -q[1] / q[2]
    dpi_dq[3, 2] = -q[3] / q[2]
    dpi_dq = dpi_dq / q[2]
    return dpi_dq


@jit(nopython=True)
def get_M(fs_u: float, fs_v: float, cu: float, cv: float, b: float) -> np.ndarray:
    """
    Stereo Camera Calibration Matrix
    :param: fs_u: focal length [m],  pixel scaling [pixels/m]
    :param: fs_v: focal length [m],  pixel scaling [pixels/m]
    :param: cu: principal point [pixels]
    :param: cv: principal point [pixels]
    :param: b: stereo baseline [m]
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
    print("'''test pi'''")
    q = np.array([1, 2, 3, 1])
    pi_q = projection(q)
    dq = projection_derivative(q)
    print(q.shape)  # (4,)
    print(q)
    print(pi_q.shape)  # (4,)
    print(pi_q)
    print(dq.shape)  # (4,4)
    print(dq)
    ############################
    ''' vec2hat test '''
    print("''' vec2hat test '''")
    x = np.array([1, 2, 3])
    x_hat = vec2skew(x)
    print(x)
    print(x_hat.shape)
    print(x_hat)
    ############################
    pass


if __name__ == '__main__':
    np.seterr(all='raise')
    VERBOSE = False
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
    # CAM Param
    fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
    fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
    cu = K[0, 2]    # principal point [pixels]
    cv = K[1, 2]    # principal point [pixels]
    b = float(b)    # stereo baseline [m]
    # Stereo camera intrinsic calibration matrix M
    M = get_M(fs_u, fs_v, cu, cv, b)

    # imu_T_cam and cam_T_imu
    cam_T_imu = np.linalg.inv(imu_T_cam)  # transformation O_T_I from the IMU to camera optical frame (extrinsic param)
    if VERBOSE:
        print(f"K: {K.shape}\n{K}\n")
        print(f"M: {M.shape}\n{M}\n")
        print(f"imu_T_cam: {imu_T_cam.shape}\n{imu_T_cam}\n")
        print(f"cam_T_imu: {cam_T_imu.shape}\n{cam_T_imu}")
    toc(start_load, name="Loading Data")
    ###################################################################################################################
    '''Init pose_trajectory '''
    pose_trajectory = np.zeros((4, 4, np.size(t)), dtype=np.float64)
    # At t = 0, R=eye(3) p =zeros(3)
    T_t = np.eye(4)
    pose_trajectory[:, :, 0] = T_t

    '''Dead Reckoning'''
    for i in tqdm(range(1, np.size(t))):
        tau = t[0, i] - t[0, i - 1]
        # Generalized velocity:[vt wt].T 6x1
        u_t = np.vstack((linear_velocity[:, i].reshape(3, 1), angular_velocity[:, i].reshape(3, 1)))  # u(t) \in R^{6}
        u_t_hat = vec2twist_hat(u_t)  # ξ^ \in R^{4x4}
        u_t_wedge = vec2twist_wedge(u_t)  # ξ` \in R^{6x6}
        T_t = T_t@linalg.expm(tau * u_t_hat)
        pose_trajectory[:, :, i] = T_t
        if i % 500 == 0:
            visualize_trajectory_2d(pose_trajectory, show_ori=True)
    visualize_trajectory_2d(pose_trajectory, show_ori=True)
    # TODO: find world_T_imu -> T_t     Tt:= W_T_I,t
    '''Observation model
    z = h(T_t, mj)+vt(noise)          vt ∼ N (0, I ⊗ V) = diag[V...V]  
    1. send mj from {w} to {C}
        world_T_cam = world_T_imu @ imu_T_cam
        m_o_ = o_T_imu @ inv(T_t) mj_
    2. proj m_o_ into image plane
        m_i_ = π(m_o)
    3. Apply intrinsic M
    z_i = M π(m_o_j) + vt(noise)
    '''
    ###################################################################################################################
    # (a) IMU Localization via EKF Prediction

    # (c) Landmark Mapping via EKF Update

    # (d) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori=True)
