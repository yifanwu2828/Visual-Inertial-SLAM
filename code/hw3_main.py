from scipy import linalg
from tqdm import tqdm
from numba import njit, prange

from utils import *


def show_map(pose, landmarks):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-', label="pose_trajectory", linewidth=9)
    ax.plot(landmarks[0, :], landmarks[1, :], 'bo', markersize=1, label="landmark", linewidth=1)
    # ax.set_xlim([-1200, 500])
    ax.set_ylim([-1200, 600])
    plt.show()


@njit(cache=True, fastmath=True)
def skew2vec(x_hat: np.ndarray) -> np.ndarray:
    """
    hat map so3 to vector
    :param x_hat:
    :return:
    """
    x1, x2, x3 = x_hat[2, 1], x_hat[0, 2], x_hat[1, 0]
    return np.vstack((x1, x2, x3))


@njit(cache=True, fastmath=True)
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
                     dtype=np.float32)
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
                     [np.zeros((1, 4), dtype=np.float32)]])


def vec2twist_adj(x: np.ndarray) -> np.ndarray:
    """
    vector to twist ad(se3)
    :param x: 6x1
    :return: 6x6 twist matrices
    """
    assert x.size == 6
    return np.block([[vec2skew(x[3:6, 0]), vec2skew(x[0:3, 0])],
                     [np.zeros((3, 3), dtype=np.float32), vec2skew(x[3:6, 0])]])


def circle_dot(s: np.ndarray) -> np.ndarray:
    """
    circle_dot
    :param s: 4x1 in homogenous coordinate
    :return: 4x6
    """
    s = s.reshape(-1)
    assert s.size == 4
    assert abs(s[-1] - 1) < 1e-8
    res = np.hstack([np.eye(3, dtype=np.float32), -vec2skew(s[:3])])
    res = np.vstack([res, np.zeros((1, 6), dtype=np.float32)])
    return res


@njit(cache=True, fastmath=True)
def reg2homo(X: np.ndarray) -> np.ndarray:
    """
    Convert Matrix to homogenous coordinate
    :param X: matrix/vector
    :type :numpy array
    return X_ -> [[X]
                  [1]]
    """
    ones = np.ones((1, X.shape[1]), dtype=np.float32)
    X_ = np.concatenate((X, ones), axis=0)
    return X_


@njit(cache=True, fastmath=True)
def projection(q: np.ndarray) -> np.ndarray:
    """
    Projection Function
    π(q) := 1/q3 @ q  ∈ R^{4}
    :param q: numpy.array
    """
    pi_q = q / q[2, :]
    return pi_q


@njit(cache=True, fastmath=True)
def projection_derivative(q: np.ndarray) -> np.ndarray:
    """
    Projection Function Derivative
    calculate dπ(q)/dq
    :param q: numpy.array
    return: dπ(q)/dq 4x4
    """
    dpi_dq = np.eye(4, dtype=np.float32)
    dpi_dq[2, 2] = 0.0
    dpi_dq[0, 2] = -q[0] / q[2]
    dpi_dq[1, 2] = -q[1] / q[2]
    dpi_dq[3, 2] = -q[3] / q[2]
    dpi_dq = dpi_dq / q[2]
    return dpi_dq


@njit(cache=True, fastmath=True)
def get_M(fsu: float, fsv: float, cu: float, cv: float, b: float) -> np.ndarray:
    """
    Stereo Camera Calibration Matrix
    :param: fs_u: focal length [m],  pixel scaling [pixels/m]
    :param: fs_v: focal length [m],  pixel scaling [pixels/m]
    :param: cu: principal point [pixels]
    :param: cv: principal point [pixels]
    :param: b: stereo baseline [m]
    :return 4x4 stereo camera calibration matrix
    """
    M = np.array([[fsu, 0, cu, 0],
                  [0, fsv, cv, 0],
                  [fsu, 0, cu, -fsu * b],
                  [0, fsv, cv, 0]],
                 dtype=np.float32)
    return M


def get_obs_model_Jacobian(M, cam_T_world, Mt, update_feature_index, mu, Nt=None, P_T=None) -> np.ndarray:
    """
    Observation Model Jacobian H_{t+1} ∈ R^{4Nt×3M}
    :param M: 4x4 stereo camera calibration matrix
    :param cam_T_world: 4x4 transformation matrix {W} -> {CAM}
    :param Mt: number_of_landmarks
    :param update_feature_index: index of update features
    :param mu: landmarks_mu_t: mean of landmarks position in world frame
    :param Nt: number of update features
    :param P_T: Transpose of Projection Matrix
    :return: H_{t+1}
    """
    # Transpose of Projection Matrix
    if P_T is None:
        P_T = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]],
                       dtype=np.float32)
    if Nt is None:
        Nt = update_feature_index.size

    H = np.zeros((4 * Nt, 3 * Mt), dtype=np.float32)  # Ht+1 ∈ R^{4Nt×3M}
    for j in prange(Nt):
        index = update_feature_index[j]
        dpi_dq = projection_derivative(cam_T_world @ mu[:, j])
        # H_ij = M @ dpi_dq @ cam_T_world @ P_T  # H_ij∈ R^{4×3}
        H_ij = np.linalg.multi_dot([M, dpi_dq, cam_T_world, P_T])  # H_ij∈ R^{4×3}
        H[j * 4:(j + 1) * 4, index * 3:(index + 1) * 3] = H_ij
    return H


@njit(cache=True, parallel=True, fastmath=True)
def get_motion_model_Jacobian(M, cam_T_imu, T_imu_inv, m, Nt):
    H = np.empty((4 * Nt, 6), dtype=np.float32)  # Ht+1 ∈{4Ntx6}
    for j in prange(Nt):
        prod = T_imu_inv @ m[:, j]
        dpi_dq = projection_derivative(cam_T_imu @ prod)
        s_circle_dot = circle_dot(prod)
        H_ij = np.linalg.multi_dot([-M, dpi_dq, cam_T_imu, s_circle_dot])
        H[j * 4:(j + 1) * 4, :] = H_ij
    return H


def get_update_Jacobian(M, cam_T_imu, T_imu_inv, Mt, update_feature_index, m, Nt=None, cam_T_world=None, P_T=None):
    """
    Combined Jacobian
    :param M: 4x4 stereo camera calibration matrix
    :param cam_T_imu: 4x4 transformation matrix {IMU} -> {CAM}
    :param T_imu_inv: 4x4 transformation matrix {W} -> {IMU}
    :param Mt: number_of_landmarks
    :param update_feature_index: index of update features
    :param m: landmarks position in world frame
    :param Nt: number of update features
    :param cam_T_world: 4x4 transformation matrix {W} -> {CAM}
    :param P_T: Transpose of Projection Matrix
    :return:
    """
    if P_T is None:
        P_T = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]],
                       dtype=np.float32)
    if Nt is None:
        Nt = update_feature_index.size
    if cam_T_world is None:
        cam_T_world = cam_T_imu @ T_imu_inv

    H = np.zeros((4 * Nt, 3 * Mt + 6), dtype=np.float32)  # Ht+1 ∈{4Ntx3M+6}

    for j in range(Nt):
        index = update_feature_index[j]
        dpi_dq_map = projection_derivative(cam_T_world @ m[:, j])
        H_ij_obs = np.linalg.multi_dot([M, dpi_dq_map, cam_T_world, P_T])  # H_ij∈ R^{4×3}
        H[j * 4:(j + 1) * 4, index * 3:(index + 1) * 3] = H_ij_obs

        prod = T_imu_inv @ m[:, j]
        dpi_dq_loc = projection_derivative(cam_T_imu @ prod)
        s_circle_dot = circle_dot(prod)
        H_ij_act = np.linalg.multi_dot([-M, dpi_dq_loc, cam_T_imu, s_circle_dot])  # H_ij∈ R^{4×6}
        H[j * 4:(j + 1) * 4, -6:] = H_ij_act

    return H


def get_kalman_gain(sigma, H, Nt, lsq=False, v=100):
    """
    Calculate Kalman Gain
    :param sigma:
    :param H: Jacobian
    :param Nt: number of update observations
    :param lsq:use lsq (faster) least-squares solution or solve (slower) exact solution
    :param v: noise constant
    :return:
    """
    # V symmetric, sigma symmetric
    # H_sigma @ H.T symmetric
    # H_sigma @ H.T+ V symmetric -> S.T = S
    V = np.kron(np.eye(4 * Nt, dtype=np.float32), v)
    H_sigma = H @ sigma
    S_T = H_sigma @ H.T + V
    if lsq:
        K_T, _, _, _ = np.linalg.lstsq(S_T, H_sigma, rcond=None)
    else:
        K_T = np.linalg.solve(S_T, H_sigma)
    return K_T.T, H_sigma


@njit(cache=True, fastmath=True)
def velocity_std(vt: np.ndarray):
    """
    Calculate the std of linear and angular velocity
    :param vt: 3x3026
    :type:numpy array
    :return:
    """
    v_x, v_y, v_z = vt[0, :], vt[1, :], vt[2, :]
    v_x_sigma = np.std(v_x)
    v_y_sigma = np.std(v_y)
    v_z_sigma = np.std(v_z)
    return v_x_sigma, v_y_sigma, v_z_sigma


@njit(cache=True, fastmath=True)
def pixel2world(pixels: np.ndarray, K: np.ndarray, b: float, world_T_cam: np.ndarray) -> np.ndarray:
    """
    Convert from pixels to world coordinates
    :param pixels: pixel coordinates with number of observations
    :param K: (left)camera intrinsic matrix with shape 3*3
    :param b: stereo camera baseline with shape 1
    :param world_T_cam: pose from camera to world
    :return: world frame landmark positions in homogenous coordinates
    """
    fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
    fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
    cu = K[0, 2]  # principal point [pixels]
    cv = K[1, 2]  # principal point [pixels]
    m_o = np.ones((4, pixels.shape[1]), dtype=np.float32)
    m_o[2, :] = fs_u * b / (pixels[0, :] - pixels[2, :])
    m_o[1, :] = (pixels[1, :] - cv) / fs_v * m_o[2, :]
    m_o[0, :] = (pixels[0, :] - cu) / fs_u * m_o[2, :]
    # Transform from pixel to world frame in in homogenous coordinates
    m_world = world_T_cam.astype(np.float32) @ m_o
    return m_world


def main():
    # np.seterr(all='raise')
    # VERBOSE = False
    # ###################################################################################################################
    # start_load = tic("########## Loading Data 1 ##########")
    # filename = "./data/10.npz"
    # t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename, load_features=True)
    # t = t.reshape(-1)
    # print(f"features: {features.shape}")
    # num_original_features = features.shape[1]
    #
    # # select subset of features
    # factor = 40  # 10
    # lst = [skip_feature_idx for skip_feature_idx in range(0, features.shape[1]) if not skip_feature_idx % factor == 0]
    # print(lst)
    # features_subset = np.delete(features, lst, axis=1)
    # print(f"Using{features_subset.shape[1] / num_original_features: .2%} to cover entire trajectory")
    # print(f"features_subset: {features_subset.shape}")
    #
    # # velocity
    # vt_x_sigma, vt_y_sigma, vt_z_sigma = velocity_std(linear_velocity)
    # wt_r_sigma, wt_p_sigma, wt_y_sigma = velocity_std(angular_velocity)
    # cov_vec = np.array([vt_x_sigma, vt_y_sigma, vt_z_sigma, wt_r_sigma, wt_p_sigma, wt_y_sigma],
    #                    dtype=np.float64) ** 2
    # cov_diag = np.diag(cov_vec)
    # # CAM Param
    # fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
    # fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
    # cu = K[0, 2]  # principal point [pixels]
    # cv = K[1, 2]  # principal point [pixels]
    # b = float(b)  # stereo baseline [m]
    # # Stereo camera intrinsic calibration matrix M
    # M = get_M(fs_u, fs_v, cu, cv, b)
    # # transformation O_T_I from the IMU to camera optical frame (extrinsic param)
    # cam_T_imu = np.linalg.inv(imu_T_cam)
    # if VERBOSE:
    #     print(f"vt_sigma: {vt_x_sigma, vt_y_sigma, vt_z_sigma}")
    #     print(f"wt_sigma: {wt_r_sigma, wt_p_sigma, wt_y_sigma}")
    #     print(f"K: {K.shape}\n{K}\n")
    #     print(f"M: {M.shape}\n{M}\n")
    #     print(f"imu_T_cam: {imu_T_cam.shape}\n{imu_T_cam}\n")
    #     print(f"cam_T_imu: {cam_T_imu.shape}\n{cam_T_imu}")
    # del filename, num_original_features, VERBOSE, features, lst
    # del fs_u, fs_v, cu, cv
    # del vt_x_sigma, vt_y_sigma, vt_z_sigma
    # del wt_r_sigma, wt_p_sigma, wt_y_sigma, cov_vec
    # toc(start_load, name="Loading Data")
    # ###################################################################################################################
    # '''Init Var'''
    # # Transpose of Projection Matrix
    # P_T = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1],
    #                 [0, 0, 0]],
    #                dtype=np.float64)
    # # indicator
    # unobserved = np.array([-1, -1, -1, -1], dtype=np.int8)
    # num_timestamps = features_subset.shape[2]
    # num_landmarks = features_subset.shape[1]  # M
    # ##################################################################################################################
    # '''Init pose_trajectory '''
    # pose_trajectory = np.empty((4, 4, num_timestamps), dtype=np.float64)
    # # At t = 0, R=eye(3) p =zeros(3)
    # T_imu_mu_t = np.eye(4)  # ∈ R^{4×4}
    # T_imu_sigma_t = np.eye(6)  # ∈ R^{6×6}
    # pose_trajectory[:, :, 0] = T_imu_mu_t
    # '''Init landmarks '''
    # landmarks_mu_t = np.zeros((3 * num_landmarks, 1), dtype=np.float64)  # µt ∈ R^{3M}
    # landmarks_sigma_t = np.eye(3 * num_landmarks, dtype=np.float64)  # Σt ∈ R^{3M×3M}
    # obs_mu_t = -1 * np.ones((4, num_landmarks), dtype=np.int16)
    # ###################################################################################################################
    #
    # for i in tqdm(range(1, num_timestamps)):
    #     tau = t[i] - t[i - 1]
    #     # (a) IMU Localization via EKF Prediction
    #     # Generalized velocity:[vt wt].T 6x1
    #     u_t = np.vstack((linear_velocity[:, i].reshape(3, 1),
    #                      angular_velocity[:, i].reshape(3, 1)))  # u(t) \in R^{6}
    #     u_t_hat = vec2twist_hat(u_t)  # ξ^ \in R^{4x4}
    #     u_t_adj = vec2twist_adj(u_t)  # ξ` \in R^{6x6}
    #
    #     # Discrete-time Pose Kinematics:
    #     T_imu_mu_t = T_imu_mu_t @ linalg.expm(tau * u_t_hat)
    #     imu_T_world = np.linalg.inv(T_imu_mu_t)
    #
    #     perturbation = linalg.expm(-tau * u_t_adj)
    #     # add noise
    #     W = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0, 0], cov=cov_diag).reshape(-1, 1)
    #     noise_adj = vec2twist_adj(W)
    #     noise_pertu = linalg.expm(tau ** 2 * noise_adj)
    #     T_imu_sigma_t = np.linalg.multi_dot([noise_pertu, perturbation, T_imu_sigma_t, perturbation.T, noise_pertu.T])
    #
    #     ###############################################################################################################
    #     # (c) Landmark Mapping via EKF Update
    #     # world frame to cam frame
    #     cam_T_world = cam_T_imu @ imu_T_world
    #     world_T_cam = T_imu_mu_t @ imu_T_cam
    #     # Valid observed features at time t
    #     features_t = features_subset[:, :, i]
    #     feature_index = tuple(np.where(np.sum(features_t, axis=0) > -4)[0])
    #     update_feature_index = []
    #     update_feature = np.empty((4, 0), dtype=np.float64)
    #
    #     # if landmarks are observed
    #     num_obs = len(feature_index)
    #     if num_obs != 0:
    #         # Extract observed_features_pixels
    #         observed_features_pixels = features_t[:, feature_index]
    #         # Transform pixels to world frame in homogenous coord
    #         m_world_ = pixel2world(observed_features_pixels, K, b, world_T_cam)
    #
    #         for j in range(num_obs):
    #             current_index = feature_index[j]
    #             # if first time seen, initialize landmarks
    #             if np.array_equal(obs_mu_t[:, current_index], unobserved):
    #                 obs_mu_t[:, current_index] = observed_features_pixels[:, j]
    #                 landmarks_mu_t = landmarks_mu_t.reshape(3, -1)
    #                 landmarks_mu_t[:, current_index] = np.delete(m_world_[:, j], 3, axis=0)
    #                 landmarks_mu_t = landmarks_mu_t.reshape(-1, 1)
    #             # else update landmark position,
    #             else:
    #                 update_feature_index.append(current_index)
    #                 update_feature = np.hstack((update_feature, m_world_[:, j].reshape(4, 1)))
    #
    #         # if update_feature is not empty
    #         Nt = len(update_feature_index)
    #         if Nt != 0:  # and False:
    #             # To homogenous coordinate
    #             landmarks_mu_t = landmarks_mu_t.reshape(3, -1)
    #             mu_t_j = reg2homo(landmarks_mu_t[:, update_feature_index])
    #             # Re-projection Error
    #             z = features_t[:, update_feature_index]
    #             z_pred = M @ projection(cam_T_world @ mu_t_j)
    #             error = (z - z_pred).reshape(-1, 1)
    #             # TODO: single mu, sigma, H
    #
    #             # Update landmarks_mu, landmarks_sigma and T_imu_mu and T_imu_sigma simultaneously
    #             '''pose update'''
    #             # H_imu = get_motion_model_Jacobian(M, cam_T_imu, imu_T_world, mu_t_j, Nt)
    #             # K_imu, H_imu_sigma = get_kalman_gain(T_imu_sigma_t, H_imu, Nt, lsq=True, v=100)
    #             # T_twist_hat = vec2twist_hat(K_imu @ error)
    #             # T_imu_mu_t = T_imu_mu_t @ linalg.expm(T_twist_hat)
    #             # T_imu_sigma_t = T_imu_sigma_t - K_imu @ H_imu_sigma
    #             '''landmarks update'''
    #             H_map = get_obs_model_Jacobian(M, cam_T_world, num_landmarks,
    #                                            np.array(update_feature_index),
    #                                            mu_t_j, Nt, P_T)
    #             K_map, H_landmarks_sigma = get_kalman_gain(landmarks_sigma_t, H_map, Nt, lsq=True, v=100)
    #             landmarks_mu_t = landmarks_mu_t.reshape(-1, 1) + K_map @ error
    #             landmarks_sigma_t = landmarks_sigma_t - K_map @ H_landmarks_sigma
    #     pose_trajectory[:, :, i] = T_imu_mu_t
    # landmarks_pos = landmarks_mu_t.reshape(3, -1)
    # visualize_trajectory_2d(pose_trajectory, landmarks_pos, show_points=False, show_ori=True)
    # show_map(pose_trajectory, landmarks_pos)
    pass


if __name__ == '__main__':
    np.seterr(all='raise')
    VERBOSE = False
    ###################################################################################################################
    start_load = tic("########## Loading Data 1 ##########")
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename, load_features=True)
    t = t.reshape(-1)
    print(f"features: {features.shape}")
    num_original_features = features.shape[1]

    # select subset of features
    factor = 3  # 10
    lst = [skip_feature_idx for skip_feature_idx in range(0, features.shape[1]) if not skip_feature_idx % factor == 0]
    features_subset = np.delete(features, lst, axis=1)
    print(f"Using{features_subset.shape[1] / num_original_features: .2%} to cover entire trajectory")
    print(f"features_subset: {features_subset.shape}")

    # velocity
    vt_x_sigma, vt_y_sigma, vt_z_sigma = velocity_std(linear_velocity)
    wt_r_sigma, wt_p_sigma, wt_y_sigma = velocity_std(angular_velocity)
    cov_vec = np.array([vt_x_sigma, vt_y_sigma, vt_z_sigma, wt_r_sigma, wt_p_sigma, wt_y_sigma],
                       dtype=np.float32) ** 2
    cov_diag = np.diag(cov_vec)
    # CAM Param
    fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
    fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
    cu = K[0, 2]  # principal point [pixels]
    cv = K[1, 2]  # principal point [pixels]
    b = float(b)  # stereo baseline [m]
    # Stereo camera intrinsic calibration matrix M
    M = get_M(fs_u, fs_v, cu, cv, b)
    # transformation O_T_I from the IMU to camera optical frame (extrinsic param)
    imu_T_cam = imu_T_cam.astype(np.float32)
    cam_T_imu = np.linalg.inv(imu_T_cam)
    if VERBOSE:
        print(f"vt_sigma: {vt_x_sigma, vt_y_sigma, vt_z_sigma}")
        print(f"wt_sigma: {wt_r_sigma, wt_p_sigma, wt_y_sigma}")
        print(f"K: {K.shape}\n{K}\n")
        print(f"M: {M.shape}\n{M}\n")
        print(f"imu_T_cam: {imu_T_cam.shape}\n{imu_T_cam}\n")
        print(f"cam_T_imu: {cam_T_imu.shape}\n{cam_T_imu}")
    del filename, num_original_features, VERBOSE, features, lst
    del fs_u, fs_v, cu, cv
    del vt_x_sigma, vt_y_sigma, vt_z_sigma
    del wt_r_sigma, wt_p_sigma, wt_y_sigma, cov_vec
    toc(start_load, name="Loading Data")
    ###################################################################################################################
    '''Init Var'''
    # Transpose of Projection Matrix
    P_T = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]],
                   dtype=np.float32)
    # indicator
    unobserved = np.array([-1, -1, -1, -1], dtype=np.int8)
    num_timestamps = features_subset.shape[2]
    num_landmarks = features_subset.shape[1]  # M
    ##################################################################################################################
    '''Init pose_trajectory '''
    pose_trajectory = np.empty((4, 4, num_timestamps), dtype=np.float32)
    pose_trajectory2 = np.empty((4, 4, num_timestamps), dtype=np.float32)
    # At t = 0, R=eye(3) p =zeros(3)
    T_imu_mu_t = np.eye(4, dtype=np.float32)  # ∈ R^{4×4}
    T_imu_sigma_t = 1e-3* np.eye(6, dtype=np.float32)  # ∈ R^{6×6}
    pose_trajectory[:, :, 0] = T_imu_mu_t
    '''Init landmarks '''
    landmarks_mu_t = np.zeros((3 * num_landmarks, 1), dtype=np.float32)  # µt ∈ R^{3M}
    landmarks_sigma_t = np.eye(3 * num_landmarks, dtype=np.float32)  # Σt ∈ R^{3M×3M}
    obs_mu_t = -1 * np.ones((4, num_landmarks), dtype=np.float32)

    '''Init combined mean and covariance matrix'''
    mu = np.block([[landmarks_mu_t], [T_imu_mu_t.reshape(-1, 1)]])
    del T_imu_mu_t, landmarks_mu_t
    sigma = np.block([[landmarks_sigma_t, np.zeros((3 * num_landmarks, 6))],
                      [np.zeros((6, 3 * num_landmarks)), T_imu_sigma_t]])
    del T_imu_sigma_t, landmarks_sigma_t
    ###################################################################################################################
    start_SLAM = tic("########## Start Visual-Inertial-SLAM ##########")
    print(f"mu:16+3M {mu.shape}")
    print(f"sigma:3M+6 x 3M+6 {sigma.shape}")
    for i in tqdm(range(1, num_timestamps)):
        tau = t[i] - t[i - 1]
        # (a) IMU Localization via EKF Prediction
        # Generalized velocity:[vt wt].T 6x1
        u_t = np.vstack((linear_velocity[:, i].reshape(3, 1),
                         angular_velocity[:, i].reshape(3, 1)))  # u(t) \in R^{6}
        u_t_hat = vec2twist_hat(u_t)  # ξ^ \in R^{4x4}
        u_t_adj = vec2twist_adj(u_t)  # ξ` \in R^{6x6}

        # Discrete-time Pose Kinematics:
        world_T_imu = mu[-16:].reshape(4, -1) @ linalg.expm(tau * u_t_hat)
        mu[-16:] = world_T_imu.reshape(-1, 1)
        imu_T_world = np.linalg.inv(world_T_imu)

        # add noise
        W = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0, 0], cov=cov_diag).reshape(-1, 1)
        noise_adj = vec2twist_adj(W)
        noise_pertu = linalg.expm(tau ** 2 * noise_adj)
        perturbation = linalg.expm(-tau * u_t_adj)
        # T_imu_sigma_t = sigma[:6, :6]
        sigma[-6:, -6:] = np.linalg.multi_dot(
            [noise_pertu, perturbation, sigma[-6:, -6:], perturbation.T, noise_pertu.T])

        ###############################################################################################################
        # (c) Landmark Mapping via EKF Update
        # world frame to cam frame
        cam_T_world = cam_T_imu @ imu_T_world
        world_T_cam = world_T_imu @ imu_T_cam
        # Valid observed features at time t
        features_t = features_subset[:, :, i]
        feature_index = tuple(np.where(np.sum(features_t, axis=0) > -4)[0])
        update_feature_index = []
        update_feature = np.empty((4, 0), dtype=np.float32)

        # if landmarks are observed
        num_obs = len(feature_index)
        if num_obs != 0:
            # Extract observed_features_pixels
            observed_features_pixels = features_t[:, feature_index]
            # Transform pixels to world frame in homogenous coord
            m_world_ = pixel2world(observed_features_pixels, K, b, world_T_cam)

            for j in range(num_obs):
                current_index = feature_index[j]
                # if first time seen, initialize landmarks
                if np.array_equal(obs_mu_t[:, current_index], unobserved):
                    obs_mu_t[:, current_index] = observed_features_pixels[:, j]

                    landmarks_mu_t = mu[0:-16].reshape(3, -1)
                    landmarks_mu_t[:, current_index] = np.delete(m_world_[:, j], 3, axis=0)
                    mu[0:-16] = landmarks_mu_t.reshape(-1, 1)
                # else update landmark position,
                else:
                    update_feature_index.append(current_index)
                    update_feature = np.hstack((update_feature, m_world_[:, j].reshape(4, 1)))

            # if update_feature is not empty
            Nt = len(update_feature_index)
            if Nt != 0:  # and False:
                # To homogenous coordinate
                landmarks_mu_t = mu[0:-16].reshape(3, -1)
                mu_t_j = reg2homo(landmarks_mu_t[:, update_feature_index])
                # Re-projection Error
                z = features_t[:, update_feature_index]
                z_pred = M @ projection(cam_T_world @ mu_t_j)
                error = (z - z_pred).reshape(-1, 1)
                # Update landmarks_mu, landmarks_sigma and T_imu_mu and T_imu_sigma simultaneously
                H_joint = get_update_Jacobian(M, cam_T_imu, imu_T_world, num_landmarks,
                                              np.array(update_feature_index), mu_t_j,
                                              Nt=Nt, cam_T_world=cam_T_world, P_T=P_T)
                '''pose update'''
                # K: 3M+6 x 4Nt,
                K_joint, H_sigma_joint = get_kalman_gain(sigma, H_joint, Nt, lsq=True, v=500)
                T_twist_hat = vec2twist_hat(K_joint[-6:, :] @ error)
                mu[-16:] = (world_T_imu @ linalg.expm(T_twist_hat)).reshape(-1, 1)
                sigma[-6:, -6:] = sigma[-6:, -6:] - K_joint[-6:, :] @ H_sigma_joint[:, -6:]
                '''landmarks update'''
                mu[0:-16] = landmarks_mu_t.reshape(-1, 1) + K_joint[:-6, :] @ error
                sigma[:-6, :-6] = sigma[:-6, :-6] - K_joint[:-6, :] @ H_sigma_joint[:, :-6]
        # Record pose trajectory at each timestamps
        pose_trajectory[:, :, i] = mu[-16:].reshape(4, -1)
    landmarks_pos = mu[0:-16].reshape(3, -1)
    visualize_trajectory_2d(pose_trajectory, landmarks_pos, show_points=True, show_ori=True)
    show_map(pose_trajectory, landmarks_pos)
    toc(start_load, name="Finish SLAM")
    ###########################################################################################################
