from scipy import linalg
from tqdm import tqdm
from numba import njit

from utils import *


def show_map(pose, landmarks):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-', label="pose_trajectory", linewidth=9)
    ax.plot(landmarks[0, :], landmarks[1, :], 'bo', markersize=1, label="landmark", linewidth=1)
    ax.set_xlim([-1200, 500])
    ax.set_ylim([-900, 900])
    plt.show()


def skew2vec(x_hat: np.ndarray) -> np.ndarray:
    """
    hat map so3 to vector
    :param x_hat:
    :return:
    """
    x1, x2, x3 = x_hat[2, 1], x_hat[0, 2], x_hat[1, 0]
    return np.vstack((x1, x2, x3))


@njit
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


def vec2twist_adj(x):
    """
    vector to twist ad(se3)
    :param x: 6x1
    :return: 6x6 twist matrices
    """
    assert x.size == 6
    return np.block([[vec2skew(x[3:6, 0]), vec2skew(x[0:3, 0])],
                     [np.zeros((3, 3)), vec2skew(x[3:6, 0])]])


@njit
def reg2homo(X: np.ndarray) -> np.ndarray:
    """
    Convert Matrix to homogenous coordinate
    :param X: matrix/vector
    :type :numpy array
    return X_ -> [[X]
                  [1]]
    """
    ones = np.ones((1, X.shape[1]), dtype=np.float64)
    X_ = np.concatenate((X, ones), axis=0)
    return X_


@njit
def projection(q: np.ndarray) -> np.ndarray:
    """
    Projection Function
    π(q) := 1/q3 @ q  ∈ R^{4}
    :param q: numpy.array
    """
    pi_q = q / q[2, :]
    return pi_q


@njit
def projection_derivative(q: np.ndarray) -> np.ndarray:
    """
    Projection Function Derivative
    calculate dπ(q)/dq
    :param q: numpy.array
    return: dπ(q)/dq 4x4
    """
    if q[2]==0:
        q[2]=1e-8
    dpi_dq = np.eye(4)
    dpi_dq[2, 2] = 0.0
    dpi_dq[0, 2] = -q[0] / q[2]
    dpi_dq[1, 2] = -q[1] / q[2]
    dpi_dq[3, 2] = -q[3] / q[2]
    dpi_dq = dpi_dq / q[2]
    return dpi_dq


@njit
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
                 dtype=np.float64)
    return M


def get_obs_model_Jacobian(M, cam_T_world, Mt, update_feature_index, mu) -> np.ndarray:
    """
    Observation model Jacobian H_{t+1} ∈ R^{4Nt×3M}
    :param M: 4x4 stereo camera calibration matrix
    :param cam_T_world: 4x4 transformation matrix {W} -> {CAM}
    :param Mt:total_number_of_landmarks: 132896
    :param update_feature_index:
    :param mu: landmarks_mu_t
    :return: H_{t+1}
    """
    # Transpose of Projection Matrix
    P_T = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]],
                   dtype=np.float64)
    Nt = len(update_feature_index)
    H = np.zeros((4 * Nt, 3 * Mt), dtype=np.float64)  # Ht+1 ∈ R^{4Nt×3M}
    for j in range(Nt):
        current_index = update_feature_index[j]
        # 4x4 4x4(4x4@4xNt) @4x4
        dpi_dq = projection_derivative(cam_T_world @ mu[:, j])
        H_ij = M @ dpi_dq @ cam_T_world @ P_T  # H_ij∈ R^{4×3}
        H[j * 4:(j + 1) * 4, current_index * 3:(current_index + 1) * 3] = H_ij
    return H


def get_mapping_kalman_gain(sigma, H, Nt, v=1):
    """
    Calculate Kalman Gain
    :return:
    """
    # V symmetric, sigma symmetric
    # H_sigma @ H.T symmetric
    # H_sigma @ H.T+ V symmetric -> S.T = S
    V = np.kron(np.eye(4 * Nt), v)
    H_sigma = H @ sigma
    S = H_sigma @ H.T + V
    S_T = S

    K_T, _, _, _ = np.linalg.lstsq(S_T, H_sigma, rcond=None)
    return K_T.T, H_sigma


def get_predicted_obs(M, cam_T_world, mu):
    """
    Predicted observations based on µt
    :param M: 4x4 stereo camera calibration matrix
    :param cam_T_world: 4x4 transformation matrix {W} -> {CAM}
    :param mu: landmarks_mu_t
    :return: z_pred
    """
    return M @ projection(cam_T_world @ mu)


@njit
def update_obs_sigma(sigma, K, H_sigma):
    """
    EKF update sigma
    :param sigma:
    :param K: Kalman gain
    :param H_sigma: Jacobian @ sigma
    :return:
    """
    return sigma - K @ H_sigma


@njit
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


@njit
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
    m_o = np.ones((4, pixels.shape[1]))
    m_o[2, :] = fs_u * b / (pixels[0, :] - pixels[2, :])
    m_o[1, :] = (pixels[1, :] - cv) / fs_v * m_o[2, :]
    m_o[0, :] = (pixels[0, :] - cu) / fs_u * m_o[2, :]
    # Transform from pixel to world frame in in homogenous coordinates
    m_world = world_T_cam @ m_o
    return m_world


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
    '''Observation model
    z = h(T_t, mj)+vt(noise)     Tt:= W_T_I,t       vt ∼ N (0, I ⊗ V) = diag[V...V]
    1. send mj from {w} to {C}
        world_T_cam = world_T_imu @ imu_T_cam
        m_o_ = o_T_imu @ inv(T_t) mj_ -- implemented
    2. proj m_o_ into image plane
        m_i_ = π(m_o)
    3. Apply intrinsic M
    z_i = M π(m_o_j) + vt(noise)
    '''
    ############################
    '''test Kalman Gain '''
    Nt = 10
    m = 2
    sigma = np.eye(3 * 2)
    H = np.ones((4 * Nt, 3 * m))
    k = get_mapping_kalman_gain(sigma, H, Nt, v=5)

    V = np.kron(np.eye(4 * Nt), 5)
    k1 = sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + V)
    print(np.allclose(k, k1))
    assert abs(np.sum(k - k1)) < 1e-4
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
                (4, 13289, 3026)
                (pixels, landmarks, timestamps)
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
    print(f"features: {features.shape}")

    # select half of features
    percent = 3
    lst = [i for i in range(0, features.shape[1]) if not i % percent == 0]
    print(lst)
    features = np.delete(features, lst, axis=1)
    print(f"features_subset: {features.shape}")

    num_timestamps = features.shape[2]
    num_landmarks = features.shape[1]  # M
    # velocity
    vt_x_sigma, vt_y_sigma, vt_z_sigma = velocity_std(linear_velocity)
    wt_r_sigma, wt_p_sigma, wt_y_sigma = velocity_std(angular_velocity)
    cov_diag = np.array([vt_x_sigma, vt_y_sigma, vt_z_sigma, wt_r_sigma, wt_p_sigma, wt_y_sigma],
                        dtype=np.float64) ** 2
    # CAM Param
    fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
    fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
    cu = K[0, 2]  # principal point [pixels]
    cv = K[1, 2]  # principal point [pixels]
    b = float(b)  # stereo baseline [m]
    # Stereo camera intrinsic calibration matrix M
    M = get_M(fs_u, fs_v, cu, cv, b)
    # transformation O_T_I from the IMU to camera optical frame (extrinsic param)
    cam_T_imu = np.linalg.inv(imu_T_cam)
    if VERBOSE:
        print(f"vt_sigma: {vt_x_sigma, vt_y_sigma, vt_z_sigma}")
        print(f"wt_sigma: {wt_r_sigma, wt_p_sigma, wt_y_sigma}")
        print(f"K: {K.shape}\n{K}\n")
        print(f"M: {M.shape}\n{M}\n")
        print(f"imu_T_cam: {imu_T_cam.shape}\n{imu_T_cam}\n")
        print(f"cam_T_imu: {cam_T_imu.shape}\n{cam_T_imu}")
    del filename
    del fs_u, fs_v, cu, cv
    del vt_x_sigma, vt_y_sigma, vt_z_sigma
    del wt_r_sigma, wt_p_sigma, wt_y_sigma
    toc(start_load, name="Loading Data")
    ###################################################################################################################
    '''Init pose_trajectory '''
    pose_trajectory = np.zeros((4, 4, num_timestamps), dtype=np.float64)
    # At t = 0, R=eye(3) p =zeros(3)
    T_imu_mu_t = np.eye(4)
    imu_sigma_t = np.diag(cov_diag)
    pose_trajectory[:, :, 0] = T_imu_mu_t
    '''Init landmarks '''
    landmarks_mu_t = np.zeros((3, num_landmarks), dtype=np.float64)  # µt ∈ R^{3M} with homogenous coord
    landmarks_sigma_t = np.eye(3 * num_landmarks, dtype=np.float64)  # Σt ∈ R^{3M×3M}
    identity = np.eye(3*num_landmarks)
    obs_mu_t = -1 * np.ones((4, num_landmarks), dtype=np.float64)
    ###################################################################################################################
    '''Debug Var'''
    idx = set()
    ##################################################################################################################
    for i in tqdm(range(1, num_timestamps)):
        tau = t[0, i] - t[0, i - 1]
        # Generalized velocity:[vt wt].T 6x1
        u_t = np.vstack((linear_velocity[:, i].reshape(3, 1),
                         angular_velocity[:, i].reshape(3, 1)))  # u(t) \in R^{6}
        u_t_hat = vec2twist_hat(u_t)  # ξ^ \in R^{4x4}
        u_t_adj = vec2twist_adj(u_t)  # ξ` \in R^{6x6}
        # Discrete-time Pose Kinematics:
        T_imu_mu_t = T_imu_mu_t @ linalg.expm(tau * u_t_hat)
        imu_T_world = np.linalg.inv(T_imu_mu_t)
        pose_trajectory[:, :, i] = T_imu_mu_t

        # world frame to cam frame
        cam_T_world = cam_T_imu @ imu_T_world
        # cam frame to world  frame
        world_T_cam = T_imu_mu_t @ imu_T_cam

        # Valid observed features at time t
        features_t = features[:, :, i]
        feature_index = tuple(np.where(np.sum(features_t, axis=0) > -4)[0])
        update_feature_index = []
        update_feature = np.empty((4, 0), dtype=np.float64)

        # landmarks are observed
        num_obs = len(feature_index)
        if num_obs != 0:
            # Extract observed_features_pixels
            observed_features_pixels = features_t[:, feature_index]
            # Transform pixels to world frame in homogenous coord
            m_world_ = pixel2world(observed_features_pixels, K, b, world_T_cam)

            for j in range(num_obs):
                current_index = feature_index[j]
                # if first time seen, initialize landmark
                if np.array_equal(obs_mu_t[:, current_index], np.array([-1, -1, -1, -1])):
                    obs_mu_t[:, current_index] = observed_features_pixels[:, j]
                    landmarks_mu_t[:, current_index] = np.delete(m_world_[:, j], 3, axis=0)

                # else update landmark position,
                # transform the world frame position to camera frame, and calculate re-projection error
                else:
                    update_feature_index.append(current_index)
                    update_feature = np.hstack((update_feature, m_world_[:, j].reshape(4, 1)))

            # if update_feature is not empty
            Nt = len(update_feature_index)
            if Nt != 0:  # and False:
                mu_t_j = reg2homo(landmarks_mu_t[:, update_feature_index])
                # plt.scatter(mu_t_j[0,:], mu_t_j[1,:])
                # print(f"{Nt},({4 * Nt},{num_landmarks})")
                H = get_obs_model_Jacobian(M, cam_T_world, num_landmarks, update_feature_index, mu_t_j)

                z = features_t[:, update_feature_index].reshape(-1, 1,)
                z_pred = get_predicted_obs(M, cam_T_world, mu_t_j).reshape(-1, 1,)

                K_map, H_sigma = get_mapping_kalman_gain(landmarks_sigma_t, H, Nt, v=5000)
                landmarks_mu_t = landmarks_mu_t.reshape(-1, 1) + K_map @ (z - z_pred)
                landmarks_mu_t = landmarks_mu_t.reshape(3, -1)

                landmarks_sigma_t = update_obs_sigma(landmarks_sigma_t, K_map, H_sigma)

    visualize_trajectory_2d(pose_trajectory, show_ori=True)
    show_map(pose_trajectory, landmarks_mu_t)

    ###########################################################################################################

    ###################################################################################################################
    # (a) IMU Localization via EKF Prediction

    # (c) Landmark Mapping via EKF Update

    # (d) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori=True)

