import time

import numpy as np
from tqdm import tqdm
from numba import jit

from utils import *


def tic(message=None):
    if message:
        print(message)
    else:
        print("############ Time Start ############")
    return time.time()


def toc(t_start, name="Operation"):
    print(f'############ {name} took: {(time.time() - t_start):.4f} sec. ############\n')


def main():
    '''
        data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra
        credit filename = "./data/03.npz"
        t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
    '''
    pass


if __name__ == '__main__':
    '''
    data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part
    '''
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename, load_features=True)

    # (a) IMU Localization via EKF Prediction


    # (c) Landmark Mapping via EKF Update

    # (d) Visual-Inertial SLAM

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu, show_ori=True)
