# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

from ceres.envs import CeresEnv
from .obstacles import ObstacleSquare, ObstacleCircle
from .nav2d_pos import Nav2dPos
from .nav2d_force import Nav2dForce
from .nav2d_rendering import Nav2dRendering
from .nav2d_obstacles import Nav2dObstacles
import numpy as np

class FixedMaze(object):
    '''
    Fixed square obstacles defining a maze
    '''
    fixed_obstacles = [
        ObstacleSquare(top_left_x=-0.70, top_left_y = 0.70, bottom_right_x=0.70, bottom_right_y=0.35),
        ObstacleSquare(top_left_x=-1.00, top_left_y = -0.35, bottom_right_x=-0.30, bottom_right_y=-0.70),
        ObstacleSquare(top_left_x=0.30, top_left_y = -0.35, bottom_right_x=1.00, bottom_right_y=-0.70),
        ObstacleSquare(top_left_x=-1.00, top_left_y = 0.05, bottom_right_x=-0.65, bottom_right_y=-0.05),
        ObstacleSquare(top_left_x=0.65, top_left_y = 0.05, bottom_right_x=1.00, bottom_right_y=-0.05),
        ObstacleSquare(top_left_x=-0.40, top_left_y = 0.05, bottom_right_x=0.40, bottom_right_y=-0.05),
        ObstacleSquare(top_left_x=-0.05, top_left_y = 0.40, bottom_right_x=0.05, bottom_right_y=-0.40),
        ObstacleSquare(top_left_x=-0.05, top_left_y = -0.70, bottom_right_x=0.05, bottom_right_y=-1.00),
    ]

class RandomHoles(object):
    '''
    Circle obstacles randomized for every episode
    '''
    n_random_circle_obstacles = 10
    random_circle_obstacle_dim_range = [0.10, 0.25]
    is_state_target_rel_pos = True
    state_lidar_angles = np.linspace(0., 2.*np.pi, 8, endpoint=False)

class Nav2dPosCeres(Nav2dRendering, Nav2dPos, CeresEnv):
    max_reference_trajectories = 1024
    max_recovery_steps = 5

class Nav2dPosFixedMazeCeres(FixedMaze, Nav2dRendering, Nav2dObstacles, Nav2dPos, CeresEnv):
    max_reference_trajectories = 1024
    max_recovery_steps = 5

class Nav2dPosFixedMazeCeres5N(Nav2dPosFixedMazeCeres):
    max_normalized_obs = 5.
    max_normalized_act = 5.

class Nav2dPosRandomHolesCeres(RandomHoles, Nav2dRendering, Nav2dObstacles, Nav2dPos, CeresEnv):
    max_reference_trajectories = 1024
    max_recovery_steps = 5

class Nav2dPosRandomHolesCeres5N(Nav2dPosRandomHolesCeres):
    max_normalized_obs = 5.
    max_normalized_act = 5.

class Nav2dForceCeres(Nav2dRendering, Nav2dForce, CeresEnv):
    max_reference_trajectories = 1024
    max_recovery_steps = 10

class Nav2dForceFixedMazeCeres(FixedMaze, Nav2dRendering, Nav2dObstacles, Nav2dForce, CeresEnv):
    max_reference_trajectories = 1024
    max_recovery_steps = 10

class Nav2dForceRandomHolesCeres(RandomHoles, Nav2dRendering, Nav2dObstacles, Nav2dForce, CeresEnv):
    max_reference_trajectories = 1024
    max_recovery_steps = 10

class Nav2dForceRandomHolesCeres5N(Nav2dForceRandomHolesCeres):
    max_normalized_obs = 5.
    max_normalized_act = 5.
