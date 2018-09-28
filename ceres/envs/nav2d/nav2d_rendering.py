# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import pygame
import numpy as np
from enum import Enum
from .nav2d_pos import Nav2dPos
from .obstacles import ObstacleSquare, ObstacleCircle

class RenderingState(Enum):
    disable, continuous, step, episode = range(4)

class Nav2dRendering(object):
    '''
    Implement rendering functions for Nav2d environments,
    with or without obstacles, with or without constraints
    '''

    rendering_window_name = 'Nav2d'
    rendering_width  = 800
    rendering_height = 800
    rendering_wait   = 1

    rendering_background_rgb = (0, 0, 0)
    rendering_background_transparent_rgba = (0, 0, 0, 0)
    rendering_agent_rgb  = (0, 0, 255)
    rendering_agent_range_rgba  = (255, 0, 0, 155)
    rendering_target_rgb = (0, 255, 0)
    rendering_agent_path_rgb = (255, 255, 255)
    rendering_constraint_line_rgb = (25, 0, 0)
    rendering_constraint_line_width = 1
    rendering_constraint_polygon_rgba = (50, 50, 50, 128)
    rendering_constraint_polygon_width = 0 # set to zero to fill
    rendering_border_polygon_rgb = (155, 155, 155, 0)
    rendering_border_polygon_width = 0 # set to zero to fill
    rendering_obstacle_polygon_rgb = (0, 0, 0)
    rendering_obstacle_polygon_width = 0 # set to zero to fill

    rendering_info_text_size  = (30)
    rendering_info_text_rgb  = (255, 255, 255)
    rendering_info_text_pos  = (10, 10)
    rendering_cmd_text_size  = (30)
    rendering_cmd_text_rgb  = (255, 255, 255)
    rendering_cmd_text_pos  = (10, 890)
    rendering_cmd_text_str = 'Commands: (n) next step, (e) next episode, (c) continuous, (ESC) disable, (h) hide help'

    def __init__(self):
        self.constraint_lines = []
        self.constraint_polygons = []
        self.init_border()
        self.rendering_state = RenderingState.continuous
        super(Nav2dRendering, self).__init__()

    def init_border(self):
        self.border_width_x = 1.5 * self.max_step_x
        self.border_width_y = 1.5 * self.max_step_y

        self.min_x_display = self.min_x - self.border_width_x
        self.max_x_display = self.max_x + self.border_width_x
        self.min_y_display = self.min_y - self.border_width_y
        self.max_y_display = self.max_y + self.border_width_y

        self.corner_pos_top_left     = (self.min_x_display, self.max_y_display)
        self.corner_pos_top_right    = (self.max_x_display, self.max_y_display)
        self.corner_pos_bottom_left  = (self.min_x_display, self.min_y_display)
        self.corner_pos_bottom_right = (self.max_x_display, self.min_y_display)

        self.world_width_with_border = self.world_width + 2*self.border_width_x
        self.world_height_with_border = self.world_height + 2*self.border_width_y
        self.rendering_width_with_border = int(self.rendering_width/self.world_width*self.world_width_with_border)
        self.rendering_height_with_border = int(self.rendering_height/self.world_height*self.world_height_with_border)

        self.border_polygon = []
        self.border_polygon.append(self.convert_world_to_pixel(self.min_x, self.min_y))
        self.border_polygon.append(self.convert_world_to_pixel(self.min_x, self.max_y))
        self.border_polygon.append(self.convert_world_to_pixel(self.max_x, self.max_y))
        self.border_polygon.append(self.convert_world_to_pixel(self.max_x, self.min_y))

    def init_rendering(self):
        pygame.init()
        pygame.font.init()
        self.rendering_window_size = (self.rendering_width_with_border, self.rendering_height_with_border)
        self.rendering_surface_constraints = pygame.Surface(self.rendering_window_size, pygame.SRCALPHA)
        self.rendering_surface_agent_range = self.rendering_surface_constraints
        self.rendering_display = pygame.display.set_mode(self.rendering_window_size)
        self.rendering_agent_radius = int(self.agent_radius * self.rendering_width / self.world_width)
        self.rendering_agent_range_radius = int(self.rendering_width*self.max_step_x/self.world_width)
        self.rendering_target_radius = int(self.target_radius * self.rendering_width / self.world_width)
        self.reset_image()
        self.agent_path_closed = False
        self.agent_path_width = 1
        self.is_init_rendering = True
        self.info_font = pygame.font.SysFont('', self.rendering_info_text_size)
        self.cmd_font = pygame.font.SysFont('', self.rendering_cmd_text_size)
        self.do_display_cmd_help = True

    def redraw(self):
        self.draw_obstacles()
        self.draw_lidar()
        self.draw_agent_range()
        self.draw_constraints()
        self.draw_agent_path()
        self.draw_agent_target()
        self.draw_info()

    def draw_obstacles(self):
        if not hasattr(self, 'obstacles'):
            return
        for o in self.obstacles:
            if isinstance(o, ObstacleSquare):
                polygon_lines_world = o.to_polygon()
                polygon_lines_pixel = [self.convert_world_to_pixel(*l) for l in polygon_lines_world]
                pygame.draw.polygon(self.rendering_display,
                                    self.rendering_obstacle_polygon_rgb,
                                    polygon_lines_pixel,
                                    self.rendering_obstacle_polygon_width)
            elif isinstance(o, ObstacleCircle):
                center_pixel = self.convert_world_to_pixel(o.center_x, o.center_y)
                radius_pixel = self.convert_world_length_to_pixel_length(o.radius)
                pygame.draw.circle(self.rendering_display,
                                   self.rendering_obstacle_polygon_rgb,
                                   center_pixel,
                                   radius_pixel,
                                   self.rendering_obstacle_polygon_width)
            else:
                raise ValueError('Invalid obstacle {0}'.format(type(o)))

    def draw_lidar(self):
        if not hasattr(self, 'lidar_segments'):
            return
        for segment in self.lidar_segments:
            p1, p2, dist = segment
            if dist == 0.:
                continue
            p1_pixel = self.convert_world_to_pixel(*p1)
            p2_pixel = self.convert_world_to_pixel(*p2)
            pygame.draw.line(self.rendering_display,
                             (255, 255, 0),
                             p1_pixel,
                             p2_pixel,
                             1)

    def render(self, mode='human', close=False):
        if close or (self.rendering_state == RenderingState.disable):
            if self.is_init_rendering:
                pygame.display.quit()
            return
        elif not self.is_init_rendering:
            self.init_rendering()
        self.reset_image()
        self.redraw()
        pygame.display.update()
        self.user_interaction()

    def save_render(self, path_save, verbose=False):
        pygame.image.save(self.rendering_display, path_save)
        if verbose:
            print('Saved window: {0}'.format(path_save))

    def user_interaction(self):
        if self.rendering_state == RenderingState.continuous:
            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.KEYDOWN:
                    if (event.key == pygame.K_n):
                        self.rendering_state = RenderingState.step
                    elif (event.key == pygame.K_e):
                        self.rendering_state = RenderingState.episode
                    elif (event.key == pygame.K_h):
                        self.do_display_cmd_help = not self.do_display_cmd_help
                    elif (event.key == pygame.K_ESCAPE):
                        self.rendering_state = RenderingState.disable
        elif self.rendering_state == RenderingState.step:
            while True:
                event = pygame.event.wait()
                if event.type == pygame.KEYDOWN:
                    if (event.key == pygame.K_n) or (event.key == pygame.K_RETURN):
                        break
                    elif (event.key == pygame.K_c):
                        self.rendering_state = RenderingState.continuous
                        break
                    elif (event.key == pygame.K_e):
                        self.rendering_state = RenderingState.episode
                        break
                    elif (event.key == pygame.K_h):
                        self.do_display_cmd_help = not self.do_display_cmd_help
                        break
                    elif (event.key == pygame.K_ESCAPE):
                        self.rendering_state = RenderingState.disable
                        break
        elif self.rendering_state == RenderingState.episode:
            done = self.done_base or (self.i_step == self.max_episode_steps)
            if done:
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN:
                        if (event.key == pygame.K_e) or (event.key == pygame.K_RETURN):
                            break
                        elif (event.key == pygame.K_c):
                            self.rendering_state = RenderingState.continuous
                            break
                        elif (event.key == pygame.K_n):
                            self.rendering_state = RenderingState.step
                            break
                        elif (event.key == pygame.K_h):
                            self.do_display_cmd_help = not self.do_display_cmd_help
                            break
                        elif (event.key == pygame.K_ESCAPE):
                            self.rendering_state = RenderingState.disable
                            break

    def reset_image(self):
        self.rendering_display.fill(self.rendering_background_rgb)
        self.rendering_surface_constraints.fill(self.rendering_background_transparent_rgba)
        if self.rendering_surface_agent_range != self.rendering_surface_constraints:
            self.rendering_surface_agent_range.fill(self.rendering_background_transparent_rgba)
        self.draw_border()

    def draw_border(self):
        pygame.draw.polygon(self.rendering_display,
                            self.rendering_border_polygon_rgb,
                            self.border_polygon,
                            self.rendering_border_polygon_width)

    def draw_agent_range(self):
        # Draw agent range
        pygame.draw.circle(self.rendering_surface_agent_range,
                           self.rendering_agent_range_rgba,
                           (self.agent_pixel_col, self.agent_pixel_row),
                           self.rendering_agent_range_radius,
        )
        if self.rendering_surface_agent_range != self.rendering_surface_constraints:
            self.rendering_display.blit(self.rendering_surface_agent_range, (0, 0))

    def draw_agent_target(self):
        # Draw agent
        pygame.draw.circle(self.rendering_display,
                           self.rendering_agent_rgb,
                           (self.agent_pixel_col, self.agent_pixel_row),
                           self.rendering_agent_radius,
        )
        # Draw target
        pygame.draw.circle(self.rendering_display,
                           self.rendering_target_rgb,
                           (self.target_pixel_col, self.target_pixel_row),
                           self.rendering_target_radius,
        )

    def draw_info(self):
        if self.do_display_cmd_help:
            info_str = 'Avg. reward: {0:.2f}'.format(self.ep_reward_avg)
            info_surface = self.info_font.render(info_str, False, self.rendering_info_text_rgb, self.rendering_background_rgb)
            self.rendering_display.blit(info_surface, self.rendering_info_text_pos)
            cmd_surface = self.cmd_font.render(self.rendering_cmd_text_str, False, self.rendering_cmd_text_rgb, self.rendering_background_rgb)
            self.rendering_display.blit(cmd_surface, self.rendering_cmd_text_pos)

    def draw_agent_path(self):
        if len(self.agent_pixel_seq) >= 2:
            pygame.draw.lines(self.rendering_display,
                              self.rendering_agent_path_rgb,
                              self.agent_path_closed,
                              self.agent_pixel_seq,
                              self.agent_path_width,
            )

    def draw_constraints(self):
        self.update_constraint_rendering()
        self.draw_constraint_polygons()
        self.draw_constraint_lines()

    def draw_constraint_polygons(self):
        for i_constraint, polygon in enumerate(self.constraint_polygons):
            if len(polygon) < 3:
                continue
            pygame.draw.polygon(self.rendering_surface_constraints,
                                self.rendering_constraint_polygon_rgba,
                                polygon,
                                self.rendering_constraint_polygon_width)
        self.rendering_display.blit(self.rendering_surface_constraints, (0, 0))

    def draw_constraint_lines(self):
        for i_constraint, (pixel_start, pixel_end) in enumerate(self.constraint_lines):
            pygame.draw.line(self.rendering_display,
                             self.rendering_constraint_line_rgb,
                             pixel_start,
                             pixel_end,
                             self.rendering_constraint_line_width,
            )

    def update_env(self):
        self.update_pixel_agent()

    def convert_world_length_to_pixel_length(self, x):
        x_pixel = int(self.rendering_width_with_border*x/self.world_width_with_border)
        return x_pixel

    def convert_world_to_pixel(self, x, y):
        col = int(self.rendering_width_with_border*(0.5 + x / self.world_width_with_border))
        row = int(self.rendering_height_with_border*(0.5 - y / self.world_height_with_border))
        return col, row

    def convert_pixel_to_world(self, col, row):
        x = self.world_width_with_border * (col/self.rendering_width_with_border - 0.5)
        y = -self.world_height_with_border * (row/self.rendering_height_with_border - 0.5)
        return x, y

    def update_pixel_agent(self):
        self.agent_pixel_col, self.agent_pixel_row = self.convert_world_to_pixel(self.agent_pos_x, self.agent_pos_y)

    def update_pixel_target(self):
        self.target_pixel_col, self.target_pixel_row = self.convert_world_to_pixel(self.target_pos_x, self.target_pos_y)

    def reset_task_specific(self):
        self.agent_pixel_seq = []
        self.update_pixel_agent()
        self.update_pixel_target()
        self.agent_pixel_seq = [(self.agent_pixel_col, self.agent_pixel_row)]

    def _step_task_specific(self):
        self.store_agent_pixel()

    def store_agent_pixel(self):
        self.agent_pixel_seq.append((self.agent_pixel_col, self.agent_pixel_row))

    def update_constraint_rendering(self):
        if not self.is_init_rendering:
            return
        self.constraint_lines = []
        self.constraint_polygons = []
        if (not hasattr(self, 'ineq_mat')) or (not hasattr(self, 'ineq_vec')):
            return
        ineq_mat = self.ineq_mat
        ineq_vec = self.ineq_vec
        if (ineq_mat is None) or (ineq_vec is None):
            assert (ineq_mat is None) and (ineq_vec is None)
            return
        if (len(ineq_mat) == 0) or (len(ineq_vec) == 0):
            assert (len(ineq_mat) == 0) and (len(ineq_vec) == 0)
            return
        assert (ineq_mat.ndim == 2) and (ineq_vec.ndim == 2), 'Dimension mismatch: got shapes {0} and {1}'.format(ineq_mat.shape, ineq_vec.shape)
        # ineq_mat X <= ineq_vec
        # X: normalized action
        # Y: real action (dx, dy)
        # X = self.normalize_act_scale * Y + self.normalize_act_shift
        # ineq_mat * self.normalize_act_scale * Y <= ineq_vec - ineq_mat * self.normalize_act_shift
        ineq_mat = np.dot(ineq_mat, self.normalize_act_scale)
        ineq_vec = ineq_vec - np.dot(ineq_mat, self.normalize_act_shift)
        for i_constraint, (A, B) in enumerate(zip(ineq_mat, ineq_vec)):
            # A X <= B, alpha_x * x + alpha_y * y <= b
            alpha_x = A[0]
            alpha_y = A[1]
            b = B[0]
            if abs(alpha_y) < self.alpha_xy_epsilon:
                if abs(alpha_x) < self.alpha_xy_epsilon: 
                    continue
                action_x = b / alpha_x
                line_start_pos_x = self.agent_pos_x + action_x
                line_end_pos_x   = self.agent_pos_x + action_x
                line_start_pos_y = self.min_y_display
                line_end_pos_y   = self.max_y_display
                line_start_pos = (line_start_pos_x, line_start_pos_y)
                line_end_pos   = (line_end_pos_x, line_end_pos_y)
                if alpha_x > 0: # left of line: forbidden right
                    constraint_polygon = [line_end_pos, line_start_pos, self.corner_pos_bottom_right, self.corner_pos_top_right]
                else: # valid right: forbidden left
                    constraint_polygon = [line_start_pos, line_end_pos, self.corner_pos_top_left, self.corner_pos_bottom_left]
            else:
                action_x_left = self.min_x_display - self.agent_pos_x
                action_x_right = self.max_x_display - self.agent_pos_x
                action_y_left, action_y_right = map(lambda x: (b - alpha_x*x)/alpha_y, [action_x_left, action_x_right])
                line_start_pos_x = self.min_x_display
                line_end_pos_x   = self.max_x_display
                line_start_pos_y = self.agent_pos_y + action_y_left
                line_end_pos_y   = self.agent_pos_y + action_y_right
                line_start_pos = (line_start_pos_x, line_start_pos_y)
                line_end_pos   = (line_end_pos_x, line_end_pos_y)

                # Check that we picked the right display range
                # y = a_line * x + b_line
                a_line = (line_end_pos_y - line_start_pos_y)/(line_end_pos_x - line_start_pos_x)
                b_line = line_end_pos_y - a_line * line_end_pos_x
                y_line_to_x = lambda y: (y - b_line)/a_line
                if (line_start_pos_y < self.min_y_display) or (line_start_pos_y > self.max_y_display):
                    if line_start_pos_y < self.min_y_display:
                        line_start_pos_y_restricted = self.min_y_display
                    else:
                        line_start_pos_y_restricted = self.max_y_display
                    line_start_pos_x_restricted = y_line_to_x(line_start_pos_y_restricted)
                    line_start_pos_restricted = (line_start_pos_x_restricted, line_start_pos_y_restricted)
                else:
                    line_start_pos_restricted = line_start_pos

                if (line_end_pos_y < self.min_y_display) or (line_end_pos_y > self.max_y_display):
                    if line_end_pos_y < self.min_y_display:
                        line_end_pos_y_restricted = self.min_y_display
                    else:
                        line_end_pos_y_restricted = self.max_y_display
                    line_end_pos_x_restricted = y_line_to_x(line_end_pos_y_restricted)
                    line_end_pos_restricted = (line_end_pos_x_restricted, line_end_pos_y_restricted)
                else:
                    line_end_pos_restricted = line_end_pos

                if alpha_y > 0: # valid below line: forbidden above
                    constraint_polygon = [line_start_pos_restricted, line_end_pos_restricted]
                    for p in [self.corner_pos_bottom_right, self.corner_pos_top_right]:
                        if p[1] > line_end_pos[1]:
                            constraint_polygon.append(p)
                    for p in [self.corner_pos_top_left, self.corner_pos_bottom_left]:
                        if p[1] > line_start_pos[1]:
                            constraint_polygon.append(p)
                else: # valid above line: forbidden below
                    constraint_polygon = [line_end_pos_restricted, line_start_pos_restricted]
                    for p in [self.corner_pos_top_left, self.corner_pos_bottom_left]:
                        if p[1] < line_start_pos[1]:
                            constraint_polygon.append(p)
                    for p in [self.corner_pos_bottom_right, self.corner_pos_top_right]:
                        if p[1] < line_end_pos[1]:
                            constraint_polygon.append(p)

            line_start_pixel_col, line_start_pixel_row = self.convert_world_to_pixel(line_start_pos_x, line_start_pos_y)
            line_end_pixel_col, line_end_pixel_row = self.convert_world_to_pixel(line_end_pos_x, line_end_pos_y)
            pixel_start = (line_start_pixel_col, line_start_pixel_row)
            pixel_end = (line_end_pixel_col, line_end_pixel_row)
            self.constraint_lines.append((pixel_start, pixel_end))
            constraint_polygon = [self.convert_world_to_pixel(*_p) for _p in constraint_polygon]
            self.constraint_polygons.append(constraint_polygon)

