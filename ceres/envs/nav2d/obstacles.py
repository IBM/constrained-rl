# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np

class Obstacle(object):
    '''
    Base class for obstacles, with export/import functions
    '''
    required_parameters = []
    def __init__(self, **kwargs):
        for _k in self.required_parameters:
            setattr(self, _k, kwargs[_k])
        self.check_parameters()

    def to_array(self):
        params = [getattr(self, _k) for _k in self.required_parameters]
        return params

    @classmethod
    def FromArray(cls, params):
        assert len(params) == len(cls.required_parameters)
        params_as_dict = {_k: _v for _k, _v in zip(cls.required_parameters, params)}
        obstacle = cls(**params_as_dict)
        return obstacle

class ObstacleSquare(Obstacle):
    '''
    Square obstacle, initialized from the 2D location of its top-left and bottom-right corners
    '''
    required_parameters = ['top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']

    def check_parameters(self):
        assert self.top_left_x < self.bottom_right_x
        assert self.top_left_y > self.bottom_right_y
        self.bottom_left_x = self.top_left_x
        self.bottom_left_y = self.bottom_right_y
        self.top_right_x = self.bottom_right_x
        self.top_right_y = self.top_left_y

    def test_collision(self, x, y, conservative=False, min_distance=0.):
        # Set conservative=True to count border as collision
        x_proj, y_proj, is_strictly_inside = self.project(x, y)
        if is_strictly_inside:
            return True
        else:
            dist = np.linalg.norm(np.array([x, y]) - np.array([x_proj, y_proj]))
            if conservative:
                is_collision = dist <= min_distance
            else:
                is_collision = dist < min_distance
        return is_collision

    def to_polygon(self):
        path_closed = []
        path_closed.append((self.top_left_x, self.top_left_y))
        path_closed.append((self.bottom_left_x, self.bottom_left_y))
        path_closed.append((self.bottom_right_x, self.bottom_right_y))
        path_closed.append((self.top_right_x, self.top_right_y))
        return path_closed

    def project(self, x, y):
        strict_inside_x = False
        strict_inside_y = False
        if x >= self.bottom_right_x:
            x_proj = self.bottom_right_x
        elif x <= self.bottom_left_x:
            x_proj = self.bottom_left_x
        else:
            x_proj = x
            strict_inside_x = True
        if y >= self.top_left_y:
            y_proj = self.top_left_y
        elif y <= self.bottom_left_y:
            y_proj = self.bottom_left_y
        else:
            y_proj = y
            strict_inside_y = True
        is_strictly_inside = strict_inside_x and strict_inside_y
        return x_proj, y_proj, is_strictly_inside

    def intersection_with_line(self, p1, p2):
        raise NotImplementedError('Intersection between square and line not implemented')


class ObstacleCircle(Obstacle):
    '''
    Circle obstacle, initialized from the 2D location of its center and its radius
    '''
    required_parameters = ['center_x', 'center_y', 'radius']

    def check_parameters(self):
        assert self.radius > 0.
        self.center_xy = np.array([self.center_x, self.center_y])
        self.intersection_line_shift = np.dot(self.center_xy, self.center_xy) - self.radius**2 # use this when computing intersection with line

    def test_collision(self, x, y, conservative=False, min_distance=0.):
        # Set conservative=True to count border as collision
        x_proj, y_proj, is_strictly_inside = self.project(x, y)
        if is_strictly_inside:
            return True
        else:
            dist = np.linalg.norm(np.array([x, y]) - np.array([x_proj, y_proj]))
            if conservative:
                is_collision = dist <= min_distance
            else:
                is_collision = dist < min_distance
        return is_collision

    def to_polygon(self):
        raise NotImplementedError('Use circle drawing function')

    def project(self, x, y):
        center_to_point = np.array([x - self.center_x, y - self.center_y])
        dist_from_center = np.linalg.norm(center_to_point)
        is_strictly_inside = dist_from_center < self.radius
        if is_strictly_inside:
            x_proj, y_proj = x, y
        else:
            center_to_surface = center_to_point / dist_from_center * self.radius
            x_proj = self.center_x + center_to_surface[0]
            y_proj = self.center_y + center_to_surface[1]
        return x_proj, y_proj, is_strictly_inside

    def intersection_with_line(self, p1, p2):
        '''
        Solve quadratic equation a x^2 + b x + c = 0
        with a = np.dot(v, v) with v unit vector between p1 and p2,
        b = 2 np.dot(v, p1 - center)
        c = np.dot(p1, p1) + np.dot(center, center) - 2 np.dot(p1, center) - radius^2
        '''
        p1 = np.array(p1)
        p2 = np.array(p2)
        unit_vec = p2 - p1
        dist = np.linalg.norm(unit_vec)
        assert dist > 0.
        unit_vec /= dist
        a = np.dot(unit_vec, unit_vec)
        b = 2. * np.dot(unit_vec, p1 - self.center_xy)
        c = np.dot(p1, p1) - 2. * np.dot(p1, self.center_xy) + self.intersection_line_shift

        delta = b**2 - 4.* a * c
        if delta < 0:
            return False, None
        delta_sqrt = np.sqrt(delta)
        # Two solutions: x1, x2
        x1 = (-b - delta_sqrt) / (2. * a)
        x2 = (-b + delta_sqrt) / (2. * a)
        if 0. <= x1 <= dist:
            x_min = x1
        elif 0. <= x2 <= dist:
            x_min = x2
        else:
            #x_min = min(x1, x2)
            return False, None
        closest = p1 + x_min * unit_vec
        return True, closest
     
