# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Constrained Exploration and Recovery from Experience Shaping
# This project is licensed under the MIT License, see LICENSE

import numpy as np
import tensorflow as tf

class SphericalCoordinates(object):
    '''
    Implement N-dimensional coordinates in Numpy and Tensorflow
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    '''

    def __init__(self, dim, input_angles=None):
        assert dim >= 2
        self.dim = dim
        self.n_angles = dim - 1
        if input_angles is None:
            self.input_angles = tf.placeholder(tf.float32, shape=(None, self.n_angles))
        else:
            self.input_angles = input_angles
        self.init_angles_to_unit_vec()
    
    def spherical_to_cartesian(self, angles, radius=1):
        assert angles.shape[-1] == self.n_angles
        shape_angles_in = angles.shape
        angles_batch = np.reshape(angles, [-1, self.n_angles])
        vec_batch = []
        for angles in angles_batch:
            previous = radius
            vec = []
            for angle in angles[:-1]:
                coord = previous*np.cos(angle)
                vec.append(coord)
                previous *= np.sin(angle)
            angle = angles[-1]
            vec.append(previous*np.cos(angle))
            vec.append(previous*np.sin(angle))
            vec_batch.append(vec)
        shape_vec_out = list(shape_angles_in)
        shape_vec_out[-1] += 1
        vec_batch = np.array(vec_batch)
        vec_batch = np.reshape(vec_batch, shape_vec_out)
        return vec_batch

    def init_angles_to_unit_vec(self, radius=None):
        angles = self.input_angles
        shape_angles_in = tf.shape(angles)
        angles = tf.reshape(angles, [-1, shape_angles_in[-1]])
        angles_cos = tf.cos(angles)
        angles_sin = tf.sin(angles)
        vec = []
        previous = 1.
        for i_angle in range(self.n_angles-1):
            axis_cos = tf.slice(angles_cos, [0, i_angle], [-1, 1])
            axis_sin = tf.slice(angles_sin, [0, i_angle], [-1, 1])
            coord = tf.multiply(previous, axis_cos)
            previous  = tf.multiply(previous, axis_sin)
            vec.append(coord)
        i_angle = self.n_angles-1
        axis_cos = tf.slice(angles_cos, [0, i_angle], [-1, 1])
        axis_sin = tf.slice(angles_sin, [0, i_angle], [-1, 1])
        vec.append(tf.multiply(previous, axis_cos))
        vec.append(tf.multiply(previous, axis_sin))
        vec = tf.concat(vec, axis=1)
        if radius is not None:
            radius = tf.reshape(radius, [-1, shape_angles_in[-1]])
            vec = tf.multiply(radius, vec)
        shape_vec_last = tf.constant([self.dim], dtype=shape_angles_in.dtype)
        shape_vec_out = tf.concat([shape_angles_in[:-1], shape_vec_last], axis=0)
        vec = tf.reshape(vec, shape_vec_out)
        self.output_unit_vec = vec

def main():
    while True:
        input_str = input('Input list of angles, in degrees, comma-separated\n')
        if len(input_str) == 0:
            #angles_deg = np.array([[0.0], [90.]])
            angles_deg = np.array([
                                   [[0.],
                                    [45.]],
                                   [[90.],
                                    [135.]]
                                  ])
            print('Use default example: {0}'.format(angles_deg))
        else:
            angles_deg = np.array([float(e) for e in input_str.split(',')])
        n_angles = angles_deg.shape[-1]
        dim = n_angles + 1
        angles_rad = np.radians(angles_deg)
        print('Degrees: {0}'.format(angles_deg))
        print('Radians: {0}'.format(angles_rad))
        sc = SphericalCoordinates(dim)
        vec_np = sc.spherical_to_cartesian(angles_rad)
        print('Unit vector')
        print('  Numpy: {0}'.format(vec_np))
        with tf.Session() as sess:
            angles_rad_reshaped = np.reshape(angles_rad, [-1, n_angles])
            vec_tf = sess.run([sc.output_unit_vec], feed_dict={sc.input_angles: angles_rad_reshaped})
            print('  Tensorflow: {0}'.format(vec_tf))

if __name__ == '__main__':
    main()
