# Copyright 2018 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np
import os
import random
import scipy as sp
from scipy import ndimage


## Input pipeline
def get_filenames(num_test_imgs_per_dataset, folders):

  filenames = []
  test_filenames = []

  for dataset_dir in folders:

    image_names = [
        f for f in sorted(os.listdir(dataset_dir)) if f.endswith('png')
    ]

    # split off test set
    random.shuffle(image_names)
    test_image_names = image_names[:num_test_imgs_per_dataset]
    image_names = image_names[num_test_imgs_per_dataset:]

    filenames += [os.path.join(dataset_dir, f) for f in image_names]

    test_filenames += [os.path.join(dataset_dir, f) for f in test_image_names]

  return filenames, test_filenames


def process_lf(lf, num_crops, mask, lfsize, patchsize):
  lf = normalize(
      tf.image.adjust_gamma(
          tf.to_float(lf[:lfsize[0] * 14, :lfsize[1] * 14, :]) / 255.0,
          gamma=0.4))
  lf = tf.transpose(
      tf.reshape(lf, [lfsize[0], 14, lfsize[1], 14, 3]), [0, 2, 1, 3, 4])
  lf = lf[:, :, (14 / 2) - (lfsize[2] / 2):(14 / 2) + (lfsize[2] / 2),
          (14 / 2) - (lfsize[3] / 2):(14 / 2) + (lfsize[3] / 2), :]
  aif = lf[:, :, lfsize[2] / 2, lfsize[3] / 2, :]
  shear0 = tf.constant(0.0)
  lf_shear_r0 = tf.expand_dims(lf[:, :, :, :, 0], 0)
  lf_shear_g0 = tf.expand_dims(lf[:, :, :, :, 1], 0)
  lf_shear_b0 = tf.expand_dims(lf[:, :, :, :, 2], 0)
  lf_shear0 = tf.stack([lf_shear_r0, lf_shear_g0, lf_shear_b0], axis=5)
  bokeh0 = tf.squeeze(project_lf(lf_shear0, mask))
  shear1 = tf.constant(1.0)
  lf_shear_r1 = shear_lf(tf.expand_dims(lf[:, :, :, :, 0], 0), shear1)
  lf_shear_g1 = shear_lf(tf.expand_dims(lf[:, :, :, :, 1], 0), shear1)
  lf_shear_b1 = shear_lf(tf.expand_dims(lf[:, :, :, :, 2], 0), shear1)
  lf_shear1 = tf.stack([lf_shear_r1, lf_shear_g1, lf_shear_b1], axis=5)
  bokeh1 = tf.squeeze(project_lf(lf_shear1, mask))
  aif_list = []
  bokeh_list0 = []
  bokeh_list1 = []
  aif_bokeh_stack = tf.stack([aif, bokeh0, bokeh1], 3)
  for i in range(num_crops):
    aif_bokeh_crop = tf.random_crop(aif_bokeh_stack,
                                    [patchsize[0], patchsize[1], 3, 3])
    aif_list.append(aif_bokeh_crop[:, :, :, 0])
    bokeh_list0.append(aif_bokeh_crop[:, :, :, 1])
    bokeh_list1.append(aif_bokeh_crop[:, :, :, 2])
  return aif_list, bokeh_list0, bokeh_list1


def read_lf(filename_queue, num_crops, mask, lfsize, patchsize):
  value = tf.read_file(filename_queue[0])
  lf = tf.image.decode_image(value, channels=3)
  aif_list, bokeh_list0, bokeh_list1 = process_lf(lf, num_crops, mask, lfsize,
                                                  patchsize)
  return aif_list, bokeh_list0, bokeh_list1


def input_pipeline(filenames, mask, lfsize, patchsize, num_crops, batch_size):
  filename_queue = tf.train.slice_input_producer([filenames], shuffle=True)
  example_list = [
      read_lf(filename_queue, num_crops, mask, lfsize, patchsize)
      for _ in range(3)
  ]
  min_after_dequeue = 0
  capacity = 12
  aif_batch, bokeh_batch0, bokeh_batch1 = tf.train.shuffle_batch_join(
      example_list,
      batch_size=batch_size,
      capacity=capacity,
      min_after_dequeue=min_after_dequeue,
      enqueue_many=True,
      shapes=[[patchsize[0], patchsize[1], 3], [patchsize[0], patchsize[1], 3],
              [patchsize[0], patchsize[1], 3]])
  return aif_batch, bokeh_batch0, bokeh_batch1


def get_eval_filenames(folders):
  filenames = []
  for dataset_dir in folders:
    image_names = [
        f for f in sorted(os.listdir(dataset_dir)) if f.endswith('png')
    ]
    filenames += [os.path.join(dataset_dir, f) for f in image_names]

  return filenames


def eval_pipeline(filenames, mask, lfsize, patchsize, num_crops, batch_size):
  filename_queue = tf.train.slice_input_producer([filenames], shuffle=False)
  example_list = [
      read_lf(filename_queue, num_crops, mask, lfsize, patchsize)
      for _ in range(1)
  ]
  min_after_dequeue = 0
  capacity = 8
  aif_batch, bokeh_batch0, bokeh_batch1 = tf.train.batch_join(
      example_list,
      batch_size=batch_size,
      capacity=capacity,
      enqueue_many=True,
      shapes=[[patchsize[0], patchsize[1], 3], [patchsize[0], patchsize[1], 3],
              [patchsize[0], patchsize[1], 3]])

  return aif_batch, bokeh_batch0, bokeh_batch1


## CNN layers


def weight_var(shape):
  return tf.get_variable(
      'weights',
      shape,
      initializer=tf.contrib.layers.xavier_initializer_conv2d())


def bias_var(shape, init_bias):
  return tf.get_variable(
      'bias', shape, initializer=tf.constant_initializer(init_bias))


def cnn_layer(input_tensor,
              w_shape,
              b_shape,
              name,
              downsample=1,
              padding_type='SAME',
              init_bias=0.0):
  with tf.variable_scope(name):
    W = weight_var(w_shape)
    return tf.nn.conv2d(
        input_tensor,
        W,
        strides=[1, downsample, downsample, 1],
        padding=padding_type,
        name=name + 'conv') + bias_var(b_shape, init_bias)


def d_cnn_layer(input_tensor,
                w_shape,
                b_shape,
                name,
                rate=1,
                padding_type='SAME',
                init_bias=0.0):
  with tf.variable_scope(name):
    W = weight_var(w_shape)
    return tf.nn.atrous_conv2d(
        input_tensor, W, rate, padding=padding_type,
        name=name + 'conv') + bias_var(b_shape, init_bias)


def bn(input_tensor, training_bool, name='instance_normalization'):
  with tf.variable_scope(name):
    depth = input_tensor.get_shape()[3]
    scale = tf.get_variable(
        'scale', [depth],
        initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = tf.get_variable(
        'offset', [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input_tensor, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input_tensor - mean) * inv
    return scale * normalized + offset


# L1 loss
def l1_loss(x, y):
  return tf.reduce_mean(tf.abs(x - y))


## Utils


# Normalize between -1 and 1, assuming input is between 0 and 1
def normalize(x):
  return x * 2.0 - 1.0


# Denormalize between 0 and 1, assuming input is between -1 and 1
def denormalize(x):
  return x * 0.5 + 0.5


# Compute spatial derivatives in x and y directions
def img_derivs(img):
  y_sz = tf.shape(img)[1]
  x_sz = tf.shape(img)[2]
  temp = img[:, 0:y_sz - 1, 0:x_sz - 1, :]
  dy = img[:, 1:y_sz, 0:x_sz - 1, :] - temp
  dx = img[:, 0:y_sz - 1, 1:x_sz, :] - temp
  return dy, dx


## Forward pass functions


# network to compute disparity from input (lf model)
def disp_net_lf(center_ds, is_training, disp_mult, name):
  with tf.variable_scope(name):

    c1 = bn(
        tf.nn.elu(cnn_layer(center_ds, [3, 3, 3, 8], [8], 'c1')), is_training,
        'n1')
    c2 = bn(
        tf.nn.elu(cnn_layer(c1, [3, 3, 8, 32], [32], 'c2')), is_training, 'n2')
    c3 = bn(
        tf.nn.elu(cnn_layer(c2, [3, 3, 32, 64], [64], 'c3')), is_training, 'n3')
    c4 = bn(
        tf.nn.elu(cnn_layer(c3, [3, 3, 64, 128], [128], 'c4')), is_training,
        'n4')

    c5 = bn(
        tf.nn.elu(d_cnn_layer(c4, [3, 3, 128, 128], [128], 'c5', rate=2)),
        is_training, 'n5') + c4
    c6 = bn(
        tf.nn.elu(d_cnn_layer(c5, [3, 3, 128, 128], [128], 'c6', rate=4)),
        is_training, 'n6') + c5
    c7 = bn(
        tf.nn.elu(d_cnn_layer(c6, [3, 3, 128, 128], [128], 'c7', rate=8)),
        is_training, 'n7') + c6
    c8 = bn(
        tf.nn.elu(d_cnn_layer(c7, [3, 3, 128, 128], [128], 'c8', rate=16)),
        is_training, 'n8') + c7
    c9 = bn(
        tf.nn.elu(d_cnn_layer(c8, [3, 3, 128, 128], [128], 'c9', rate=32)),
        is_training, 'n9') + c8

    c10 = bn(
        tf.nn.elu(cnn_layer(c9, [3, 3, 128, 64], [64], 'c10')), is_training,
        'n10')
    c11 = bn(
        tf.nn.elu(cnn_layer(c10, [3, 3, 64, 32], [32], 'c11')), is_training,
        'n11')
    c12 = cnn_layer(c11, [3, 3, 32, 1], [1], 'c12')

    # This is where the bilateral solver should go, but this code was difficult to open-source.

    d = disp_mult * tf.tanh(c12)
    ds = d

    return d, ds


# network to refine 4D disparities
def disp_refine_net(x, u_sz, is_training, name):
  with tf.variable_scope(name):
    b_sz = tf.shape(x)[0]
    y_sz = tf.shape(x)[1]
    x_sz = tf.shape(x)[2]
    v_sz = u_sz
    nc = v_sz * u_sz
    x = tf.reshape(x, [b_sz, y_sz, x_sz, nc])
    c1 = bn(
        tf.nn.elu(cnn_layer(x, [3, 3, nc, nc], [nc], 'c1')), is_training, 'n1')
    c2 = bn(
        tf.nn.elu(cnn_layer(c1, [3, 3, nc, nc], [nc], 'c2')), is_training, 'n2')
    c3 = cnn_layer(c2, [3, 3, nc, nc], [nc], 'c3') + x
    return tf.reshape(c3, [b_sz, y_sz, x_sz, v_sz, u_sz])


# shear light field rays to refocus
def shear_lf(lf, shear):
  with tf.name_scope('render_lf'):
    b_sz = tf.shape(lf)[0]
    y_sz = tf.shape(lf)[1]
    x_sz = tf.shape(lf)[2]
    u_sz = tf.shape(lf)[3]
    v_sz = u_sz

    # create and transform light field grid
    b_vals = tf.to_float(tf.range(b_sz))
    y_vals = tf.to_float(tf.range(y_sz))
    x_vals = tf.to_float(tf.range(x_sz))
    v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz) / 2.0
    u_vals = tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz) / 2.0

    b, y, x, v, u = tf.meshgrid(
        b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

    # warp coordinates by disparities
    y_t = y + v * shear
    x_t = x + u * shear

    v_r = v + tf.to_float(v_sz) / 2.0
    u_r = u + tf.to_float(u_sz) / 2.0

    # indices for linear interpolation
    b_1 = tf.to_int32(b)
    y_1 = tf.to_int32(tf.floor(y_t))
    y_2 = y_1 + 1
    x_1 = tf.to_int32(tf.floor(x_t))
    x_2 = x_1 + 1
    v_1 = tf.to_int32(v_r)
    u_1 = tf.to_int32(u_r)

    y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
    y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
    x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
    x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

    # assemble interpolation indices
    interp_ind_1 = tf.stack([b_1, y_1, x_1, v_1, u_1], -1)
    interp_ind_2 = tf.stack([b_1, y_2, x_1, v_1, u_1], -1)
    interp_ind_3 = tf.stack([b_1, y_1, x_2, v_1, u_1], -1)
    interp_ind_4 = tf.stack([b_1, y_2, x_2, v_1, u_1], -1)

    # gather light fields to be interpolated
    lf_1 = tf.gather_nd(lf, interp_ind_1)
    lf_2 = tf.gather_nd(lf, interp_ind_2)
    lf_3 = tf.gather_nd(lf, interp_ind_3)
    lf_4 = tf.gather_nd(lf, interp_ind_4)

    # calculate interpolation weights
    y_1_f = tf.to_float(y_1)
    x_1_f = tf.to_float(x_1)
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1

    w_1 = d_y_1 * d_x_1
    w_2 = d_y_2 * d_x_1
    w_3 = d_y_1 * d_x_2
    w_4 = d_y_2 * d_x_2

    return tf.multiply(w_1, lf_1) + tf.multiply(w_2, lf_2) + tf.multiply(
        w_3, lf_3) + tf.multiply(w_4, lf_4)


# render light field rays from input image, given disparities
def render_lf(img, disp, offset, u_sz):
  with tf.name_scope('render_lf'):
    b_sz = tf.shape(img)[0]
    y_sz = tf.shape(img)[1]
    x_sz = tf.shape(img)[2]
    v_sz = u_sz

    lf_r = tf.expand_dims(tf.expand_dims(img, 3), 4)

    # create and transform light field grid
    b_vals = tf.to_float(tf.range(b_sz))
    y_vals = tf.to_float(tf.range(y_sz))
    x_vals = tf.to_float(tf.range(x_sz))
    v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz) / 2.0
    u_vals = tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz) / 2.0

    b, y, x, v, u = tf.meshgrid(
        b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

    # warp coordinates by disparities
    y_t = y + v * (disp + offset)
    x_t = x + u * (disp + offset)

    v_r = tf.zeros_like(b)
    u_r = tf.zeros_like(b)

    # indices for linear interpolation
    b_1 = tf.to_int32(b)
    y_1 = tf.to_int32(tf.floor(y_t))
    y_2 = y_1 + 1
    x_1 = tf.to_int32(tf.floor(x_t))
    x_2 = x_1 + 1
    v_1 = tf.to_int32(v_r)
    u_1 = tf.to_int32(u_r)

    y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
    y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
    x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
    x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

    # assemble interpolation indices
    interp_ind_1 = tf.stack([b_1, y_1, x_1, v_1, u_1], -1)
    interp_ind_2 = tf.stack([b_1, y_2, x_1, v_1, u_1], -1)
    interp_ind_3 = tf.stack([b_1, y_1, x_2, v_1, u_1], -1)
    interp_ind_4 = tf.stack([b_1, y_2, x_2, v_1, u_1], -1)

    # gather light fields to be interpolated
    lf_1 = tf.gather_nd(lf_r, interp_ind_1)
    lf_2 = tf.gather_nd(lf_r, interp_ind_2)
    lf_3 = tf.gather_nd(lf_r, interp_ind_3)
    lf_4 = tf.gather_nd(lf_r, interp_ind_4)

    # calculate interpolation weights
    y_1_f = tf.to_float(y_1)
    x_1_f = tf.to_float(x_1)
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1

    w_1 = d_y_1 * d_x_1
    w_2 = d_y_2 * d_x_1
    w_3 = d_y_1 * d_x_2
    w_4 = d_y_2 * d_x_2

    return tf.multiply(w_1, lf_1) + tf.multiply(w_2, lf_2) + tf.multiply(
        w_3, lf_3) + tf.multiply(w_4, lf_4)


# render disparity rays from input disparity
def render_depth_lf(disp, u_sz):
  with tf.name_scope('render_depth_lf'):
    b_sz = tf.shape(disp)[0]
    y_sz = tf.shape(disp)[1]
    x_sz = tf.shape(disp)[2]
    v_sz = u_sz

    disp_r = tf.expand_dims(disp, 4)

    # create and transform light field grid
    b_vals = tf.to_float(tf.range(b_sz))
    y_vals = tf.to_float(tf.range(y_sz))
    x_vals = tf.to_float(tf.range(x_sz))
    v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz) / 2.0
    u_vals = tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz) / 2.0

    b, y, x, v, u = tf.meshgrid(
        b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

    # warp coordinates by disparities
    y_t = y + v * disp_r
    x_t = x + u * disp_r

    v_r = tf.zeros_like(b)
    u_r = tf.zeros_like(b)

    # indices for linear interpolation
    b_1 = tf.to_int32(b)
    y_1 = tf.to_int32(tf.floor(y_t))
    y_2 = y_1 + 1
    x_1 = tf.to_int32(tf.floor(x_t))
    x_2 = x_1 + 1
    v_1 = tf.to_int32(v_r)
    u_1 = tf.to_int32(u_r)

    y_1 = tf.clip_by_value(y_1, 0, y_sz - 1)
    y_2 = tf.clip_by_value(y_2, 0, y_sz - 1)
    x_1 = tf.clip_by_value(x_1, 0, x_sz - 1)
    x_2 = tf.clip_by_value(x_2, 0, x_sz - 1)

    # assemble interpolation indices
    interp_ind_1 = tf.stack([b_1, y_1, x_1, v_1, u_1], -1)
    interp_ind_2 = tf.stack([b_1, y_2, x_1, v_1, u_1], -1)
    interp_ind_3 = tf.stack([b_1, y_1, x_2, v_1, u_1], -1)
    interp_ind_4 = tf.stack([b_1, y_2, x_2, v_1, u_1], -1)

    # gather light fields to be interpolated
    lf_1 = tf.gather_nd(disp_r, interp_ind_1)
    lf_2 = tf.gather_nd(disp_r, interp_ind_2)
    lf_3 = tf.gather_nd(disp_r, interp_ind_3)
    lf_4 = tf.gather_nd(disp_r, interp_ind_4)

    # calculate interpolation weights
    y_1_f = tf.to_float(y_1)
    x_1_f = tf.to_float(x_1)
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1

    w_1 = d_y_1 * d_x_1
    w_2 = d_y_2 * d_x_1
    w_3 = d_y_1 * d_x_2
    w_4 = d_y_2 * d_x_2

    return tf.multiply(w_1, lf_1) + tf.multiply(w_2, lf_2) + tf.multiply(
        w_3, lf_3) + tf.multiply(w_4, lf_4)


# integrate light field over aperture to render bokeh image
def project_lf(lf, mask):
  with tf.name_scope('project_lf'):
    return tf.reduce_sum(
        tf.reduce_sum(lf * tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(mask, 0), 1), 2), 5),
                      4), 3) / tf.reduce_sum(mask)


# full forward pass (lf model)
def forward_model_lf(center, bokeh0, bokeh1, mask, lfsize, is_training,
                     disp_mult):
  with tf.variable_scope('forward_model') as scope:

    u_sz = lfsize[2]

    d, ds = disp_net_lf(center, is_training, disp_mult, 'disparity')

    dr = render_depth_lf(ds, u_sz)
    dp = disp_refine_net(dr, u_sz, is_training, 'disparities_out')

    shear0 = tf.Variable(0.0, 'shear0')
    shear1 = tf.Variable(0.0, 'shear1')

    lf_r0 = render_lf(center[:, :, :, 0], dp, shear0, u_sz)
    lf_g0 = render_lf(center[:, :, :, 1], dp, shear0, u_sz)
    lf_b0 = render_lf(center[:, :, :, 2], dp, shear0, u_sz)
    lf_render0 = tf.stack([lf_r0, lf_g0, lf_b0], axis=5)

    lf_r1 = render_lf(center[:, :, :, 0], dp, shear1, u_sz)
    lf_g1 = render_lf(center[:, :, :, 1], dp, shear1, u_sz)
    lf_b1 = render_lf(center[:, :, :, 2], dp, shear1, u_sz)
    lf_render1 = tf.stack([lf_r1, lf_g1, lf_b1], axis=5)

    bokeh_render0 = project_lf(lf_render0, mask)
    bokeh_render1 = project_lf(lf_render1, mask)

  return d, ds, dr, dp, bokeh_render0, bokeh_render1, shear0, shear1


# additional forward pass functions for compositional model
def disp_net_compositional(aif, num_depths, is_training, name):
  with tf.variable_scope(name):

    c1 = bn(
        tf.nn.elu(cnn_layer(aif, [3, 3, 3, 4], [4], 'c1', downsample=1)),
        is_training, 'n1')
    c2 = bn(
        tf.nn.elu(cnn_layer(c1, [3, 3, 4, 8], [8], 'c2', downsample=1)),
        is_training, 'n2')
    c3 = bn(
        tf.nn.elu(cnn_layer(c2, [3, 3, 8, 16], [16], 'c3', downsample=1)),
        is_training, 'n3')
    c4 = bn(
        tf.nn.elu(cnn_layer(c3, [3, 3, 16, 64], [64], 'c4', downsample=1)),
        is_training, 'n4')

    c5 = bn(
        tf.nn.elu(d_cnn_layer(c4, [3, 3, 64, 64], [64], 'c5', rate=2)),
        is_training, 'n5') + c4
    c6 = bn(
        tf.nn.elu(d_cnn_layer(c5, [3, 3, 64, 64], [64], 'c6', rate=4)),
        is_training, 'n6') + c5
    c7 = bn(
        tf.nn.elu(d_cnn_layer(c6, [3, 3, 64, 64], [64], 'c7', rate=8)),
        is_training, 'n7') + c6
    c8 = bn(
        tf.nn.elu(d_cnn_layer(c7, [3, 3, 64, 64], [64], 'c8', rate=16)),
        is_training, 'n8') + c7
    c9 = bn(
        tf.nn.elu(d_cnn_layer(c8, [3, 3, 64, 64], [64], 'c9', rate=32)),
        is_training, 'n9') + c8

    c10 = bn(
        tf.nn.elu(cnn_layer(c9, [3, 3, 64, 64], [64], 'c10')), is_training,
        'n10') + c9
    c11 = bn(
        tf.nn.elu(cnn_layer(c10, [3, 3, 64, 32], [32], 'c11')), is_training,
        'n11')

    d = cnn_layer(c11, [3, 3, 32, num_depths], [num_depths], 'c12')

    # This is where the bilateral solver should go, but this code was difficult to open-source.

    ds = tf.nn.softmax(d)

    return d, ds


# shift and scale probabilistic depths to refocus
def transform_depths(d, shift, scale):
  with tf.variable_scope('transform_depths') as scope:
    b_sz = tf.shape(d)[0]
    y_sz = tf.shape(d)[1]
    x_sz = tf.shape(d)[2]
    c_sz = tf.shape(d)[3]

    # create and transform light field grid
    b_vals = tf.to_float(tf.range(b_sz))
    y_vals = tf.to_float(tf.range(y_sz))
    x_vals = tf.to_float(tf.range(x_sz))
    c_vals = tf.to_float(tf.range(c_sz))

    b, y, x, c = tf.meshgrid(b_vals, y_vals, x_vals, c_vals, indexing='ij')

    # warp depths by shift and scale
    c_t = (c + shift) / scale

    # indices for linear interpolation
    b_1 = tf.to_int32(b)
    y_1 = tf.to_int32(y)
    x_1 = tf.to_int32(x)
    c_1 = tf.to_int32(tf.floor(c_t))
    c_2 = c_1 + 1

    c_1 = tf.clip_by_value(c_1, 0, c_sz - 1)
    c_2 = tf.clip_by_value(c_2, 0, c_sz - 1)

    # assemble interpolation indices
    interp_ind_1 = tf.stack([b_1, y_1, x_1, c_1], -1)
    interp_ind_2 = tf.stack([b_1, y_1, x_1, c_2], -1)

    #gather light fields to be interpolated
    img_1 = tf.gather_nd(d, interp_ind_1)
    img_2 = tf.gather_nd(d, interp_ind_2)

    # calculate interpolation weights
    c_1_f = tf.to_float(c_1)
    d_c_1 = 1.0 - (c_t - c_1_f)
    d_c_2 = 1.0 - d_c_1

    w_1 = d_c_1
    w_2 = d_c_2

    return tf.multiply(w_1, img_1) + tf.multiply(w_2, img_2)


# render shallow depth-of-field image
def render_bokeh(aif, kernels, d):
  with tf.variable_scope('render_bokeh') as scope:
    num_depths = tf.shape(d)[3]
    # weighting after convolution
    bokeh_r = tf.reduce_sum(
        d * tf.nn.depthwise_conv2d(
            tf.tile(tf.expand_dims(aif[:, :, :, 0], 3), [1, 1, 1, num_depths]),
            tf.expand_dims(kernels, 3),
            strides=[1, 1, 1, 1],
            padding='SAME'),
        axis=3)
    bokeh_g = tf.reduce_sum(
        d * tf.nn.depthwise_conv2d(
            tf.tile(tf.expand_dims(aif[:, :, :, 1], 3), [1, 1, 1, num_depths]),
            tf.expand_dims(kernels, 3),
            strides=[1, 1, 1, 1],
            padding='SAME'),
        axis=3)
    bokeh_b = tf.reduce_sum(
        d * tf.nn.depthwise_conv2d(
            tf.tile(tf.expand_dims(aif[:, :, :, 2], 3), [1, 1, 1, num_depths]),
            tf.expand_dims(kernels, 3),
            strides=[1, 1, 1, 1],
            padding='SAME'),
        axis=3)
    return tf.stack([bokeh_r, bokeh_g, bokeh_b], axis=3)


# full forward pass (compositional model)
def forward_model_compositional(aif, bokeh0, bokeh1, kernels, num_depths,
                                is_training):
  with tf.variable_scope('forward_model') as scope:

    shift0 = tf.Variable(0.0)
    shift1 = tf.constant(0.0)
    scale0 = tf.constant(1.0)
    scale1 = tf.constant(1.0)

    d, ds = disp_net_compositional(aif, num_depths, is_training, 'disparities')

    d0 = transform_depths(ds, shift0, scale0)

    bokeh_render0 = render_bokeh(aif, kernels, d0)
    bokeh_render1 = render_bokeh(aif, kernels, ds)

  return d, ds, bokeh_render0, bokeh_render1, shift0, shift1, scale0, scale1


if __name__ == '__main__':

  # Directories for saving logs/checkpoints.
  log_root = '/tmp/'
  name = 'lytro_test'
  log_dir = log_root + name + '/logs/'
  train_dir = log_root + name + '/logs/train/'
  test_dir = log_root + name + '/logs/test/'
  check_dir = log_root + name + '/logs/checkpoints/'

  # Dataset directory
  num_test_imgs_per_dataset = 1
  folders = [
      'Path_to_Srinivasan2017_Flower_Dataset'
  ]

  # Parameters
  batchsize = 1
  learning_rate = 0.0001
  max_steps = 240000
  lfsize = [372, 540, 12, 12]
  patchsize = [192, 192]
  num_crops = 4
  disp_mult = 10.0

  # Which forward model to use.
  # is_lf = True  # The "Light Field" model.
  is_lf = False  # The "Compositional" model.

  # The aperture indicator function
  mask = np.array(\
    [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],\
     [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],\
     [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],\
     [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],\
    dtype=np.float32)

  # get filenames for dataset
  filenames, test_filenames = get_filenames(num_test_imgs_per_dataset, folders)

  # input pipeline (queues)
  with tf.device('/cpu:0'):
    aif_batch, bokeh_batch0, bokeh_batch1 = input_pipeline(
        filenames, mask, lfsize, patchsize, num_crops, batchsize)

  # tensorflow placeholders
  is_training = tf.placeholder(tf.bool, shape=[])

  if is_lf:

    # LF Model
    # forward pass
    d, ds, dr, dp, bokeh_render0, bokeh_render1, shear0, shear1 = forward_model_lf(
        aif_batch, bokeh_batch0, bokeh_batch1, mask, lfsize, is_training,
        disp_mult)

    # losses to minimize (lf model)
    lam_tv = 1e-10
    tv_loss = lam_tv * tf.reduce_mean(tf.image.total_variation(d))

    bokeh_loss = l1_loss(bokeh_batch0, bokeh_render0) + l1_loss(
        bokeh_batch1, bokeh_render1)

    lam_disp = 1.0
    disp_expand_loss = lam_disp * l1_loss(dr, dp)

    train_loss = bokeh_loss + disp_expand_loss

    # summaries for tensorboard (lf model)
    tf.summary.scalar('loss', train_loss)
    tf.summary.scalar('loss_bokeh', bokeh_loss)
    tf.summary.scalar('loss_disp_expand', disp_expand_loss)
    tf.summary.scalar('shear0', tf.squeeze(shear0))
    tf.summary.scalar('shear1', tf.squeeze(shear1))
    tf.summary.image('central_disparity', d)
    tf.summary.image('central_disparity_smooth', ds)
    tf.summary.image('input_aif', aif_batch)
    tf.summary.image('input_bokeh0', bokeh_batch0)
    tf.summary.image('input_bokeh1', bokeh_batch1)
    tf.summary.image('render_bokeh0', bokeh_render0)
    tf.summary.image('render_bokeh1', bokeh_render1)
    tf.summary.image(
        'dr',
        tf.expand_dims(
            tf.reshape(
                tf.transpose(dr, perm=[0, 3, 1, 4, 2]),
                [batchsize, patchsize[0] * lfsize[2], patchsize[1] * lfsize[3]
                ]), 3))
    tf.summary.image(
        'dp',
        tf.expand_dims(
            tf.reshape(
                tf.transpose(dp, perm=[0, 3, 1, 4, 2]),
                [batchsize, patchsize[0] * lfsize[2], patchsize[1] * lfsize[3]
                ]), 3))
    tf.summary.histogram('central_disparity', d)
    tf.summary.histogram('central_disparity_smooth', ds)
    merged = tf.summary.merge_all()

    # optimization
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(train_loss)

  if not is_lf:

    num_depths = 31
    kernel_width = 31

    kernels = np.zeros(
        (kernel_width, kernel_width, num_depths), dtype=np.float32)
    for i in range(num_depths):
      depth_val = i - (num_depths - 1.0) / 2.0
      radius = np.abs(depth_val / 2.0)
      yy, xx = np.meshgrid(
          np.arange(kernel_width) - (kernel_width - 1.0) / 2.0,
          np.arange(kernel_width) - (kernel_width - 1.0) / 2.0,
          indexing='ij')
      i_kernel = np.zeros((kernel_width, kernel_width))
      i_kernel[yy * yy + xx * xx <= radius * radius] = 1.0
      kernels[:, :, i] = i_kernel / np.sum(i_kernel)
    kernels_tf = tf.constant(kernels, dtype=tf.float32)

    # Compositional Model
    # forward pass
    d, ds, bokeh_render0, bokeh_render1, shift0, shift1, scale0, scale1 = forward_model_compositional(
        aif_batch, bokeh_batch0, bokeh_batch1, kernels_tf, num_depths,
        is_training)

    # losses to minimize
    lam_tv = 1e-10
    tv_loss = lam_tv * tf.reduce_mean(tf.image.total_variation(d))

    bokeh_loss = l1_loss(bokeh_batch0, bokeh_render0) + l1_loss(
        bokeh_batch1, bokeh_render1)

    train_loss = bokeh_loss + tv_loss

    # summaries for tensorboard
    tf.summary.scalar('loss', train_loss)
    tf.summary.scalar('loss_bokeh', bokeh_loss)
    tf.summary.scalar('loss_tv', tv_loss)
    tf.summary.scalar('shift0', shift0)
    tf.summary.scalar('shift1', shift1)
    tf.summary.scalar('scale0', scale0)
    tf.summary.scalar('scale1', scale1)
    tf.summary.image('depths',
                     tf.expand_dims(tf.to_float(tf.argmax(d, axis=3)), axis=3))
    tf.summary.image('depths_smooth',
                     tf.expand_dims(tf.to_float(tf.argmax(ds, axis=3)), axis=3))
    tf.summary.image('input_aif', aif_batch)
    tf.summary.image('input_bokeh0', bokeh_batch0)
    tf.summary.image('input_bokeh1', bokeh_batch1)
    tf.summary.image('render_bokeh0', bokeh_render0)
    tf.summary.image('render_bokeh1', bokeh_render1)
    merged = tf.summary.merge_all()

    # optimization
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(train_loss)

  # training
  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=None)

    latest_checkpoint = tf.train.latest_checkpoint(check_dir)
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(max_steps):
      # run optimization step
      _, i_loss = sess.run(
          [train_step, train_loss], feed_dict={is_training: True})
      print i, i_loss
      # training set summaries for tensorboard
      if (i + 1) % 20 == 0:
        trainsummary, = sess.run([merged], feed_dict={is_training: True})
        train_writer.add_summary(trainsummary, i)
      if (i + 1) % 2000 == 0:
        saver.save(sess, check_dir + 'model.ckpt', global_step=i)
    train_writer.close()

    coord.request_stop()
    coord.join(threads)
