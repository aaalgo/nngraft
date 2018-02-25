#!/usr/bin/env python
import sys
# that's my local caffe installation
sys.path.append('../caffe/python')
import tempfile
import numpy as np
import tensorflow as tf
import caffe

# This program proves the concept of hybrid network
# training with a bottom half implemented with caffe
# and top half implemented with tensorflow.

# We demonstrate the idea with a trivial example
# where each half contains only a linear layer.
# But the create_xxx_net and load_xxx_weights
# can be replaced with code that creats/loads
# arbitrarily complex network with single input
# and output.

# Our symbol model is
#     Input: X shape 1x2x4x4,  value 1s
#            Y shape 1,        value 0

#     L1 <-- X * W1 + b1,   shape 1x10
#            W1 shape 32x10, value 7s
#            b1 shape 10, value 0
#
#            L1 shape 1x10, value  224s  (2*4*4 * 7)

#     L2 <-- L1 * W2 + b2
#            W2 shape 10x1, value 5s
#            b2 shape 1, value 0

#            L2 shape 1, value 11200
#
#     Loss <-- l2_loss(L2, Y)
#            value = 0.5 * (11200-0)^2 = 62720000

# Gradient calculation:
#
#     d Loss/d L1 = (L2 - Y) * W2
#     d Loss/d X = d Loss/d L1 * d L1 / d X
#                = (L2 - Y) * W2 * W1
#                = 11200 * [7]_{10x1}' * [5]_{32x10}'
#                = [3920000]_10

#     d Loss/d W2 = (L2 - Y) * L1
#                = [2508800]_10

#     We optimize with SGD with top half learning rate = 0.01
#     and bottom half learning rate 0.001

#     W2 updated is  5 - 2508800 * 0.01 = -25083

#     d Loss/d W1 = d Loss/ d L2 * d L2 / d L1  * d L1 / d W1
#                 = 11200 * [5]_{10x1} * [1]_{1x32}
#                 = [56000]_{10x32}

#     W1 updated is  7 - 56000 * 0.001 = -49.0
                 


# This can be replaced with one which loads
# a real and complicated caffe model
def create_caffe_net ():
    model_txt = '''
        name: 'testnet'     force_backward: true
        layer { type: 'DummyData' name: 'X' top: 'X'
          dummy_data_param { shape: { dim: 1 dim: 2 dim: 4 dim: 4 }}
        }
        layer { type: 'InnerProduct' name: 'L1' bottom: 'X' top: 'L1'
          inner_product_param { num_output: 10 }
        }
    '''
    # write prototxt into a temporary file
    model_file = tempfile.NamedTemporaryFile(delete=True) 
    with open(model_file.name, 'w') as f:
        f.write(model_txt)
        pass
    solver_txt = 'net: "' + model_file.name + '''"
        base_lr: 0.001
        lr_policy: "step"
        solver_mode: GPU
        stepsize: 1000000
        '''
    solver_file = tempfile.NamedTemporaryFile(delete=True) 
    with open(solver_file.name, 'w') as f:
        f.write(solver_txt)
        pass

    solver = caffe.SGDSolver(solver_file.name)
    return solver.net, solver

# and weights from pre-trained models
def load_caffe_weights (net):
    net.params['L1'][0].data[...] = np.ones((10, 32), dtype=np.float32) * 7
    net.params['L1'][1].data[...] = np.zeros((10,), dtype=np.float32)
    pass

# The tensorflow net can also be more complicated
class create_tf_net:    # this is to be used as a function
    def __init__ (net):
        net.L1 = tf.placeholder(tf.float32, shape=(1, 10))
        net.Y = tf.placeholder(tf.float32, shape=(1,))

        # weights here are placeholders
        # we'll load real weights later
        net.W2 = tf.Variable(tf.zeros((10, 1)))
        net.B2 = tf.Variable(tf.zeros((1,)))

        net.L2 = tf.matmul(net.L1, net.W2) + net.B2
        net.loss =  tf.nn.l2_loss(net.L2 - net.Y)

        net.dL1 = tf.gradients(net.loss, net.L1)[0]

        net.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(net.loss)
        pass
    pass

# In practice this is not necessary.
# when two nets are put together, the weights of the
# top half must be trained from scratch anyway

# in practice set to random initialization, or
# set proper initializer when create the variables and
# have this done by tf.global_variables_initializer()
def load_tf_weights (net, sess):
    sess.run(net.W2.assign(np.ones((10,1), dtype=np.float32) * 5))
    sess.run(net.B2.assign(np.zeros((1,), dtype=np.float32)))
    pass


# init caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# create network architectures
caffe_net, caffe_solver = create_caffe_net()
tf_net = create_tf_net()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # load parameters
    # logitcally this happens within session 
    load_caffe_weights(caffe_net)
    load_tf_weights(tf_net, sess)

    while True: # training, even though we only do one example

        # in practice x and y are sampled from some dataset
        # with augmentation
        x = np.ones((1,2,4,4), dtype = np.float32)
        y = np.zeros((1,), dtype = np.float32)


        # caffe forward part
        caffe_net.blobs['X'].data[...] = x

        caffe_net.forward()

        # extract L1 from caffe net
        l1 = caffe_net.blobs['L1'].data

        # optional channel swapping and reshaping

        # tensorflow part
        _, dl1, loss = sess.run([tf_net.optimizer, tf_net.dL1, tf_net.loss],
                              feed_dict = {tf_net.L1: l1, tf_net.Y: y})

        # optional channel swapping and reshaping

        # forward gradients to caffe part
        caffe_net.blobs['L1'].diff[...] = dl1

        # caffe backward part
        # caffe_net.backward() -- not necessary
        # -- step below does forward (wasted) and backward

        # update caffe gradients
        caffe_solver.step(1)

        # we only do one example

        print 'loss (expect 62720000):', loss
        print 'dX (expect 3920000s):', caffe_net.blobs['X'].diff
        print 'new W2 (expect -25083s):', tf_net.W2.eval()
        print 'new W1 (expect -49.0s):', caffe_net.params['L1'][0].data
        break
    pass

