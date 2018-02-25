#!/usr/bin/env python
import sys
# that's my local caffe installation
sys.path.append('../caffe/python')
import tempfile
import numpy as np
import tensorflow as tf
import caffe

class create_tf_net:    # this is to be used as a function
    def __init__ (net):
        net.X = tf.placeholder(tf.float32, shape=(1, 2, 4, 4))
        net.dL1 = tf.placeholder(tf.float32, shape=(1, 10))

        # weights here are placeholders
        # we'll load real weights later
        net.W1 = tf.Variable(tf.zeros((32, 10)))
        net.B1 = tf.Variable(tf.zeros((1,)))

        net.L1 = tf.matmul(tf.reshape(net.X, [1, -1]), net.W1) + net.B1

        net.loss =  tf.reduce_sum(net.L1 * net.dL1)

        net.dX = tf.gradients(net.loss, net.X)[0]
        net.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(net.loss)
        pass
    pass

def load_tf_weights (net, sess):
    sess.run(net.W1.assign(np.ones((32,10), dtype=np.float32) * 7))
    sess.run(net.B1.assign(np.zeros((1,), dtype=np.float32)))
    pass


# This can be replaced with one which loads
# a real and complicated caffe model
def create_caffe_net ():
    model_txt = '''
        name: 'testnet'     force_backward: true
        layer { type: 'DummyData' name: 'L1' top: 'L1'
          dummy_data_param { shape: { dim: 1 dim: 10 }}
        }
        layer { type: 'DummyData' name: 'Y' top: 'Y'
          dummy_data_param { shape: { dim: 1 }}
        }
        layer { type: 'InnerProduct' name: 'L2' bottom: 'L1' top: 'L2'
          inner_product_param { num_output: 1 }
        }
        layer { type: 'EuclideanLoss' name: 'loss' bottom: 'L2', bottom: 'Y' top: 'loss'}
    '''
    # write prototxt into a temporary file
    model_file = tempfile.NamedTemporaryFile(delete=True) 
    with open(model_file.name, 'w') as f:
        f.write(model_txt)
        pass
    solver_txt = 'net: "' + model_file.name + '''"
        base_lr: 0.01
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
    net.params['L2'][0].data[...] = np.ones((1, 10), dtype=np.float32) * 5
    net.params['L2'][1].data[...] = np.zeros((1,), dtype=np.float32)
    pass

# init caffe
caffe.set_device(0)
caffe.set_mode_gpu()

tf_net = create_tf_net()
caffe_net, caffe_solver = create_caffe_net()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    load_tf_weights(tf_net, sess)
    load_caffe_weights(caffe_net)

    while True: # training, even though we only do one example

        x = np.ones((1,2,4,4), dtype = np.float32)
        y = np.zeros((1,), dtype = np.float32)

        l1 = sess.run(tf_net.L1, feed_dict = {tf_net.X: x})

        caffe_net.blobs['L1'].data[...] = l1

        #caffe_net.forward()
        #caffe_net.backward()
        caffe_solver.step(1)

        # extract L1 from caffe net
        dl1 = caffe_net.blobs['L1'].diff

        # optional channel swapping and reshaping

        # tensorflow part
        _, dx = sess.run([tf_net.optimizer, tf_net.dX],
                              feed_dict = {tf_net.X: x, tf_net.dL1: dl1})


        # we only do one example

        print 'loss (expect 62720000):', caffe_net.blobs['loss'].data
        print 'dX (expect 3920000s):', dx
        print 'new W2 (expect -25083s):', caffe_net.params['L2'][0].data
        print 'new W1 (expect -49s):', tf_net.W1.eval()
        break
    pass

