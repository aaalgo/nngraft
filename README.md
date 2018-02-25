# nngraft

This project demonstrates how to train a hybrid network with caffe and
tensorflow components end-to-end.  The code implements a trivial 3-layer
model so that net outputs and updated weights can be verified by
hand-calculated values (and are verified to be correct).  The code is structured such that more
complicated components can be easily plugged in.

Read the comments within `caffe_tf.py` for details.  The other program
`tf_caffe.py` implements the archetecture the other way around.


