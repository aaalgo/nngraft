# nngraft

This project demonstrates how to train a hybrid network with caffe and
tensorflow components end-to-end.  The code use a trivial 3-layer
model so that net outputs and updated weights can be verified by
hand-calculated values (and are verified to be correct), but the code is structured such that more
complicated components can be easily plugged in.

Read the comments of `caffe_tf.py` for details.  The other program
`tf_caffe.py` implements the archetecture the other way around.


