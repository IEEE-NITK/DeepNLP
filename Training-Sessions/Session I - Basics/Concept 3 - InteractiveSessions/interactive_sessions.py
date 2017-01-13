import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.constant([[1. ,2.]])
neg_op = tf.neg(x)

result = neg_op.eval()
print(result)

sess.close()