# Session Logging

* Import Tensorflow 
```python
import tensorflow as tf
``` 

* Define a tensor matrix and negate it
```python
x = tf.constant([[1.,2.]])
neg_op = tf.neg(x)
```

* Now we use a session with a special argument passed in it.

```python
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(neg_op)
    print(result)
```

* As you will see in the output, the value will be negated.

* What does this mean?
* This command is basically to assign GPUs to perform computations automatically for the given operations.
* For more information on this click [here](https://www.tensorflow.org/how_tos/using_gpu/)
```
Device mapping: no known devices.
I tensorflow/core/common_runtime/direct_session.cc:252] Device mapping:

Neg: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] Neg: /job:localhost/replica:0/task:0/cpu:0
Const: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] Const: /job:localhost/replica:0/task:0/cpu:0
[[-1. -2.]]
```