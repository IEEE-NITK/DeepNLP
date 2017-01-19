# Performing Operations in TensorFlow

* Initially import all the necessary libraries for the operation
```python
import tensorflow as tf
```

* Define a 1x2 Tensor matrix
```python
x = tf.constant([[1.0,2.0]])
```

* Let's negate the values and see what happens. 

```python
neg_x = tf.neg(x)
print(neg_x)
```

* As you can see, nothing happens when you print it out. It doesn't even perform the negation computation.
* For that to happen, you need to summon a session so that you can launch your negation operation.
* **What is a Session in Tensorflow**
* A session allocates resources (on one or more machines) for that and holds the actual values of intermediate results and variables.

```python
with tf.Session() as sess:
    result = sess.run(neg_x)
    print(result)
```



