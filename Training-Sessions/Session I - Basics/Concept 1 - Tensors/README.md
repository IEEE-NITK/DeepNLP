# Basics of TensorFlow

## Concept 1 - Defining Tensors

* Initially import the required libraries
```python
import tensorflow as tf
import numpy as np

```

* Now define a 2x2 matrix in the following ways

* First let's define a standard array that is defined in the Python library.
* However (not all) Tensorflow or NumPy functions/ methods can be applied to this array. 
```python
m1 = [[1.,2.], [3.,4.]]
```

* Now let's define a NumPy array along the same lines.
* We can manipulate a NumPy array by calling methods defined in the NumPy Library by using this array.

```python
m2 = np.array([[1.0,2.0], [3.0,4.0]])
```

* **What is tf.constant?**
* The tf.constant() op takes a numpy array (or something implicitly convertible to a numpy array), and returns a tf.Tensor whose value is the same as that array.
For more reference, click [here](https://www.tensorflow.org/api_docs/python/constant_op/)

* Now let's define a Tensor
```python
m3 = tf.constant([[1.0,2.0], [3.0,4.0]])
```

* Now let's check the type of each of the variables
```python
print(type(m1))
print(type(m2))
print(type(m3))
```

* As we can see that we have different types of objects wherein all of them cannot be manipulated using Tensorflow methods.
* So we use a function called `convert_to_tensor(...)` which does exactly what you think it does! :smile:
* `dtype` in TensorFlow refers to the data type of the tensor output.

```python
t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)
```

* Let's check again what we obtained. If our function worked or not
```python
print(type(t1))
print(type(t2))
print(type(t3))
```

