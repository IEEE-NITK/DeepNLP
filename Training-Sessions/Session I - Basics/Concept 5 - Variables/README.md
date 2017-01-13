# Using Variables

* Let's try to understand how variables work in Tensorflow!
* Let's start a Tensorflow session.
```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

* Take in a random series of numbers! Just for fun, let's think of them as [Neural Activations]()
```python
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
```

* Intialize a variable called spike which is of Boolean data type and set its initial value to be `False`.
* All variables must be initialized in Tensorflow by calling the `run()` method on its `initializer`.
```python
spike = tf.Variable(False)
spike.initializer.run()
```

* Loop throught the data and update the spike variable when there is a significant increase
* The if-else methods are two ways to assign values to the variable.
* The function takes in two arguments the variable name and its value and assigns the value of the variable in the function itself.
```python
for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        updater = tf.assign(spike, tf.constant(True))
        updater.eval()
    else:
        tf.assign(spike, False).eval()
    print("Spike", spike.eval())
```

Finally close the session at the end!
```python
sess.close()
```
