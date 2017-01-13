# Saving Variables

* Create an interactive session after importing Tensorflow into the program
```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

* Input a set of random variables in a list
```python
raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
```

```python
spikes = tf.Variable([False] * len(raw_data), name='spikes')
spikes.initializer.run()
```

* The saver op will enable saving and restoring
```python
saver = tf.train.Saver()
```

* Loop through the data and update the spike variable when there is a significant increase
```python
for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()
```

* Now save your variable to your disk
```python
save_path = saver.save(sess, "spikes.ckpt")
print("spikes data saved in file: %s" % save_path)
```

* Make sure to close your session at the end of it!
```python
sess.close()
```