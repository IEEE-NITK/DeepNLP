# Loading Variables

* Now that you have learned how to save variables, you will learn how to load variables what you had already saved!

* Start by creating an Interactive Session in Python
```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

Create a Boolean variable called spikes and initialize its initial value to be `False` 
```python
spikes = tf.Variable([False]*8, name='spikes')
```

Restore the variable data from the disk and you can manipulate it as you want!
```python
saver = tf.train.Saver()
try:
    saver.restore(sess, 'spikes.ckpt')
    print(spikes.eval())
except:
    print('file not found')
```

Close the session at the end!
```python
sess.close()
```
