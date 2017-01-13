# Interactive Session

* Interactive Sessions are another way to use a Session.
* **What is the difference between a __Session__ and __Interactive Session__?**
Interactive Sessions allow uus to use variables without constantly calling the session object! (Basically less timing) 

```python
import tensorflow as tf
sess = tf.InteractiveSession()
``` 

* Let's try to do the same operation of negating the elements of the matrix.
```python
x = tf.constant([[1.,2.]])
neg_x = tf.neg(x)
``` 

* In an Interactive Session, you can simply call the eval method on the op value.
```python
result = neg_op.eval()
print(result)
``` 

* At the end, don't forget to close the session
```python
sess.close()
``` 
