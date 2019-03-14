import tensorflow as tf
n_inputs = 28*28 
n_hidden1 = 300 
n_hidden2 = 100
n_outputs = 26

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X") 
y = tf.placeholder(tf.int64, shape=(None), name="y")

from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"): 
   hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
   hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2") 
   logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"): #function is equivalent to applying the softmax activation function and then computing the cross entropy, but it is more efficient, and it properly takes care of corner cases like logits equal to 0.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"): 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"): 
    correct = tf.nn.in_top_k(logits, y, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer() 
saver = tf.train.Saver()
#Construction phase
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("/tmp/data/")
import numpy as np
n_epochs = 100
batch_size = 1000
#no_batches=np.int(len(dataset)/batch_size)

with tf.Session() as sess:
    init.run() 
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)  
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) 
        acc_test = accuracy.eval(feed_dict={X:mnist.validation.images, y: mnist.validation.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test) 
    save_path = saver.save(sess, "./my_model_final.ckpt")

with tf.Session() as sess: 
    saver.restore(sess, "./my_model_final.ckpt") 
    X_new_scaled = [...] # some new images (scaled from 0 to 1) 
    Z = logits.eval(feed_dict={X: X_new_scaled}) 
    y_pred = np.argmax(Z, axis=1)
#-----------------------------------------------------------------------------------------#
#please download the data into the working directory from this link https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format
import pandas as pd
dataset=pd.read_csv('A_Z HandwrittenData.csv')
        
X_data=dataset.iloc[:,1:]#Images
y_data=dataset.iloc[:,0]#Target/Labels 0-25 A-Z
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_data,y_data,test_size=0.2)

def next_batch(batch_index):
    start_index=batch_size*batch_index
    end_index=start_index+batch_index
    if batch_index==n_batches:
        end_index=len(X_train)
    return X_train.iloc[start_index:end_index,:],np.asanyarray(y_train.iloc[start_index:end_index]).ravel()

n_batches=np.int(len(X_train)/batch_size)

with tf.Session() as sess:
    init.run() 
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            X_batch, y_batch = next_batch(iteration)  
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) 
        loss_train = loss.eval(feed_dict={X: X_batch, y: y_batch.ravel()})
        acc_test = accuracy.eval(feed_dict={X:X_test, y: np.asarray(y_test).ravel()})
        print(epoch, "Train accuracy:", acc_train, "Train loss:", loss_train, "Test accuracy:", acc_test) 
    save_path = saver.save(sess, "./my_model_final.ckpt")

