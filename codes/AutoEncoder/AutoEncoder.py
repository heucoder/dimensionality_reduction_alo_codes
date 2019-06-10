# coding:utf-8
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris, load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def AutoEncoder(data,
                hidden_layers = None,
                noise = 0,
                drop_rate = 0,
                n_epochs = 301,
                learning_rate = 0.01,
                optimizer_type = 'adam',
                verbose = 1):

    # if not isinstance(hidden_layers, list) or hidden_layers == []:
    #     raise ("hidden layer input error")

    reset_graph()
    n_inputs = data.shape[1]
    n_outputs = n_inputs

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])

    # add noise
    X_noise = X + noise * tf.random_normal(tf.shape(X))

    # dropout
    training = tf.placeholder_with_default(False, shape=(), name = "training")
    X_drop = tf.layers.dropout(X_noise, drop_rate, training=training)

    hiddens = [X_drop]
    for i in range(len(hidden_layers)):
        n_layer = hidden_layers[i]
        hidden = tf.layers.dense(hiddens[i], n_layer, )
        hiddens.append(hidden)

    outputs = tf.layers.dense(hiddens[-1], n_outputs)
    hiddens.append(outputs)
    # hidden = tf.layers.dense(X_drop, hidden_layers)
    # outputs = tf.layers.dense(hidden, n_outputs)

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

    if optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    training_op = optimizer.minimize(reconstruction_loss)

    init = tf.global_variables_initializer()

    # coding layer
    codings = hiddens[len(hiddens)//2]
    # codings = hidden

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            sess.run(training_op, feed_dict={X: data, training: True})
            loss_train = reconstruction_loss.eval(feed_dict={X: data})
            if epoch % 100 == 0 and verbose:
                print("\r{}".format(epoch), "Train MSE:", loss_train)
        data_ndim = codings.eval(feed_dict={X: data})

    return data_ndim

if __name__ == '__main__':
    iris = load_digits()
    X = iris.data
    Y = iris.target
    X_2d = AutoEncoder(X, [2], learning_rate = 0.2,  n_epochs = 1000)
    plt.subplot(211)
    plt.scatter(X_2d[:, 0], X_2d[:, 1] , c = Y)

    pca = PCA(n_components=2)
    X_2d1 = pca.fit_transform(X)
    plt.subplot(212)
    plt.scatter(X_2d1[:, 0], X_2d1[:, 1], c=Y)
    plt.show()