from sklearn.datasets import load_iris, load_digits
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sklearn_mds(n_com=2):
    mds = MDS(n_components=n_com)
    data = load_digits().data
    target = load_digits().target
    data_2d = mds.fit_transform(data)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c = target)
    plt.show()


def cal_pairwise_dist(x):
    # '''计算pairwise 距离, x是matrix
    # (a-b)^2 = a^2 + b^2 - 2*a*b
    # '''
    sum_x = np.sum(np.square(x), 1)
    # print -2 * np.dot(x, x.T)
    # print np.add(-2 * np.dot(x, x.T), sum_x).T
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def tensor_mds(n_com = 2):
    data = load_digits().data
    std = StandardScaler()
    data = std.fit_transform(data)
    target = load_digits().target
    n, feature = data.shape
    tf.reset_default_graph()
    X_dist = cal_pairwise_dist(data)
    # print(X_dist[0])

    X = tf.placeholder(name = "X", dtype = tf.float32, shape=[n, n])
    Y = tf.get_variable(name = "Y",
                        shape = [n, n_com],
                        initializer=tf.random_uniform_initializer())



    sum_y = tf.reduce_sum(tf.square(Y), 1)
    Y_dist = tf.add(tf.transpose(tf.add(-2*tf.matmul(Y, tf.transpose(Y)), sum_y)), sum_y)

    loss = tf.log(tf.reduce_sum(tf.square((X - Y_dist))))

    optimizer = tf.train.RMSPropOptimizer(learning_rate = 1)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    n_epochs = 1000

    with tf.Session() as sess:
        init.run()
        for i in range(n_epochs):
            training_op.run(feed_dict={X:X_dist})
            if i % 200 == 0:
                loss_val = loss.eval(feed_dict={X:X_dist})
                print("loss: ", loss_val)

        data_2d = sess.run(Y)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c = target)
    plt.show()

if __name__ == '__main__':
    # sklearn_mds()
    tensor_mds()