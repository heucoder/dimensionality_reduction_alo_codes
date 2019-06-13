# coding:utf-8

'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import tensorflow as tf

def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def cal_perplexity(dist, idx=0, beta=1.0):
    '''计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    这里的perp仅计算了熵，方便计算
    '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob == 0:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
        prob /= sum_prob
    # 困惑度和pi\j的概率分布
    return perp, prob

def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''二分搜索寻找beta,并计算pairwise的prob
    '''

    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." %(i,n))

        betamin = -np.inf
        betamax = np.inf
        #dist[i]需要换不能是所有点
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    # 每个点对其他点的条件概率分布pi\j
    return pair_prob

def tsne(data, no_dims=2, perplexity=30.0, max_iter=800):
    '''

    :param data: (n_samples, n_features)
    :param no_dims: target dimension
    :param perplexity:
    :param max_iter:
    :return: (n_samples, no_dims)
    '''

    # init
    tf.reset_default_graph()

    (n, d) = data.shape
    print(n, d)

    # 对称化
    P = seach_prob(data, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)  # pij
    P = np.maximum(P, 1e-12)
    # 随机初始化Y y = np.random.randn(n, no_dims)
    # tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)

    X = tf.placeholder(name="X", dtype=tf.float32, shape=[n, n])

    Y = tf.get_variable(name="Y", shape=[n, no_dims],
                        initializer=tf.random_normal_initializer())

    sum_y = tf.reduce_sum(tf.square(Y), 1)
    temp = tf.add(tf.transpose(tf.add(-2 * tf.matmul(Y, tf.transpose(Y)), sum_y)), sum_y)
    num = tf.divide(1, 1 + temp)
    # 不知道这句能不能执行
    one_ = tf.constant([x for x in range(n)])
    one_hot = tf.one_hot(one_, n)
    num = num - num * one_hot

    Q = num / tf.reduce_sum(num)
    Q = tf.maximum(Q, 1e-12)

    learning_rate = 500
    loss = tf.reduce_sum(X * tf.log(tf.divide(X, Q)))

    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    print("begin")
    with tf.Session() as sess:
        init.run()
        for iter in range(max_iter):
            sess.run(train_op, feed_dict={X: P})
            if iter % 50 == 0:
                l = sess.run(loss, feed_dict={X: P})
                print("%d\t%f" % (iter, l))
        y = sess.run(Y)
    print("finished")

    return y
if __name__ == '__main__':
    data = load_digits().data
    label = load_digits().target
    data_2d = tsne(data)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], 20, label)
    plt.show()