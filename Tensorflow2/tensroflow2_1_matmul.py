import tensorflow as tf
import numpy as np

loaded_data = np.loadtxt('./data-01.csv', delimiter=',')

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("x_data.shape = ", x_data.shape)
print("t_data.shape = ", t_data.shape)

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

X = tf.placeholder(tf.float32, [None, 3])   # Node = 현재 25X3 이지만, 25X3 이 아닌 None을 지정하면 차후 50X3 125X3 등으로 확장이 가능함
T = tf.placeholder(tf.float32, [None, 1])

y = tf.matmul(X, W) + b    # 현재 X, W, b, 를 바탕으로 계산된 값

loss = tf.reduce_mean(tf.square(y - T))    # MSE(Mean Squard Error : 1/2*(y-t)^2) 손실함수 정의

learning_rate = 1e-5    # 학습율

optimizer = tf.train.GradientDescentOptimizer(learning_rate)    # 경사하강법 알고리즘 (RMSP 등 알고리즘 종류 많음) 적용하고 optimizer

train = optimizer.minimize(loss)    # optimizer를 통한 손실함수 최소화