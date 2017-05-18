import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time() #시간재기

num_points = 1000 #점 개수
vectors_set = [] #함수값 넣을곳

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1]) #점의 집합 -> 배열


x_data = [v[0] for v in vectors_set] #x값
y_data = [v[1] for v in vectors_set] #y값

plt.plot(x_data, y_data, 'ro') #그려라 ?'ro'
# plt.show()

import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #기울기 [1]은 1행1열의 행렬을 나타냄
b = tf.Variable(tf.zeros([1])) #모든 원소의 값이 0인 행렬?
y = W * x_data + b #직선의 방정식(y=ax+b)

loss = tf.reduce_mean(tf.square(y - y_data)) #y그래프와 기존의 y_data그래프 사이의 거리를 제곱한 값의 평균
#square : 제곱

#!경사하강법 : W와 b를 수정해가며 오차값을 줄여나감, 기울기를 음의 방향으로 줄여나가면서,

optimizer = tf.train.GradientDescentOptimizer(0.5) #0.5 = 학습속도? 조금만 올리거나 내려도 이상해짐
train = optimizer.minimize(loss) #위 학습속도를 가지고 오차값을 최소화하라?

init = tf.global_variables_initializer() #변수들 초기화

sess = tf.Session()
sess.run(init) #시작

for step in range(8): #8번만 반복
    sess.run(train) #학습시작
    print(sess.run(W), sess.run(b))
    print(step, sess.run(loss))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.show()

end_time = time.time()

print(end_time-start_time)