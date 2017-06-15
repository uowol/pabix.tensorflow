import tensorflow as tf
import numpy as np
from functions import showOperation as showOp



showOp(tf.Variable([1,2,3,4,5,6]))           # [1 2 3 4 5 6]
showOp(tf.constant([1,2,3,4,5,6]))           # 결과 동일, 위의 것이 상수로 변수를 초기화시켰으므로

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random()>0.5:                              # n의 양이 많으면...! 절반은 ~(if)하고 절반은 ~(else)해라
        vectors_set.append([np.random.normal(0.0, 0.9),     # 평균 0, 표준편차 0.9
                            np.random.normal(0.0, 0.9)])    # 평균 0, 표준편차 0.9
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),     # 평균 3, 표준편차 0.5
                            np.random.normal(1.0, 0.5)])    # 평균 1, 표준편차 0.5

vectors = tf.constant(vectors_set)
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))   # k개 가져왕, 크기는 -1(=끝까지)

showOp(centroids)

# vectors: 2000*2  centroids: 4*2  <- 안맞음
expanded_vectors = tf.expand_dims(vectors,0)                # 벡터를 확장하자!
expanded_centroids = tf.expand_dims(centroids,1)

# vectors: 1*2000*2  centroids: 4*1*2  <- '1'은 연산시 다른 텐서의 해당 차원크기에 맞게 계산을 반복
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)
# subtract: 유클리드제곱거리 구함

means = tf.concat([tf.reduce_mean(
    tf.gather(
        vectors, tf.reshape(
            tf.where(
                tf.equal(
                    assignments, c
                )
            ), [1,-1]
        )
    ), reduction_indices=[1]) for c in range(k)], 0)

update_centroids = tf.assign(centroids, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run(update_centroids,centroids,assignments)