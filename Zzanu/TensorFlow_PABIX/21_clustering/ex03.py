import tensorflow as tf
import numpy as np
from functions import showOperation as showOp


# showOp(tf.Variable([1,2,3,4,5,6]))           # [1 2 3 4 5 6]
# showOp(tf.constant([1,2,3,4,5,6]))           # 결과 동일, 위의 것이 상수로 변수를 초기화시켰으므로

num_points = 20000
vectors_set = []

for i in range(num_points):
    if np.random.random()>0.5:                              # n의 양이 많으면...! 절반은 ~(if)하고 절반은 ~(else)해라
        vectors_set.append([np.random.normal(0.0, 0.9),     # 평균 0, 표준편차 0.9
                            np.random.normal(0.0, 0.9)])    # 평균 0, 표준편차 0.9
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),     # 평균 3, 표준편차 0.5
                            np.random.normal(1.0, 0.5)])    # 평균 1, 표준편차 0.5

vectors = tf.constant(vectors_set)
# 2000: x성분    *    2: y성분
# 4:    x성분    *    2: y성분


k = 6
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))   # 텐서 k-0개 가져왕, 크기는 -1(=끝까지)
# random_shuffle: 값의 첫번째 차원을 기준으로 쒜킷
# slice(input_, begin, size, name=None): 특정 부분을 추출 -> logs

showOp(centroids)


# vectors: 2000*2  centroids: 4*2  <- 안맞음!
expanded_vectors = tf.expand_dims(vectors,0)                # 벡터를 확장하자!
expanded_centroids = tf.expand_dims(centroids,1)

showOp(expanded_centroids)
showOp(expanded_vectors)


# vectors: 1*2000*2  centroids: 4*1*2  <- '1'은 연산시 다른 텐서의 해당 차원크기에 맞게 계산을 반복
# -> vectors: 1개의 묶음 => 2000개의 xy성분(2개)
# -> centroids: 4개의 묶음 => 1개의 xy성분(2개)
# 모든 벡터들을 구심점에 비교 -> 다른 구심점에 비교 -> ...    (4번 반복! ? 구심점이 4개니까!)


# assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0
# subtract: x - y element-wise                              # for 유클리드제곱거리 구하려고!

                                                            # ? 유클리드제곱거리: 두 점 사이의 거리를 계산
# tf.reduce_sum -> logs
diff = tf.subtract(expanded_vectors,expanded_centroids)     # 모든 벡터와 각 구심점 사이의 차이
sqr = tf.square(diff)                                       # 제곱! -> 거리 Get!
print(diff)
showOp(diff)
distance = tf.reduce_sum(sqr, 2)                            # x성분 + y성분       4*2000(*'2')      4*2000
assignments = tf.argmin(distance, 0)                        # 최소값(x성분 + y성분) Get! 1묶음씩 비교    2000   가까운 점 리턴
# !!! assignments엔 가장 근접한 구심점이 할당됨
print(distance)
showOp(distance)
print(assignments)
showOp(assignments)
# 뇌피셜 :
#     벡터 2000개 생성
#     a = 1번 구심점과의 거리차 : ~2000개
#     b = 2번 구심점과의 거리차 : ~2000개
#     c = 3번 구심점과의 거리차 : ~2000개
#     d = 4번 구심점과의 거리차 : ~2000개
#     a,b,c,d 한 벡터씩 비교 -> 가장 거리차가 작은 값의 구심점 Get!
#     2000개의 벡터와 2000개의 가까운 구심점 얻음


showOp(assignments[0])
showOp(assignments[0])
# !!! 우리가 지금 하고 있는거는 변수를 취급하므로 호출할 때 마다 값이 다름.


# tf.concat: 텐서들을 하나의 차원에서 이어붙입니다. -> logs
# tf.reduce_mean: 평균값을 구합니다 === tf.reduce_sum
means = tf.concat([tf.reduce_mean(                          # 5. reduction_indices=[1] : Get!
    tf.gather(                                              # 4. 2000개의 벡터에서 두번째 매개값이 가리키는 벡터만 뽑음
                                                            # - 1 * 반환받은 크기 * 2(xy성분)
        vectors, tf.reshape(                                # 3. 텐서의 구조 변경 [2000,1]->[1,반환받은 크기] : 다시 1차 배열로 만들어줌
            tf.where(                                       # 2. True:1 / False:0 => [2000]->[2000,1] : 아래로 쭈루루루룩
                                                            # - True로 바뀐 벡터의 위치 반환
                tf.equal(                                   # 1. 한 군집과 매칭되는 assignments 텐서의 각 원소 위치를 True로 표시
                    assignments, c                          # - 군집의 번호는 c에 매핑했습니다.
                )
            ), [1,-1]
        )
    ), reduction_indices=[1]) for c in range(k)], 0)

# tf.assign: 어떤 텐서를 다른 텐서로 초기화 (?)
update_centroids = tf.assign(centroids, means)                  # centroids텐서에 means대입
# showOp(update_centroids)


# 뇌피셜(update) :
#     벡터 2000개 생성, 구심점 4개 생성
#   ---여기부터
#     a = 1번 구심점과의 거리차 : ~2000개
#     b = 2번 구심점과의 거리차 : ~2000개
#     c = 3번 구심점과의 거리차 : ~2000개
#     d = 4번 구심점과의 거리차 : ~2000개
#     a,b,c,d 한 벡터씩 비교 -> 가장 거리차가 작은 값의 구심점 Get!
#     2000개의 벡터와 2000개의 가까운 구심점 얻음
#     !!! 다시 한번 더 정확한 구심점(중심)을 찾자(4개)
#     각 구심점에 가까운 점끼리 묶어서 평균으로 새로운 중심을 구하자
#   ---여기까지 반복 -> '군집화'

init_op = tf.global_variables_initializer()                     # 모든 변수들 초기화!

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session()
sess.run(init_op)

for step in range(200):
    _, centroid_values, assignment_values = sess.run([update_centroids,centroids,assignments]) # 변수의 값이 반복으로 정확해짐

print(_)
print(centroid_values)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {"x": [],"y":[],"cluster":[]}                            # 키값 : 데이터

for i in range(len(assignment_values)):
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
print(df,len(assignment_values))
sns.lmplot("x","y",data=df,fit_reg=False,size=6,hue="cluster",  # hue: 분류 => 1은 1끼리 색이 같고...
           palette="Set1",legend=False,col="cluster")           # palette: 색종류, col: 분할
plt.show()