# 선형회귀분석 -> 입력데이터, 출력 값(레이블) 사용 -> 지도학습알고리즘
# But 모든 데이터에 레이블이 있는 것은 아님 -> 군집화, 자율학습

# [ 군 집 화 ] : 데이터 분석의 사전작업으로 사용됨
# toDo : K-평균 알고리즘 : 유사한 것을 자동 그룹화
# ! 추정할 목표 변수나 결과 변수가 없음

# ? 랭크 : 배열의 차원 (0랭크 : 스칼라, 1랭크 : 일차원, 2랭크 : 행렬)
# 텐서=다차원배열 <- 랭크, 구조, 차원번호
# ([[], 0, 0-D], [[D0], 1, 1-D], [[D0, D1], 2, 2-D])

import tensorflow as tf

points = [[0 for col in range(2)] for row in range(2000)]  # 2*2000 배열 생성
print(points)
vectors = tf.constant(points)  # 상수로 만들어얌
expanded_vectors = tf.expand_dims(vectors, 0)  # 2D -> 3D (텐서의 0번째 위치에 한 차원 삽입)
# 삽입할 위치는 0, 1, 2 가능
print(expanded_vectors.get_shape())  # 모양 알려줘얌

nRandom = tf.random_normal([3,4])
uRandom = tf.random_uniform([3,4])
print(nRandom, uRandom)
# random_normal과 random_uniform의 차이점?
# - random_normal : 정규분포 형태의 난수 생성
# - random_uniform : 균등분포 형태의 난수 생성
# ? 균등분포와 정규분포를 따로 쓰는 이유
# ->