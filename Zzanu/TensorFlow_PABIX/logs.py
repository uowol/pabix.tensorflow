import tensorflow as tf
import numpy as np
from functions import showOperation as showOp

# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/array_ops.html#slice

# < tf.slice >
# tf.slice(input_, begin, size, name=None)
# 특정 부분을 추출합니다.
# begin은 0부터 시작하고, size는 1부터 시작합니다.

# 'input'은 [[[1, 1, 1], [2, 2, 2]],
#            [[3, 3, 3], [4, 4, 4]],
#            [[5, 5, 5], [6, 6, 6]]]
# tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
# tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
#                                             [4, 4, 4]]]
# tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
#                                            [[5, 5, 5]]]

# input = [[[1, 1, 1], [2, 2, 2]],
#            [[3, 3, 3], [4, 4, 4]],
#            [[5, 5, 5], [6, 6, 6]]]
# showOp(tf.slice(input, [1, 0, 0], [1, 2, 3]))

# [1, 0, 0]: 2번째 차원의 1번째의 1번째에서 시작
# [1, 2, 3]: 1차원 가져오고 2개, 3개씩 가져와라

# < tf.reduce_sum >
# tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 텐서의 차원을 감소시키는 수학 연산
# reduction_indices: 없앨 차원 = 더해서 없앨 차원

# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
# tf.reduce_sum(x) ==> 6
# tf.reduce_sum(x, 0) ==> [2, 2, 2]
# tf.reduce_sum(x, 1) ==> [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
# tf.reduce_sum(x, [0, 1]) ==> 6

# x = [[1,1,1]
#     ,[1,1,1]]
# showOp(tf.reduce_sum(x,1))
#
# x = [[[1,1,1],
#       [2,2,2],
#       [3,3,3]]
#     ,[[4,4,4],
#       [5,5,5],
#       [6,6,6]]
#     ,[[7,7,7],
#       [8,8,8],
#       [9,9,9]]]
# showOp(tf.reduce_sum(x,2))

# < tf.concat >
# 텐서들을 하나의 차원에서 이어붙입니다.

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
# tensor t3의 구조(shape)는 [2, 3]
# tensor t4의 구조(shape)는 [2, 3]
# tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
# tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]



