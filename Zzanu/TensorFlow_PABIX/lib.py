import tensorflow as tf
import functions

c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.constant([1, 3, 5])
v1 = tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
v2 = tf.constant([[1, 2, 3], [7, 8, 9]])

print('-----------slice------------')
functions.showOperation(tf.slice(c1, [2], [3]))             # [5 7 9]
functions.showOperation(tf.slice(v1, [0, 2], [1, 2]))       # [[3 4]]
functions.showOperation(tf.slice(v1, [0, 2], [2, 2]))       # [[3 4] [9 0]]
functions.showOperation(tf.slice(v1, [0, 2], [2,-1]))       # [[3 4 5 6] [9 0 1 2]]

print('-----------split------------')
# functions.showOperation(tf.split(0, 2, c1)) # [[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]]
# functions.showOperation(tf.split(0, 5, c1)) # [[1, 3], [5, 7], [9, 0], [2, 4], [6, 8]]
# functions.showOperation(tf.split(0, 2, v1)) # [[[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 0, 1, 2]]]
# functions.showOperation(tf.split(1, 2, v1)) # [[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [0, 1, 2]]]

print('-----------tile------------')
functions.showOperation(tf.tile(c2, [3]))   # [1 3 5 1 3 5 1 3 5]
# [[1 2 3 1 2 3] [7 8 9 7 8 9] [1 2 3 1 2 3] [7 8 9 7 8 9]]
functions.showOperation(tf.tile(v2, [2, 2]))

print('-----------pad------------')         # 2차원에 대해서만 동작
# [[0 0 0 0 0 0 0]
#  [0 0 1 2 3 0 0]
#  [0 0 7 8 9 0 0]
#  [0 0 0 0 0 0 0]]
functions.showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'CONSTANT'))
# [[9 8 7 8 9 8 7]
#  [3 2 1 2 3 2 1]
#  [9 8 7 8 9 8 7]
#  [3 2 1 2 3 2 1]]     # 3 2 1 2 3 2 1 2 3 2 1 처럼 반복
functions.showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'REFLECT'))
# [[2 1 1 2 3 3 2]
#  [2 1 1 2 3 3 2]
#  [8 7 7 8 9 9 8]
#  [8 7 7 8 9 9 8]]     # 3 2 1 (1 2 3) 3 2 1. 가운데와 대칭
functions.showOperation(tf.pad(v2, [[1, 1], [2, 2]], 'SYMMETRIC'))

print('-----------concat------------')
# functions.showOperation(tf.concat(0, [c1, c2]))     # [1 3 5 7 9 0 2 4 6 8 1 3 5]
# functions.showOperation(tf.concat(1, [v1, v2]))     # [[1 2 3 4 5 6 1 2 3] [7 8 9 0 1 2 7 8 9]]
# functions.showOperation(tf.concat(0, [v1, v2]))   # error. different column size.

c3, c4 = tf.constant([1, 3, 5]), tf.constant([[1, 3, 5], [5, 7, 9]])
v3, v4 = tf.constant([2, 4, 6]), tf.constant([[2, 4, 6], [6, 8, 0]])

print('-----------pack------------')           # 차원 증가. tf.pack([x, y]) = np.asarray([x, y])
# functions.showOperation(tf.pack([c3, v3]))      # [[1 3 5] [2 4 6]]
# functions.showOperation(tf.pack([c4, v4]))      # [[[1 3 5] [5 7 9]]  [[2 4 6] [6 8 0]]]

# t1 = tf.pack([c3, v3])
# t2 = tf.pack([c4, v4])

print('-----------unpack------------')         # 차원 감소
# functions.showOperation(tf.unpack(t1))          # [[1, 3, 5], [2, 4, 6]]
# functions.showOperation(tf.unpack(t2))          # [[[1, 3, 5], [5, 7, 9]],  [[2, 4, 6], [6, 8, 0]]]

print('-----------reverse------------')
# functions.showOperation(tf.reverse(c1, [True]))         # [8 6 4 2 0 9 7 5 3 1]
# functions.showOperation(tf.reverse(v1, [True, False]))  # [[7 8 9 0 1 2] [1 2 3 4 5 6]]
# functions.showOperation(tf.reverse(v1, [True, True ]))  # [[2 1 0 9 8 7] [6 5 4 3 2 1]]

print('-----------transpose------------')      # perm is useful to multi-dimension .
# functions.showOperation(tf.transpose(c3))       # [1 3 5]. not 1-D.
# functions.showOperation(tf.transpose(c4))       # [[1 5] [3 7] [5 9]]
# functions.showOperation(tf.transpose(c4, perm=[0, 1]))   # [[1 3 5] [5 7 9]]
# functions.showOperation(tf.transpose(c4, perm=[1, 0]))   # [[1 5] [3 7] [5 9]]

print('-----------gather------------')
functions.showOperation(tf.gather(c1, [2, 5, 2, 5]))     # [5 0 5 0]
print(v1)
functions.showOperation(tf.gather(v1, [0, 1,0,1,0,0]))           # [[1 2 3 4 5 6] [7 8 9 0 1 2]]
functions.showOperation(tf.gather(v1, [[0, 0], [1, 1]])) # [[[1 2 3 4 5 6] [1 2 3 4 5 6]]  [[7 8 9 0 1 2] [7 8 9 0 1 2]]]

print('-----------one_hot------------')         # make one-hot matrix.
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]
#  [ 0.  1.  0.]]
functions.showOperation(tf.one_hot([0, 1, 2, 1], 3))
# [[ 0.  0.  0.  1.]
#  [ 0.  0.  0.  0.]
#  [ 0.  1.  0.  0.]]
functions.showOperation(tf.one_hot([3, -1, 1], 4))
