import tensorflow as tf
from functions import showOperation as showOp

showOp(tf.zeros([2,3]))                     # [[ 0.  0.  0.] [ 0.  0.  0.]]
showOp(tf.ones([2,3], tf.int32))            # [[1 1 1] [1 1 1]]
showOp(tf.zeros_like(tf.ones([2,3])))       # [[ 0.  0.  0.] [ 0.  0.  0.]]
showOp(tf.fill([2,3], 2))                   # [[2 2 2] [2 2 2]]
showOp(tf.fill([2,3], 2.0))                 # [[ 2.  2.  2.] [ 2.  2.  2.]]

print('# ------------------------------------------------ #')

showOp(tf.linspace(1.0, 10.0, 4))           # [  1.   4.   7.  10.]
showOp(tf.range(5))                         # [0 1 2 3 4]
showOp(tf.range(0, 5))                      # [0 1 2 3 4]
showOp(tf.range(0, 10, 2))                  # [0 2 4 6 8]

print('# ------------------------------------------------ #')

# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

# [[-0.14479451 -0.50265551  1.38471031] [-2.46794224  0.56639165 -0.59352636]]
showOp(tf.random_normal([2, 3]))
# [[ 5.0084033   6.28203726  5.49111032] [ 4.81292725  4.8216362   4.82326126]]
showOp(tf.random_normal([2, 3], mean=5.0))
# [[ 0.02739749  0.15692481 -0.18835409] [-0.20757729  0.76803416 -0.07633832]]
showOp(tf.random_normal([2, 3], stddev=0.35))
# [[ 4.7160387   5.51960945  5.0228653 ] [ 4.14505386  5.03473711  5.20692873]]
showOp(tf.random_normal([2, 3], mean=5.0, stddev=0.35, seed=1))

print('# ------------------------------------------------ #')

# -0.1293 -0.0633 0.0984 0.0508 0.1916 0.1197 -0.3135 -0.0823 -0.0300 0.0430
for _ in range(10):
    showOp(tf.reduce_sum(tf.random_normal([2, 3], stddev=0.35) / 6))

# 5.03024 4.9332 4.97116 4.96042 4.99974 5.15176 4.961 4.64518 5.26943 5.18463
for _ in range(10):
    showOp(tf.reduce_sum(tf.random_normal([2, 3], mean=5.0, stddev=0.35) / 6))

print('# ------------------------------------------------ #')

# tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

# [[ 0.67391849  0.66735017  0.0794853 ] [ 0.64219582  0.47089899  0.68388402]]
showOp(tf.random_uniform([2, 3]))
# [[ 1.28794432  2.02983379  2.73823977] [ 2.97947931  4.48065186  4.69495058]]
showOp(tf.random_uniform([2, 3], minval=5))    # no error. do not work.
# [[ 13.32174492   3.7804091   10.80069256] [  8.99943733   0.50998271   9.10907364]]
showOp(tf.random_uniform([2, 3], maxval=15))
# [[ 8.30814171  8.18294716  6.43582296] [ 5.31517982  9.81415558  8.0894165 ]]
showOp(tf.random_uniform([2, 3], 5, 10))
# [[ 7.41970491  7.5310154   6.27571869] [ 5.62371159  9.66505051  8.32304668]]
showOp(tf.random_uniform([2, 3], 5, 10, seed=7))

print('# ------------------------------------------------ #')

# tf.random_shuffle(value, seed=None, name=None)

showOp(tf.random_shuffle(tf.Variable([1,2,3,4,5,6])))           # [3 4 6 5 2 1]
showOp(tf.random_shuffle(tf.Variable([[1,2], [3,4], [5,6]])))   # [[5 6] [3 4] [1 2]]

print('# ------------------------------------------------ #')

# [[ 0.23903739  0.92039955  0.05051243] [ 0.49574447  0.83552229  0.02647042]]
showOp(tf.random_uniform([2, 3], seed=1))
# [[ 0.23903739  0.92039955  0.05051243] [ 0.49574447  0.83552229  0.02647042]]
showOp(tf.random_uniform([2, 3], seed=1))

print('# ------------------------------------------------ #')

# tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# The generated values follow a normal distribution with specified mean and standard deviation,
# except that values whose magnitude is more than 2 standard deviations
# from the mean are dropped and re-picked.

# [[-0.50039721 -1.03818107 -0.1909811 ] [-0.41344216 -0.91877717  0.24455361]]
showOp(tf.truncated_normal([2,3]))
# [[ 3.40512562 -0.11788981  1.92399371] [ 6.6576376   4.21847486  4.41285849]]
showOp(tf.truncated_normal([2,3], stddev=3.5))
