import numpy as np
num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random()>0.5:                              # n의 양이 많으면...! 절반은 ~(if)하고 절반은 ~(else)해라
        vectors_set.append([np.random.normal(0.0, 0.9),     # 평균 0, 표준편차 0.9
                            np.random.normal(0.0, 0.9)])    # 평균 0, 표준편차 0.9
    else:
        vectors_set.append([np.random.normal(3.0, 0.5),     # 평균 3, 표준편차 0.5
                            np.random.normal(1.0, 0.5)])    # 평균 1, 표준편차 0.5

import matplotlib.pyplot as plt
import pandas as pd                                         # more 복잡한 형태의 데이터 처리
import seaborn as sns                                       # pyplot 업그레이드

df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                  "y": [v[1] for v in vectors_set]})
sns.lmplot("x","y",data=df, fit_reg=False, size=6)
plt.show()