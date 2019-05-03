# import random
#
# a = ['a', 'b', 'c']
# b = [1, 2, 3]
#
# c = list(zip(a, b))
#
# random.shuffle(c)
#
# random.
# a, b = zip(*c)


import numpy as np

a = np.array([0,1,2,3,4])
b = np.array([5,6,7,8,9])

indices = np.arange(a.shape[0])
np.random.shuffle(indices)

a = a[indices]
b = b[indices]
print(a)
print(b)