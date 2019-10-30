import numpy as np
import time
# def func(x):
# 	x = np.array(x)
# 	return 2*x
# a = np.array([1,2,3])
# b = np.array([1,2,3])
# #print(a)
# #print(func(a))
# print(a*b)
# print(np.multiply(a,b))
# print(a.dot(b.transpose()))
# #for i in range(2,-1,-1):
# 	#print(a[i])

# a = np.random.normal(size=100000000)
# b = np.random.normal(size=100000000)

# start = time.time()
# np.multiply(a, b)
# finish = time.time()
# print(finish-start)

# start = time.time()
# a*b
# finish = time.time()
# print(finish-start)

# 1000000 loops, best of 3: 1.57 µs per loop

# %timeit a - b
# # 1000000 loops, best of 3: 1.47 µs per loop

# %timeit np.divide(a, b)
# # 100000 loops, best of 3: 3.51 µs per loop

# %timeit a / b
# # 100000 loops, best of 3: 3.38 µs per loop	

a = np.zeros([2,2])
b = np.zeros([3,3])
c = [a,b]
print(c[0])
print(c[1])