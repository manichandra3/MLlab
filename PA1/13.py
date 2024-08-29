import numpy as np

array = np.random.uniform(low=1, high=10, size=(4, 5))
print(array)
print('transpose:')
print(array.transpose())

array1 = np.zeros(10)
array2 = np.ones(10)
array3 = np.ones(10) * 5
ans = np.concatenate((array1, array2, array3))
print(ans)

array4 = np.arange(10, 51, 2)
print(array4)

rand_num = np.random.uniform(low=0, high=1, size=1)
print(rand_num)

file = open('filename.txt', 'w')
file.write('%s\n' % array)
