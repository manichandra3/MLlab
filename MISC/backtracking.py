# [1, 2, 3]
#            []
#       [1]  [2]  [3]
#     [1,2] [1,3] [2,3]
#          [1,2,3]
import numpy as np


def test(nums, n):
    path = []
    res = []

    def backtrack(i):
        if i == n:
            res.append(path.copy())
            return
        next_number = nums[i]
        # Not use the number
        backtrack(i + 1)
        # Use the number
        path.append(next_number)
        backtrack(i + 1)
        path.pop()

        return res

    return backtrack(0)


print(test([1, 2, 3, 4], 4))
z = np.array([[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])
print(z)
z = z.reshape(-1, 1)
np.random.shuffle(z)
print(z)
