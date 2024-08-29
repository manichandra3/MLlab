# swap using a temporary variable
a = 2
b = 3
print('before swap ', a, b)
temp = b
b = a
a = temp
print('after swap ', a, b)

# swap using no temp variable
a = 2
b = 3
print('before swap ', a, b)
a = a ^ b
b = a ^ b
a = a ^ b
print('after swap ', a, b)
