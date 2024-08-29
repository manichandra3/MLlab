list1 = list(range(1, 11))
list2 = list(range(11, 21))
print(list1)
print(list2)
for i in range(len(list2)):
    list1.append(list2[i])
print(list1)
