foo = [1, 1, 2, 3, 3, 4, 5, 6]
bar = []
dict = {}
for i in range(len(foo)):
    if foo[i] not in dict:
        bar.append(foo[i])
        dict[foo[i]] = i

print(bar)


