from itertools import zip_longest


lst1 = list(range(5))
lst2 = list(range(10))
print(lst1)
print(lst2)

for a, b in zip_longest(lst1, lst2):
    print(a, b)
