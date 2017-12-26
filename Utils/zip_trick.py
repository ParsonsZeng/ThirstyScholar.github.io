tup1 = tuple(range(3))
tup2 = tuple(range(4, 7))
tup3 = tuple(range(7, 10))

print(tup1)
print(tup2)
print(tup3)

Lst1 = [tup1, tup2, tup3]
print(Lst1)

Lst2 = [*zip(*Lst1)]
print(Lst2)

Lst1 = [*zip(*Lst2)]
print(Lst1)

