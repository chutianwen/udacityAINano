import numpy as np

class A:
	def __init__(self):
		self.a = 1
		self.b = 4
		self.__hash = None

	def __hash__(self):
		self.__hash = self.__hash or hash(self.a)
		return self.__hash + int(np.random.randint(0, 100))

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and
		        self.a == other.a)


# Test object if inside set/dict requires __eq__ and __hash__ to be same. (seemingly)
DICT = dict()
for id in range(2):
	DICT[A()] = id
print("size of dict", len(DICT))

t2 = A()
t3 = A()
print("t2 hash:", hash(t2), "\tt3 hash:", hash(t3))
print("t2 in dict", t2 in DICT)
print("t2 == t3", t2 == t3)
print("Dict value", DICT[t2])

import copy
t4 = copy.copy(t3)
print(t4)
print(t3 == t4)

t3.b = 5

# t3.a = 4
T = {t2, t3}
print(len(T), "DDD")
for tt in T:
	print(tt.b)

print(hash(t3))

a = [1,2,3]
print("\n")
for x in a[:]:
	if x < 3:
		a.remove(x)
	print(a)
print(a)