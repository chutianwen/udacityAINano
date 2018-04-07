def cross(a, b):
	return [s + t for s in a for t in b]


rows = 'ABCDEFGHI'
cols = '123456789'
boxes = cross(rows, cols)

print(boxes)
print("Hello")
