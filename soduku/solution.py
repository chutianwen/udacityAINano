from soduku.utils import *


def search2(values):
	"""
	Using depth-first search and propagation, try all possible values.
	This method is wrong, since reduce_puzzle(values) will also change values.
	Backtracking is not sufficient.
	:param values:
	:return:
	"""
	# First, reduce the puzzle using the previous function

	values = reduce_puzzle(values)
	if values is False:
		return False  ## Failed earlier
	if all(len(values[s]) == 1 for s in boxes):
		return values  ## Solved!
	# Choose one of the unfilled squares with the fewest possibilities
	n, s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)

	# Now use recurrence to solve each one of the resulting sudokus, and
	oldValue = values[s]
	for value in values[s]:
		values[s] = value
		attempt = search(values)
		if attempt:
			return attempt
	values[s] = oldValue
	return False

def search(values):
	"Using depth-first search and propagation, try all possible values."
	# First, reduce the puzzle using the previous function
	values = reduce_puzzle(values)
	if values is False:
		return False ## Failed earlier
	if all(len(values[s]) == 1 for s in boxes):
		return values ## Solved!
	# Choose one of the unfilled squares with the fewest possibilities
	n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
	# Now use recurrence to solve each one of the resulting sudokus, and
	for value in values[s]:
		new_sudoku = values.copy()
		new_sudoku[s] = value
		attempt = search(new_sudoku)
		if attempt:
			return attempt


grid2 = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
values = grid_values(grid2)
# display(values)
res = search2(values)
display(res)
print(search(values) == search2(values))