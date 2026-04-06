

class CSP: 
	def __init__(self,variables, domains, constraints): 
		"""
		Initialization of the CSP class

		Parameters:
		- variables
		- domains
		- constraints

		Objective:
		- solution: sudoku solution
		- viz: everything needed for visualization
		"""
		self.variables = variables 
		self.domains = domains 
		self.constraints = constraints 
		self.solution = None
		self.viz = []

	@staticmethod
	def print_sudoku(puzzle): 
		for i in range(9): 
			if i % 3 == 0 and i != 0: 
				print("- - - - - - - - - - - ") 
			for j in range(9): 
				if j % 3 == 0 and j != 0: 
					print(" | ", end="") 
				print(puzzle[i][j], end=" ") 
			print() 

	def visualize(self):
		if not self.viz:
			print("No visualization steps recorded")
			return

		print("\n" + "=" * 10 + " Visualization " + "=" * 10)
		for step_number, step in enumerate(self.viz, start=1):
			action = step["action"]
			var = step["var"]
			value = step["value"]
			board = step["board"]
			print(f"Step {step_number}: {action} {value} at cell {var}")
			self.print_sudoku(board)
			print()

	def solve(self): 
		assignment = {} 
		self.solution = self.backtrack(assignment, self.domains) 
		return self.solution 
	
	def forward_checking(self, var, value, assignment, domains_copy):
		"""
		Function that removes the value from the domains of free variables that are in the constraints of the var

		Parameters:
		- var: variable that was assigned the value
		- value: value that was assigned to the variable
		- assignment: dict with all the assignments to the variables
		- domains_copy: get the copy of the domains to avoid unwanted changes

		"""
		for neighbor in self.constraints[var]:
			if neighbor not in assignment:
				if value in domains_copy[neighbor]:
					domains_copy[neighbor].remove(value)
				if len(domains_copy[neighbor]) == 0:
					return False
		return True
	
	# Validate the input puzzle
	@staticmethod
	def validate_input(puzzle):
		if len(puzzle) != 9:
			return False

		for row in puzzle:
			if len(row) != 9:
				return False
			for val in row:
				if not isinstance(val, int) or val < 0 or val > 9:
					return False

		# Check duplicates in rows
		for row in puzzle:
			values = [val for val in row if val != 0]
			if len(values) != len(set(values)):
				return False

		# Check duplicates in columns
		for col in range(9):
			values = []
			for row in range(9):
				if puzzle[row][col] != 0:
					values.append(puzzle[row][col])
			if len(values) != len(set(values)):
				return False

		# Check duplicates in 3x3 blocks
		for start_row in range(0, 9, 3):
			for start_col in range(0, 9, 3):
				values = []
				for row in range(start_row, start_row + 3):
					for col in range(start_col, start_col + 3):
						if puzzle[row][col] != 0:
							values.append(puzzle[row][col])
				if len(values) != len(set(values)):
					return False

		return True

	# Verify if the value is not already assigned to a neighbor
	def is_consistent(self, var, value, assignment):
		for neighbor in self.constraints[var]:
			if neighbor in assignment and assignment[neighbor] == value:
				return False
		return True
	
	# Build the board for visualization based on the current assignment
	def build_board(self, assignment):
		board = [row[:] for row in puzzle]
		for (i, j), value in assignment.items():
			board[i][j] = value
		return board
	  
	def backtrack(self, assignment, domains): 
		"""
		Backtracking algorithm

		Parameters:
		- assignment: dict with all the assignments to the variables

		Returns:
		- assignment: dict with all the assigments to the variables, or None if solution is not found. Return the first found solution
		"""
		# Verify if every var are already assigned
		if len(assignment) == len(self.variables):
			return assignment
		
		# Chose a non assigned var 
		for var in self.variables:
			if var not in assignment:
				current_var = var
				break
		
		# Try every values in its domain
		for value in domains[current_var]:

			if self.is_consistent(current_var, value, assignment):
				assignment[current_var] = value
				
				self.viz.append({
					"action": "assign",
					"var": current_var,
					"value": value,
					"board": self.build_board(assignment)
				})

				domains_copy = {v: domains[v][:] for v in domains}

				domains_copy[current_var] = [value]

				# Update domains and constraints
				if self.forward_checking(current_var, value, assignment, domains_copy):
	
					# Continue recursivly to find a solution in this path
					result = self.backtrack(assignment, domains_copy)
					if result is not None:
						return result
				
				self.viz.append({
					"action": "backtrack from",
					"var": current_var,
					"value": value,
					"board": self.build_board({k: v for k, v in assignment.items() if k != current_var})
				})

				del assignment[current_var]
			
		return None

TEST_CASES = {
	"valid_standard": [
		[5, 3, 0, 0, 7, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 5, 0, 0, 0],
		[0, 9, 8, 0, 0, 0, 0, 6, 0],
		[0, 0, 0, 0, 0, 3, 0, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 0, 6],
		[0, 0, 0, 0, 0, 0, 2, 8, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 8],
		[0, 0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 4, 0, 0]
	],
	"already_solved": [
		[5, 3, 4, 6, 7, 8, 9, 1, 2],
		[6, 7, 2, 1, 9, 5, 3, 4, 8],
		[1, 9, 8, 3, 4, 2, 5, 6, 7],
		[8, 5, 9, 7, 6, 1, 4, 2, 3],
		[4, 2, 6, 8, 5, 3, 7, 9, 1],
		[7, 1, 3, 9, 2, 4, 8, 5, 6],
		[9, 6, 1, 5, 3, 7, 2, 8, 4],
		[2, 8, 7, 4, 1, 9, 6, 3, 5],
		[3, 4, 5, 2, 8, 6, 1, 7, 9]
	],
	"invalid_duplicate": [
		[5, 5, 0, 0, 7, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 5, 0, 0, 0],
		[0, 9, 8, 0, 0, 0, 0, 6, 0],
		[0, 0, 0, 0, 0, 3, 0, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 0, 6],
		[0, 0, 0, 0, 0, 0, 2, 8, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 8],
		[0, 0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 4, 0, 0]
	],
	"invalid_size": [
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9]
	]
}

def run_test_case(test_name, show_visualization=False):
	puzzle = TEST_CASES[test_name]

	print("\n" + "=" * 10 + f" {test_name} " + "=" * 10)

	if not CSP.validate_input(puzzle):
		print("Invalid puzzle")
		return

	variables = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 0]

	domains = {}
	for var in variables:
		i, j = var
		used = set()

		for col in range(9):
			if puzzle[i][col] != 0:
				used.add(puzzle[i][col])

		for row in range(9):
			if puzzle[row][j] != 0:
				used.add(puzzle[row][j])

		start_row = (i // 3) * 3
		start_col = (j // 3) * 3
		for row in range(start_row, start_row + 3):
			for col in range(start_col, start_col + 3):
				if puzzle[row][col] != 0:
					used.add(puzzle[row][col])

		domains[var] = [v for v in range(1, 10) if v not in used]

	constraints = {}
	for var in variables:
		i, j = var
		neighbors = set()

		for col in range(9):
			if col != j and (i, col) in variables:
				neighbors.add((i, col))

		for row in range(9):
			if row != i and (row, j) in variables:
				neighbors.add((row, j))

		start_row = (i // 3) * 3
		start_col = (j // 3) * 3
		for row in range(start_row, start_row + 3):
			for col in range(start_col, start_col + 3):
				if (row, col) != var and (row, col) in variables:
					neighbors.add((row, col))

		constraints[var] = neighbors

	globals()["puzzle"] = puzzle

	print("*" * 7, "Initial puzzle", "*" * 7)
	CSP.print_sudoku(puzzle)

	csp = CSP(variables, domains, constraints)
	sol = csp.solve()

	solution = [row[:] for row in puzzle]
	if sol is not None:
		for (i, j), value in sol.items():
			solution[i][j] = value

		print("*" * 7, "Solved puzzle", "*" * 7)
		csp.print_sudoku(solution)
		if show_visualization:
			csp.visualize()
	else:
		print("Solution does not exist")

run_test_case("valid_standard", show_visualization=True)
run_test_case("already_solved")
run_test_case("invalid_duplicate")
run_test_case("invalid_size")