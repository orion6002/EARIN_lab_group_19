

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
		self.viz = None


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
		return 0

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
	
	# Verify if the value is not already assigned to a neighbor
	def is_consistent(self, var, value, assignment):
		for neighbor in self.constraints[var]:
			if neighbor in assignment and assignment[neighbor] == value:
				return False
		return True
	  
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
				
				domains_copy = {v: domains[v][:] for v in domains}

				domains_copy[current_var] = [value]

				# Update domains and constraints
				if self.forward_checking(current_var, value, assignment, domains_copy):
	
					# Continue recursivly to find a solution in this path
					result = self.backtrack(assignment, domains_copy)
					if result is not None:
						return result
					
				del assignment[current_var]
			
		return None

puzzle = [[5, 3, 0, 0, 7, 0, 0, 0, 0], 
		  [0, 0, 0, 1, 0, 5, 0, 0, 0], 
		  [0, 9, 8, 0, 0, 0, 0, 6, 0], 
		  [0, 0, 0, 0, 0, 3, 0, 0, 1], 
		  [0, 0, 0, 0, 0, 0, 0, 0, 6], 
		  [0, 0, 0, 0, 0, 0, 2, 8, 0], 
		  [0, 0, 0, 0, 0, 0, 0, 0, 8], 
		  [0, 0, 0, 0, 0, 0, 0, 1, 0], 
		  [0, 0, 0, 0, 0, 0, 4, 0, 0] 
		] 	
# Based on the puzzle create variables, domains, and constraints for initialization of CSP class

variables = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 0]
domains = {var: list(range(1, 10)) for var in variables}
constraints = {}

for var in variables:
    i, j = var
    neighbors = set()

    # same column
    for col in range(9):
        if col != j and (i, col) in variables:
            neighbors.add((i, col))

    # same row
    for row in range(9):
        if row != i and (row, j) in variables:
            neighbors.add((row, j))

    # same block 3x3
    start_row = (i // 3) * 3
    start_col = (j // 3) * 3
    for row in range(start_row, start_row + 3):
        for col in range(start_col, start_col + 3):
            if (row, col) != var and (row, col) in variables:
                neighbors.add((row, col))

    constraints[var] = neighbors

print('*'*7,'Solution','*'*7) 
csp = CSP(variables, domains, constraints) 
sol, viz = csp.solve() 
csp.print_sudoku(puzzle)
solution = [[0 for i in range(9)] for i in range(9)] 
if sol is not None:
	for i,j in sol: 
		solution[i][j]=sol[i,j] 
		
	csp.print_sudoku(solution)
	csp.visualize()
else:
	print("Solution does not exist")
