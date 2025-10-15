'''
Author: Chris Hinkson @cmh02
Class: Software Quality Assurance (COMP5710)
Assignment: W6 - Data Flow Analysis

In this assignment I prepare a Data Flow Analysis class that can parse a Python source code file.
It uses Python's Abstract Syntax Tree (AST) module to parse the code and extract assignment statements.

Several data structures are built to represent different data at various stages of the analysis:
- A list of tuples representing assignment statements, i.e. [(variable, AST node object)]
- A list of dictionaries representing operations performed in assignments, i.e. [{"target": variable, "operation": operation_type, "value": value}]
- A dictionary mapping source variables to target variables (flows), i.e. {source: target}
- A unique set of non-source variables (sources), i.e. {variable}
- A list of strings representing paths through the data flows, i.e. [path]
'''

'''
MODULE IMPORTS
'''
import os
import ast
from typing import List, Dict, Any

'''
DATA FLOW ANALYZER CLASS
'''
class DataFlowAnalyzer():
	'''
	Perform a DataFlowAnalysis on a given Python source code file.
	'''

	def __init__(self, codeFilePath: str, autoExtractTree) -> None:
		'''
		Initialize the DataFlowAnalyzer with the path to the code file.

		Parameters:
		- codeFilePath: Path to the Python source code file to analyze
		- autoExtractTree: If True, automatically extract the parse tree upon initialization
		'''

		# Make sure that the code file exists
		if not os.path.isfile(codeFilePath):
			raise FileNotFoundError(f"File not found: {codeFilePath}")

		# Save the file path
		self.codeFilePath = codeFilePath

		# If autoExtractTree is true, run the analysis immediately
		self.ast = None
		if autoExtractTree:
			self.extractParseTree()

	def extractParseTree(self) -> None:
		'''
		Parse the source code file into an AST and visit each node
		'''

		# Parse the source code file into an AST
		with open(self.codeFilePath, "r") as targetFile:
			self.ast = ast.parse(targetFile.read())

	def getAllAssignmentNodes(self) -> List[tuple]:
		'''
		Get all assignment nodes from the AST.

		Returns:
		- List of tuples containing (variable, assignment AST node)
		'''

		# Ensure that the AST has been extracted
		if self.ast is None:
			raise ValueError("AST has not been extracted. Call extractParseTree() first.")
		
		# Complete a walk through the tree to find assignments
		foundAssignments = []
		for node in ast.walk(self.ast):

			# Check if we are at assign statement
			if isinstance(node, ast.Assign):

				# Get the target of the assignment
				targetNode = node.targets[0]

				# Handle name assignments
				if isinstance(targetNode, ast.Name):
					foundAssignments.append((targetNode.id, node.value))

				# Handle tuple-to-tuple assignments
				elif isinstance(targetNode, ast.Tuple) and isinstance(node.value, ast.Tuple):
					for left, right in zip(targetNode.elts, node.value.elts):
						if isinstance(left, ast.Name):
							foundAssignments.append((left.id, right))

				# Handle tuple-to-single assignments (in the case of a function or something)
				elif isinstance(targetNode, ast.Tuple):
					for left in targetNode.elts:
						if isinstance(left, ast.Name):
							foundAssignments.append((left.id, node.value))

				# Handle attribute assignments
				elif isinstance(targetNode, ast.Attribute):
					foundAssignments.append((f"{targetNode.value.id}.{targetNode.attr}", node.value))

				# Handle other assignment types
				else:
					foundAssignments.append((ast.dump(targetNode), node.value))

		# Return the list of found assignments
		return foundAssignments
	
	def getAssignmentOperations(self, assignmentList: List[tuple]) -> List[dict]:
		'''
		Analyze the list of assignment nodes and extract operation details.

		Parameters:
		- assignmentList: List of tuples containing (variable, assignment AST node)

		Returns:
		- List of dictionaries with operation details
		'''

		# Make sure the list has something in it
		if not assignmentList:
			raise ValueError("Cannot get operations from empty assignment list!")
		
		# Iterate over the list to check
		assignments = []
		for variable, assignment in assignmentList:

			# Handle binary operations
			if isinstance(assignment, ast.BinOp):

				# Get terms
				leftTerm = assignment.left.id if isinstance(assignment.left, ast.Name) else None
				rightTerm = assignment.right.id if isinstance(assignment.right, ast.Name) else None

				# Get the operation type
				op_type = type(assignment.op).__name__

				# Add the operation to the list
				assignments.append({
					"target": variable,
					"operation": op_type,
					"operands": (leftTerm, rightTerm)
				})

			# Handle literal assignments
			elif isinstance(assignment, ast.Constant):
				assignments.append({
					"target": variable,
					"operation": "Constant",
					"value": assignment.value
				})

			# Handle variable-to-variable assignments
			elif isinstance(assignment, ast.Name):
				assignments.append({
					"target": variable,
					"operation": "Copy",
					"source": assignment.id
				})

			# Handle function calls
			elif isinstance(assignment, ast.Call):
				func_name = assignment.func.id if isinstance(assignment.func, ast.Name) else None
				args = [arg.id if isinstance(arg, ast.Name) else None for arg in assignment.args]
				assignments.append({
					"target": variable,
					"operation": "FunctionCall",
					"function": func_name,
					"arguments": args
				})
			
			# Case 4: fallback (for debugging or unhandled cases)
			else:
				assignments.append({
					"target": variable,
					"operation": "Unknown",
					"ast": assignment
				})
		
		# Return the list of assignments with operations
		return assignments
	
	def buildAllFlows(self, operationsList: List[dict]) -> Dict[str, Any]:
		'''
		Build the flows from a list of operations.

		Parameters:
		- operationsList: List of dictionaries with operation details

		Returns:
		- Dictionary mapping source variables to target variables
		'''

		# Make sure the list has something in it
		if not operationsList:
			raise ValueError("Cannot build operations dict from empty operations list!")

		# Build flows by adding each element
		flows = {}
		for operation in operationsList:
			
			# Get target and type
			target = operation.get("target")
			operationType = operation.get("operation")

			# If we have a constant assignment, then we just add direct mapping
			if operationType == "Constant":
				flows[operation.get("value")] = target

			# If we have a binary operation, then we add all operands as mappings
			if operationType in ["Add", "Sub", "Mult", "Div", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv"]:
				for operand in operation.get("operands", []):
					if not operand:
						raise ValueError("Add operation missing operand!")
					flows[operand] = target

			# If we have a FunctionCall operation, then we add all arguments and parameters as mappings
			if operationType == "FunctionCall":

				# Go find the function definition in the AST
				functionDefinition = None
				for node in ast.walk(self.ast):
					if isinstance(node, ast.FunctionDef) and node.name == operation.get("function"):
						functionDefinition = node
						break

				# Use the function definition to get parameter names
				if not functionDefinition:
					raise ValueError(f"Could not find function definition for {operation.get('function')}!")
				parameters = [arg.arg for arg in functionDefinition.args.args]
	
				# Get arguments
				arguments = operation.get("arguments", [])
				if len(arguments) != len(parameters):
					raise ValueError(f"Function call argument count does not match parameter count for {operation.get('function')}!")
				
				# Map each argument to the parameter
				for argument, parameter in zip(arguments, parameters):
					flows[argument] = parameter
			
		return flows
	
	def buildAllPaths(self, flows: Dict[str, Any]) -> List[str]:
		'''
		Build all paths from the flows dictionary.
		'''

		# Make sure the flows dict has something in it
		if not flows:
			raise ValueError("Cannot build paths from empty flows dictionary!")
		
		# Create a path for each source in the flows
		paths = []
		visitedElements = set()
		for source in flows.keys():

			# Check if we've already seen this element as part of a path
			if source in visitedElements:
				continue

			# Start the path with the current key
			currentPath = [str(source)]
			while source in flows.keys():

				# Follow the flow to the next node
				source = flows.get(source)
				currentPath.append(str(source))

				# Mark this element as visited, so we don't start a new path from it
				visitedElements.add(source)

			# Add the completed path to the list
			paths.append(" -> ".join(currentPath))

		# Return the list of paths
		return paths

# Example usage of the DataFlowAnalyzer class on `calc.py`
if __name__ == "__main__":

	# Create the analyzer and run the analysis
	codeFilePath = "W6 - Data Flow Analysis/calc.py"
	analyzer = DataFlowAnalyzer(codeFilePath, autoExtractTree=True)

	# Get all assignments, operations, flows, and paths
	assignments = analyzer.getAllAssignmentNodes()
	operations = analyzer.getAssignmentOperations(assignments)
	flows = analyzer.buildAllFlows(operations)
	paths = analyzer.buildAllPaths(flows)
	
	# Print the results
	print(f"Paths for {codeFilePath}:")
	for path in paths:
		print(f"| {path}")
