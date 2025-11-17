'''
Author: Akond Rahman 
'''

import random 

'''
IMPORTS ADDED BY CHRIS HINKSON @CMH02

I have added the below imports for use in this assignment.
'''
import json
import tqdm

def divide(v1, v2):
	return v1/ v2 

def simpleFuzzer(): 
	
	# Make a list of negative and positive ints in increments of 100 for fuzzing
	intFuzz = [x for x in range(-10000, 10001, 100)]
	
	# Make copies of int list for floats
	floatFuzz = [float(x) for x in intFuzz]

	# Make infinite values for fuzzing
	floatFuzz.append(float('inf'))
	floatFuzz.append(float('-inf'))

	# Use the blns file to get strings for testing
	stringFuzz = []
	pathToBlnsFile = "W10 - Whitebox Fuzzing/blns.json"
	with open(pathToBlnsFile) as blnsFile:
		stringFuzz = json.load(blnsFile)


	# Iterate over all fuzz inputs and try (note that i use tqdm for a quick progress bar)
	successfulTests = 0
	failedTests = 0
	failedFuzzes = []
	failedFuzzErrorMessages = []

	for fuzzInput in tqdm.tqdm(iterable=intFuzz, desc='Fuzzing Ints', unit='inputs'):
		try:
			result = divide(fuzzInput, fuzzInput)
			successfulTests += 1
		except Exception as e:
			failedTests += 1
			failedFuzzes.append(fuzzInput)
			failedFuzzErrorMessages.append(str(e))

	for fuzzInput in tqdm.tqdm(iterable=floatFuzz, desc='Fuzzing Floats', unit='inputs'):
		try:
			result = divide(fuzzInput, fuzzInput)
			successfulTests += 1
		except Exception as e:
			failedTests += 1
			failedFuzzes.append(fuzzInput)
			failedFuzzErrorMessages.append(str(e))

	for fuzzInput in tqdm.tqdm(iterable=stringFuzz, desc='Fuzzing Strings', unit='inputs'):
		try:
			result = divide(fuzzInput, fuzzInput)
			successfulTests += 1
		except Exception as e:
			failedTests += 1
			failedFuzzes.append(fuzzInput)
			failedFuzzErrorMessages.append(str(e))

	# Print summary and save results to a .txt file
	print(f"Total successful tests: {successfulTests}")
	print(f"Total failed tests: {failedTests}")

	resultsFilePath = "W10 - Whitebox Fuzzing/fuzzing_results.txt"
	with open(resultsFilePath, 'w') as resultsFile:
		resultsFile.write(f"Fuzzing Results Summary\n")
		resultsFile.write(f"Total successful tests: {successfulTests}\n")
		resultsFile.write(f"Total failed tests: {failedTests}\n")

		resultsFile.write(f"\nFailed Fuzz Inputs and Error Messages:\n")
		for fuzzInput, fuzzErrorMessage in zip(failedFuzzes, failedFuzzErrorMessages):
			resultsFile.write(f"\nFuzz Input: {fuzzInput} \nProduced Error: {fuzzErrorMessage}\n")

if __name__=='__main__':
	simpleFuzzer()
