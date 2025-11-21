'''
Author: Chris Hinkson @cmh02
Class: COMP4710 - Software Quality Assurance
Workshop 12 - Pairwise Testing
'''

# Module Imports
import os
import tqdm
import subprocess
import allpairspy

if __name__ == "__main__":

	# Define each of the bandit parameters with their values
	banditParameters_A = ["file", "vuln"]
	banditParameters_L = ["all", "low", "medium", "high"]
	banditParameters_I = ["all", "low", "medium", "high"]
	banditParameters_F = ["csv", "custom", "html", "json", "screen", "txt", "xml", "yaml"]

	# Combine parameter lists
	banditParameters = [banditParameters_A, banditParameters_L, banditParameters_I, banditParameters_F]

	# Use allpairs to generate pairwise combinations
	pairTests = [pair for i, pair in enumerate(allpairspy.AllPairs(banditParameters))]

	# Save all generated tests to a file
	with open("allPairsTestCases.txt", "w") as testFile:
		for i, pairs in tqdm.tqdm(iterable=enumerate(pairTests), total=len(pairTests), desc="Writing Test Cases to File", unit="test cases"):
			testFile.write(f"{i + 1}: {pairs}\n")

	# Make sure bandit output directory exists
	os.makedirs("banditoutput", exist_ok=True)

	# Run bandit with each generated test case
	for i, test in tqdm.tqdm(iterable=enumerate(pairTests), total=len(pairTests), desc="Running Bandit Tests", unit="test cases"):

		# Make sure parameters are set and valid
		if test[0] not in banditParameters_A:
			raise Exception(f"Invalid parameter for -a during test {i+1}: {test[0]}")
		if test[1] not in banditParameters_L:
			raise Exception(f"Invalid parameter for -l during test {i+1}: {test[1]}")
		if test[2] not in banditParameters_I:
			raise Exception(f"Invalid parameter for -i during test {i+1}: {test[2]}")
		if test[3] not in banditParameters_F:
			raise Exception(f"Invalid parameter for -f during test {i+1}: {test[3]}")

		# Make the command to run bandit 
		# NOTE: Bandit was being weird about flags, so I had to use full names in the command for -i and -l)
		command = [
			"bandit",
			"-r",
			"detr-main",
			"-a", test[0],
			"--severity-level", test[1],
			"--confidence-level", test[2],
			"-f", test[3]
		]
		
		# Actually run bandit and save output to file
		with open(os.path.join("banditoutput", f"testcase_{i+1}.txt"), "w") as outputFile:
			subprocess.run(command, stdout=outputFile, stderr=subprocess.STDOUT)

	# Go through and concat all the test case outputs into a single file for canvas deliverable
	with open("allbanditoutputs.txt", "w") as allOutputsFile:
		for i in tqdm.tqdm(iterable=range(len(pairTests)), total=len(pairTests), desc="Concatenating Bandit Outputs", unit="test cases"):
			with open(os.path.join("banditoutput", f"testcase_{i+1}.txt"), "r") as individualOutputFile:
				allOutputsFile.write("\n========================================\n")
				allOutputsFile.write(f"OUTPUT FOR TEST CASE {i+1}\n")
				allOutputsFile.write("========================================\n\n")
				allOutputsFile.write(individualOutputFile.read())
				allOutputsFile.write("\n\n")