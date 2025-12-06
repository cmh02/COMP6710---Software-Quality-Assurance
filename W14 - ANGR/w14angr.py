'''
Author: Chris Hinkson @cmh02
Class: COMP5710 - Software Quality Assurance
Workshop 14 - ANGR
'''

'''
MODULE IMPORTS

This section will import any modules for w14.
'''

# System
import os
import tarfile

# ANGR
import angr

# Display
from tqdm import tqdm


'''
TAR FILE EXTRACTION
	
This section will extract all of the binaries from the provided tar file.
'''

# Extract all files from the provided tar file
binaryCompressedPath = "W14 - ANGR/binaries.tar.gz"
binaryExtractedPath = "W14 - ANGR/binaries"
with tarfile.open(binaryCompressedPath, "r:gz") as tarFile:
	tarFile.extractall(path=os.path.dirname(binaryExtractedPath))

# Print number of files extracted
print(f"Extracted {len(os.listdir(binaryExtractedPath))} files to the {binaryExtractedPath}/ directory.")

'''
OUTPUT UTILITY

This section makes a quick utility for writing output to multiple files.
'''
def writeToFiles(text: str, fileHandlers: list):
	for fileHandler in fileHandlers:
		fileHandler.write(text)

'''
BINARY ANALYSIS

This section will get the information for each binary file.
'''

# Define output path for writing results
analysisOutputPath = "W14 - ANGR/output/binary_analysis_output.txt"
os.makedirs(os.path.dirname(analysisOutputPath), exist_ok=True)

# Open output file for writing
with open(analysisOutputPath, "w") as outputFileGlobal:

	# Print file header
	outputFileGlobal.write("W14 Binary Analysis Output\n")
	outputFileGlobal.write("Author: Chris Hinkson @cmh02\n")
	outputFileGlobal.write("============================\n")

	# Iterate over binaries with tqdm
	for binaryFile in tqdm(os.listdir(binaryExtractedPath), desc="Analyzing Binaries", unit="files"):
		
		# Get full path to binary
		binaryPath = os.path.join(binaryExtractedPath, binaryFile)
		if not os.path.isfile(binaryPath):
			print(f"Skipping non-file: {binaryPath}")
			continue

		# Load binary with ANGR
		project = angr.Project(thing=binaryPath, auto_load_libs=False)

		# Get the control flow graph
		cfg = project.analyses.CFGFast()

		# Write analysis to output file
		individualOutputFilePath = os.path.join(os.path.dirname(analysisOutputPath), f"individualoutput", f"{binaryFile}.txt")
		os.makedirs(os.path.dirname(individualOutputFilePath), exist_ok=True)
		with open(individualOutputFilePath, "w") as individualOutputFile:

			# Combine file handlers for writing output
			allFileHandlers = [outputFileGlobal, individualOutputFile]

			# Write analysis results
			writeToFiles(f"\n\n===============================\n", allFileHandlers)
			writeToFiles(f"\nAnalysis for Binary: {binaryPath}\n", allFileHandlers)
			writeToFiles(f"Main Object: {project.loader.main_object}\n", allFileHandlers)
			writeToFiles(f"Number of Nodes: {len(cfg.graph.nodes)}\n", allFileHandlers)
			writeToFiles(f"Number of Edges: {len(cfg.graph.edges)}\n", allFileHandlers)

			# Iterate over nodes and print out cfg
			writeToFiles(f"\n\nControl Flow Graph ({cfg.model.graph}):\n", allFileHandlers)
			for node in cfg.graph.nodes:
				writeToFiles(f"\tNode: {node} [Name: {node.name} | Size: {node.size} | Memory Range: {node.addr} - {node.addr + node.size}]\n", allFileHandlers)
				for succ in node.successors:
					writeToFiles(f"\t\tSuccessor: {succ} [Name: {succ.name} | Size: {succ.size} | Memory Range: {succ.addr} - {succ.addr + succ.size}\n", allFileHandlers)