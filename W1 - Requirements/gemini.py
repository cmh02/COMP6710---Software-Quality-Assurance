'''
W1 - REQUIREMENTS - GEMINI

Author: Chris Hinkson [@cmh02]
Class: COMP6710 - Software Quality Assurance

Assignment Instructions: https://github.com/paser-group/continuous-secsoft/tree/master/fall25-sqa/workshops/w1-req
'''

'''
MODULE IMPORTS

Below are the four needed packages:
-> unittest: Provides unit testing framework
-> pandas: For data loading and storage
-> numpy: For data storage
-> scipy: For mann-whitney u
'''
import unittest
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

'''
DR. RAHMAN'S MANN-WHITNEY U IMPLEMENTATION

The below function is provided along with this assignment on Dr. Rahman's github.
'''
# Note: This function is provided by Dr. Rahman
def getComparisonFromDrRahman(ls1, ls2):
	str2ret = ''
	u_statistic, p_value = mannwhitneyu(ls1, ls2, alternative='two-sided')
	print(f"U statistic: {u_statistic}")
	print(f"P-value: {p_value}")
	if p_value < 0.05: 
		str2ret = 'Significant'
	else:
		str2ret = 'Not-Significant'
	return str2ret

'''
GEMINI MANN-WHITNEY U IMPLEMENTATION

The below function is Google Gemini's implementation of the Mann-Whitney U test. I also provide a small wrapper for the
generated function to match Dr. Rahman's expected output.
'''
# Note: This function is provided by Google Gemini (AI Model)
def mann_whitney_u_test(sample1, sample2, alternative='two-sided'):
    """
    Performs the Mann-Whitney U test on two independent samples.

    The Mann-Whitney U test (also known as the Wilcoxon rank-sum test)
    is a non-parametric statistical hypothesis test used to compare two 
    independent samples to determine if they come from the same distribution. 
    It's an alternative to the independent samples t-test when the data 
    is not normally distributed.

    Args:
        sample1 (list or numpy.ndarray): The first sample.
        sample2 (list or numpy.ndarray): The second sample.
        alternative (str): The alternative hypothesis to test. 
                           Can be 'two-sided', 'less', or 'greater'.
                           - 'two-sided': The distributions are not equal.
                           - 'less': The distribution of the first sample is 
                                     stochastically less than the second.
                           - 'greater': The distribution of the first sample is 
                                        stochastically greater than the second.

    Returns:
        tuple: A tuple containing the U statistic and the p-value.
    
    Raises:
        ValueError: If either sample is empty or contains non-numeric data.
    """
    # Convert samples to numpy arrays for efficient processing
    try:
        sample1 = np.array(sample1, dtype=float)
        sample2 = np.array(sample2, dtype=float)
    except ValueError:
        raise ValueError("Samples must contain only numeric data.")

    if not sample1.size or not sample2.size:
        raise ValueError("Samples cannot be empty.")

    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(sample1, sample2, alternative=alternative)
    
    return u_statistic, p_value

# Wrapper function for mann_whitney_u_test to match the expected output
def getComparisonFromGemini(ls1, ls2):
	str2ret = ''
	u_statistic, p_value = mann_whitney_u_test(ls1, ls2, alternative='two-sided')
	print(f"U statistic: {u_statistic}")
	print(f"P-value: {p_value}")
	if p_value < 0.05: 
		str2ret = 'Significant'
	else:
		str2ret = 'Not-Significant'
	return str2ret

'''
UNIT TESTS

This class (adopting the same signature as the one provided by Dr. Rahman) is designed to provide test cases for testing
Google Gemini's implementation of the Mann-Whitney U test. It also provides a test case for the implementation provided
by Dr. Rahman on his github. 

For the tests to function properly, the provided datafile ('perf-data.csv') must be in the same directory as this file.
'''
class TestGenAI(unittest.TestCase):

	# Test Setup
	def setUp(self):

		# Load the data into pandas dataframe
		self.df = pd.read_csv('perf-data.csv', usecols=['A', 'B'])
		self.A = self.df['A'].to_numpy()
		self.B = self.df['B'].to_numpy()

		# Print data and metrics for verification
		print(f"-------------------- UNIT TEST SETUP ---------------------------------")
		print(f"Loaded {len(self.df)} rows from perf-data.csv!")
		print(f"-> Dataset A: {self.A[:5]} ... {self.A[-5:]}")
		print(f"-> Dataset B: {self.B[:5]} ... {self.B[-5:]}")
		print()
		print(f"Dataset Metrics:")
		print(f"-> Length A: {len(self.A)}, Length B: {len(self.B)}")
		print(f"-> Mean A: {np.mean(self.A)}, Mean B: {np.mean(self.B)}")
		print(f"-> Median A: {np.median(self.A)}, Median B: {np.median(self.B)}")
		print(f"-> StdDev A: {np.std(self.A)}, StdDev B: {np.std(self.B)}")
		print(f"----------------------------------------------------------------------")

	# Test Case 1: Class Example
	def test_ClassExample(self):
		self.assertEqual('Not-Significant', getComparisonFromDrRahman(self.A, self.B))

	# Test Case 2: Gemini Example
	def test_GeminiExample(self):
		self.assertEqual('Not-Significant', getComparisonFromGemini(self.A, self.B))

'''
PROGRAM EXECUTION

The below code will run the unit tests if this file is executed directly.
'''
if __name__ == '__main__':
	unittest.main()