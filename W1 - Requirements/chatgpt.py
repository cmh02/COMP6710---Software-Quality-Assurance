'''
W1 - REQUIREMENTS - CHATGPT

Author: Chris Hinkson [@cmh02]
Class: COMP6710 - Software Quality Assurance

Assignment Instructions: https://github.com/paser-group/continuous-secsoft/tree/master/fall25-sqa/workshops/w1-req
'''

'''
MODULE IMPORTS
'''
import unittest
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu

'''
UTILITY FUNCTIONS
'''

# Note: This function is provided by Dr. Rahman
def getComparisonFromScipy(ls1, ls2):
	str2ret = ''
	u_statistic, p_value = mannwhitneyu(ls1, ls2, alternative='two-sided')
	print(f"U statistic: {u_statistic}")
	print(f"P-value: {p_value}")
	if p_value < 0.05: 
		str2ret = 'Significant'
	else:
		str2ret = 'Not-Significant'
	return str2ret

# Note: This function is provided by ChatGPT
def chatgpt1_performMannWhitneyU(group_1, group_2, alpha=0.05, alternative='two-sided'):
	"""
	Performs the Mann-Whitney U test for two independent groups.
	
	Parameters:
		group_1 (list or array): Data for the first group
		group_2 (list or array): Data for the second group
		alpha (float): Significance level for the test (default is 0.05)
		alternative (str): Specifies the type of test ('two-sided', 'less', or 'greater')
		
	Returns:
		tuple: U-statistic, p-value, and conclusion message
	"""
	
	# Perform the Mann-Whitney U test
	stat, p_value = mannwhitneyu(group_1, group_2, alternative=alternative)
	
	# Interpret the result
	if p_value < alpha:
		conclusion = "Reject the null hypothesis: The two groups are significantly different."
	else:
		conclusion = "Fail to reject the null hypothesis: The two groups are not significantly different."
	
	return stat, p_value, conclusion

# Wrapper function for chatgpt1_performMannWhitneyU to match the expected output
def getComparisonFromChatGPT1(ls1, ls2):
	str2ret = ''
	u_statistic, p_value, conclusion = chatgpt1_performMannWhitneyU(ls1, ls2, alternative='two-sided')
	print(f"U statistic: {u_statistic}")
	print(f"P-value: {p_value}")
	if p_value < 0.05: 
		str2ret = 'Significant'
	else:
		str2ret = 'Not-Significant'
	return str2ret

# Note: This function is provided by ChatGPT
def chatgpt2_performMannWhitneyU(sample1, sample2, alternative='two-sided'):
	"""
	Performs the Mann-Whitney U test for two independent samples.

	Parameters:
	- sample1: array-like, first sample of data
	- sample2: array-like, second sample of data
	- alternative: string, defines the alternative hypothesis ('two-sided', 'less', 'greater')

	Returns:
	- U-statistic
	- p-value
	"""
	
	# Ensure that input samples are numpy arrays for efficient operations
	sample1 = np.asarray(sample1)
	sample2 = np.asarray(sample2)

	# Perform the Mann-Whitney U test using scipy
	u_statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
	
	return u_statistic, p_value

# Wrapper function for chatgpt2_performMannWhitneyU to match the expected output
def getComparisonFromChatGPT2(ls1, ls2):
	str2ret = ''
	u_statistic, p_value = chatgpt2_performMannWhitneyU(ls1, ls2, alternative='two-sided')
	print(f"U statistic: {u_statistic}")
	print(f"P-value: {p_value}")
	if p_value < 0.05: 
		str2ret = 'Significant'
	else:
		str2ret = 'Not-Significant'
	return str2ret

'''
UNIT TESTS
'''

class TestGenAI(unittest.TestCase):
		
	# Test the class-provided code
	def test_ClassExample(self):
		x = [1, 1, 2, 3, 1, 1, 4]
		y = [6, 4, 7, 1, 3 , 7, 3, 7]
		self.assertEqual('Significant', getComparisonFromScipy(x, y))

	# Test the ChatGPT-provided code (version 1)
	def test_ChatGPT1Example(self):
		x = [1, 1, 2, 3, 1, 1, 4]
		y = [6, 4, 7, 1, 3 , 7, 3, 7]
		self.assertEqual('Significant', getComparisonFromChatGPT1(x, y))

	# Test the ChatGPT-provided code (version 2)
	def test_ChatGPT2Example(self):
		x = [1, 1, 2, 3, 1, 1, 4]
		y = [6, 4, 7, 1, 3 , 7, 3, 7]
		self.assertEqual('Significant', getComparisonFromChatGPT2(x, y))

if __name__ == '__main__':
	unittest.main()