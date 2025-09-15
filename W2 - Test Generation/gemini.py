'''
W2 - TEST GENERATION - GEMINI

Author: Chris Hinkson [@cmh02]
Class: COMP6710 - Software Quality Assurance

Assignment Instructions: https://github.com/paser-group/continuous-secsoft/tree/master/fall25-sqa/workshops/w2-testing
'''

'''
PROVIDED ASSIGNMENT IMPLEMENTATION CODE

The following code is provided as part of the assignment.

Development Note: This code had an undefined variable 'categ' in the print statement. I have created
a parameter 'categ' for the function to resolve this issue. I also added a return statement to return
the trend and p-value for testing purposes, as designated by Gemini's code production and the need
for unittest assertions.
'''
import pymannkendall as pmk
def understandTrends(data_, categ):
    trend, h, p, z, Tau, s, var_s, slope, intercept = pmk.original_test(data_)
    print('For {} trend is {} and p-value is {}'.format(categ, trend, p))
    return trend, p

'''
GEMINI GENERATED TEST CASES

Development Note: Upon giving Gemini the initial prompt, all that it gave me were arrays for test cases, not
the code itself. I had to give it an additional prompt "I need you to generate them in entirety as runnable 
code, not just arrays." in order to obtain actual test code. It then generated all of the following code.
'''
import numpy as np
import unittest
import io
import sys

class TestUnderstandTrends(unittest.TestCase):
    
    def setUp(self):
        # Redirect standard output to a string buffer to capture print statements
        self.held_output = io.StringIO()
        sys.stdout = self.held_output

    def tearDown(self):
        # Restore standard output
        self.held_output.close()
        sys.stdout = sys.__stdout__

    # 1. Positive Trend
    def test_positive_trend(self):
        """Tests a dataset with a clear positive trend."""
        data_positive = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        trend, p_value = understandTrends(data_positive, "PositiveTrend")
        self.assertEqual(trend, 'increasing')
        self.assertLess(p_value, 0.05)
        
    # 2. Negative Trend
    def test_negative_trend(self):
        """Tests a dataset with a clear negative trend."""
        data_negative = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        trend, p_value = understandTrends(data_negative, "NegativeTrend")
        self.assertEqual(trend, 'decreasing')
        self.assertLess(p_value, 0.05)

    # 3. No Trend
    def test_no_trend(self):
        """Tests a random dataset with no significant trend."""
        data_no_trend = [2, 5, 1, 8, 3, 6, 9, 4, 7, 10]
        trend, p_value = understandTrends(data_no_trend, "NoTrend")
        self.assertEqual(trend, 'no trend')
        self.assertGreater(p_value, 0.05)

    # 4. Data with Duplicates
    def test_duplicates(self):
        """Tests a dataset containing duplicate values."""
        data_duplicates = [1, 2, 2, 3, 4, 4, 5, 6, 6, 7]
        trend, p_value = understandTrends(data_duplicates, "Duplicates")
        # For this specific data, the trend should still be increasing
        self.assertEqual(trend, 'increasing')
        self.assertLess(p_value, 0.05)

    # 5. Small Dataset
    def test_small_dataset(self):
        """Tests a very small dataset."""
        data_small = [1, 5, 2, 8, 3]
        trend, p_value = understandTrends(data_small, "SmallDataset")
        # A small dataset may not have a significant trend, but should still produce a result
        self.assertIn(trend, ['increasing', 'decreasing', 'no trend'])
        
    # 6. Long Dataset
    def test_long_dataset(self):
        """Tests a long dataset with a clear trend."""
        data_long = list(range(1, 101))
        trend, p_value = understandTrends(data_long, "LongDataset")
        self.assertEqual(trend, 'increasing')
        self.assertLess(p_value, 0.05)

    # 7. Data with Missing Values
    def test_missing_values(self):
        """Tests a dataset containing missing values (NaN)."""
        data_missing = [1, 2, 3, np.nan, 5, 6, 7, 8, np.nan, 10]
        trend, p_value = understandTrends(data_missing, "MissingValues")
        # The library handles NaNs by skipping them; the trend should still be increasing
        self.assertEqual(trend, 'increasing')
        self.assertLess(p_value, 0.05)

    # 8. Data with Outliers
    def test_outliers(self):
        """Tests a dataset with a clear trend but containing outliers."""
        data_outliers = [1, 2, 3, 100, 5, 6, 7, 8, 9, 10]
        trend, p_value = understandTrends(data_outliers, "Outliers")
        self.assertEqual(trend, 'increasing')
        self.assertLess(p_value, 0.05)

    # 9. Constant Values
    def test_constant_values(self):
        """Tests a dataset where all values are the same."""
        data_constant = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        trend, p_value = understandTrends(data_constant, "ConstantValues")
        self.assertEqual(trend, 'no trend')
        self.assertGreater(p_value, 0.05)

    # 10. Mixed Positive and Negative Slopes
    def test_mixed_slopes(self):
        """Tests a dataset with both increasing and decreasing values."""
        data_mixed = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        trend, p_value = understandTrends(data_mixed, "MixedSlopes")
        self.assertEqual(trend, 'no trend')
        self.assertGreater(p_value, 0.05)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)