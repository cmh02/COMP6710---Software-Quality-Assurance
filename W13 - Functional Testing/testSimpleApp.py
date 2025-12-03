import unittest
import sys
import json
from simpleApp import app

class AppTestCase(unittest.TestCase):
	def setUp(self):
		self.client = app.test_client()

	def test_home_get(self):
		response = self.client.get('/')
		self.assertEqual(response.status_code, 200)
		self.assertIn(b"Welcome to a Simple Flask API!", response.data)

	def test_sqa(self):
		response = self.client.get('/sqa')
		self.assertEqual(response.status_code, 200)
		self.assertIn(b"Welcome to the SQA course!", response.data)
		
	'''
	BEGIN ADDITIONS BY CHRIS HINKSON @cmh02
	'''

	def test_SSP(self):
		response = self.client.get('/ssp')
		self.assertEqual(response.status_code, 200)
		self.assertIn(b"Secure Software Process", response.data)

	def test_VANITY(self):
		response = self.client.get('/vanity')
		self.assertEqual(response.status_code, 200)
		self.assertIn(b"Chris Hinkson", response.data)

	def test_MYPYTHON(self):
		response = self.client.get('/mypython')
		pythonVersion = sys.version
		self.assertEqual(response.status_code, 200)
		self.assertIn(pythonVersion.encode(), response.data)

	def test_CSSE(self):
		response = self.client.get('/csse')
		self.assertEqual(response.status_code, 200)
		self.assertIn(b"Department of Computer Science and Software Engineering", response.data)

	'''
	END ADDITIONS BY CHRIS HINKSON @cmh02
	'''

if __name__ == '__main__':
	unittest.main()