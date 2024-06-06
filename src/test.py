import unittest
from src import data_loaders

class TestDataLoaders(unittest.TestCase):
    def test_load_data(self):
        data = data_loaders.label_encoded_data()
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 4)

