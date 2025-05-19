import unittest
from neetcode75 import two_sum_return_values, two_sum_return_indexes, two_sum_return_all_index_pairs

class TestTwoSum(unittest.TestCase):

    def test_two_sum_return_indexes(self):
        self.assertEqual(two_sum_return_indexes([2,7,11,15], 9) , (0, 1))
        self.assertEqual(two_sum_return_indexes([2,7,11,15,1,1,3], 2) , (4, 5))
        self.assertEqual(two_sum_return_indexes([], 9) , None)
        self.assertEqual(two_sum_return_indexes([2], 9) , None)
        self.assertEqual(two_sum_return_indexes([2,7,11,15], 10) , None)

    def test_two_sum_exclusive_pairs(self):
        self.assertEqual(two_sum_return_all_index_pairs([4,4,8,10,0,8,4], 8), [[0,1], [2,4], [4,5],[0,6]])

    def test_two_sum_values(self):
        self.assertEqual(two_sum_return_values([2,7,11,15], 9) , (2, 7))

if __name__ == '__main__':
    unittest.main()
