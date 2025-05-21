import unittest
from neetcode75 import \
    two_sum_return_values, two_sum_return_indexes, two_sum_return_all_index_pairs, \
    valid_parentheses, \
    max_profit, \
    is_duplicate_with_set, is_duplicate_with_sort

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

class TestValidParentheses(unittest.TestCase):

    def test_valid_parentheses(self):
        self.assertEqual(valid_parentheses('()'), True)
        self.assertEqual(valid_parentheses('{()[]}'), True)
        self.assertEqual(valid_parentheses('{(]}'), False)
        self.assertEqual(valid_parentheses('([])'), True)

class TestMaxProfit(unittest.TestCase):

    def test_max_profit(self):
        self.assertEqual(max_profit([10,1,5,6,7,1]), 6)
        self.assertEqual(max_profit([10,8,7,5,2]), 0)

class TestIsDuplicate(unittest.TestCase):

    def test_is_duplicate_with_set(self):
        self.assertEqual(is_duplicate_with_set([]), False)
        self.assertEqual(is_duplicate_with_set([1,4,1]), True)
        self.assertEqual(is_duplicate_with_set([5,3,6]), False)

    def test_is_duplicate_with_sort(self):
        self.assertEqual(is_duplicate_with_sort([]), False)
        self.assertEqual(is_duplicate_with_sort([1,4,1]), True)
        self.assertEqual(is_duplicate_with_sort([5,3,6]), False)

if __name__ == '__main__':
    unittest.main()
