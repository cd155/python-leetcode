import unittest
from neetcode75 import twoSumNeetcode, twoSum

class TestTwoSum(unittest.TestCase):

    def test_twoSumNeetcode_basic(self):
        self.assertEqual(twoSumNeetcode([2,7,11,15], 9) , (2, 7))

if __name__ == '__main__':
    unittest.main()
