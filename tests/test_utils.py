import math
from unittest import TestCase

from utils import load_dataset, entropy


class Test(TestCase):
    def test_load_dataset(self):
        print(load_dataset("../datasets/volleyball.csv"))

    def test_entropy(self):
        D = ["yes", "yes", "no", "yes", "no"]
        K = ["yes", "no", "maybe"]
        expected_entropy = -((2/5) * math.log2(2/5) + (3/5) * math.log2(3/5))
        actual_entropy = entropy(D, K)
        self.assertEqual(expected_entropy, actual_entropy)
