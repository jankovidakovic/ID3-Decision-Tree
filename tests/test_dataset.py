import math
from unittest import TestCase

from utils import load_dataset


def simple_entropy(*values):
    return -sum([(value / sum(values)) * math.log2(value / sum(values)) for value in values])


class TestDataset(TestCase):

    def test_group_by_feature(self):
        dataset = load_dataset("../datasets/volleyball.csv")
        print(dataset.group_by_feature("temperature"))

    def test_filter_by_feature(self):
        dataset = load_dataset("../datasets/volleyball.csv")
        print(dataset.filter_by_feature("temperature", "cold"))

    def test_entropy(self):
        dataset = load_dataset("../datasets/volleyball.csv")

    def test_information_gain(self):
        dataset = load_dataset("../datasets/volleyball.csv")
        print(dataset.information_gain("humidity"))

    def test_second_level_ig(self):
        dataset = load_dataset("../datasets/volleyball.csv")
        data_subset = dataset.filter_by_feature("weather", "sunny")
        print(data_subset.information_gain("temperature"))
        print(data_subset.information_gain("humidity"))
        print(data_subset.information_gain("wind"))

    def test_iter(self):
        dataset = load_dataset("../datasets/volleyball.csv")
        for feature, label in dataset.items():
            print(str(feature) + "=" + label)



