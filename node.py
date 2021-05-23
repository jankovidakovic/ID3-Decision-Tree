from typing import Union, ItemsView


class Leaf:
    """Leaf of the decision tree which stores the class label

    """

    def __init__(self, label):
        """Creates a new Leaf that stores the given label

        :param label: label to be stored within the leaf
        """

        self.__label = label

    @property
    def label(self):
        return self.__label

    def __str__(self):
        return self.__label

    def __repr__(self):
        return f"Leaf({self.__label})"


class Node:
    """Node of the decision tree. Stores decision information and children nodes.

    Feature of the node corresponds to the most discriminatory feature of the dataset on which the node
    was constructed. Most frequent label of the dataset is also stored, and should be used in prediction
    for examples for which the feature of this node has some value unseen in the training process.
    Child nodes are indexed by the value of the feature that the node corresponds to.

    """

    def __init__(self, feature, most_frequent_label):
        """Constructs a new Node that splits some dataset by the given feature.

        :param feature: Feature that the node splits the dataset by
        :param most_frequent_label: most frequent classification label of the dataset
        """

        self.__feature = feature
        self.__most_frequent_label = most_frequent_label
        self.__children = {}  # indexed by feature values

    def add_child(self, branch_value, child):
        """Adds the child node.

        Child node is stored together with the branch value, which indicates value of the current node's feature
        in the examples of the dataset that the subtree corresponds to.

        :param branch_value: value of the node's feature
        :param child: instance of Node corresponding to the decision subtree
        """

        self.__children[branch_value] = child

    def children(self):
        """Provides a way to iterate over the subtrees(child nodes) and their corresponding feature values.

        :return: Iterable of 2-tuples containing feature value and child node
        """

        yield from self.__children.items()

    @property
    def feature(self):
        return self.__feature

    @property
    def most_frequent_label(self):
        return self.__most_frequent_label
