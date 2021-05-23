from typing import Union, ItemsView


class Leaf:
    """Leaf of the decision tree which stores the class label

    """

    def __init__(self, label: str) -> None:
        """Creates a new Leaf that stores the given label

        :param label: label to be stored within the leaf
        """

        self.__label: str = label

    @property
    def label(self) -> str:
        return self.__label

    def __str__(self) -> str:
        return self.__label

    def __repr__(self) -> str:
        return f"Leaf({self.__label})"


class Node:
    """Node of the decision tree. Stores decision information and children nodes.

    Feature of the node corresponds to the most discriminatory feature of the dataset on which the node
    was constructed. Most frequent label of the dataset is also stored, and should be used in prediction
    for examples for which the feature of this node has some value unseen in the training process.
    Child nodes are indexed by the value of the feature that the node corresponds to.

    """

    def __init__(self, feature: str, most_frequent_label: str) -> None:
        """Constructs a new Node that splits some dataset by the given feature.

        :param feature: Feature that the node splits the dataset by
        :param most_frequent_label: most frequent classification label of the dataset
        """

        self.__feature: str = feature
        self.__most_frequent_label: str = most_frequent_label
        self.__children: dict[str, Union[Node, Leaf]] = {}  # indexed by feature values

    def add_child(self, branch_value: str, child) -> None:
        """Adds the child node.

        Child node is stored together with the branch value, which indicates value of the current node's feature
        in the examples of the dataset that the subtree corresponds to.

        :param branch_value: value of the node's feature
        :param child: instance of Node corresponding to the decision subtree
        """

        self.__children[branch_value]: Union[Node, Leaf] = child

    def children(self) -> ItemsView:
        """Provides a way to iterate over the subtrees(child nodes) and their corresponding feature values.

        :return: Iterable of 2-tuples containing feature value and child node
        """

        yield from self.__children.items()

    @property
    def feature(self) -> str:
        return self.__feature

    @property
    def most_frequent_label(self) -> str:
        return self.__most_frequent_label
