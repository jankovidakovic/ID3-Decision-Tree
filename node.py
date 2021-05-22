from dataset import Dataset
from typing import Optional, Union


class Leaf:
    def __init__(self, parent, label: str):
        self._parent: Node = parent
        self._label = label

    @property
    def parent(self):
        return self._parent

    @property
    def label(self):
        return self._label

    def __str__(self):
        branch_string: str = ""
        node: Node = self._parent
        level: int = 1
        while not node.is_root():
            branch_string = "{}:" + str(node) + " " + branch_string
            node = node.parent
            level += 1
        return branch_string.format(*[_ for _ in range(1, level+1)]) + self._label

    def __repr__(self):
        return self.__str__()


class Node:

    def __init__(self, parent, dataset: Dataset, parent_feature_value: Optional[str]):
        self._dataset: Optional[Dataset] = dataset
        # TODO - hide the dataset, exposing only needed methods to clients
        self._parent: Node = parent
        self._children: list[Union[Leaf, Node]] = []
        self._parent_feature_value: Optional[str] = parent_feature_value
        self._subtree_feature_name: Optional[str] = None

    @staticmethod
    def Root(dataset: Dataset):
        return Node(None, dataset, None)

    def is_root(self):
        return self._parent is None and self._parent_feature_value is None

    @property
    def subtree_feature_name(self):
        return self._subtree_feature_name

    @property
    def parent_feature_value(self):
        return self._parent_feature_value

    @property
    def dataset(self):
        return self._dataset

    @property
    def parent(self):
        return self._parent

    def add_child(self, child):
        self._children.append(child)

    def __next__(self):
        return self.__iter__().__next__()

    def __iter__(self):
        return self._children.__iter__()

    @subtree_feature_name.setter
    def subtree_feature_name(self, value):
        self._subtree_feature_name = value

    def __str__(self):
        if self.is_root():
            return ""
        else:
            return self._parent._subtree_feature_name + "=" + self._parent_feature_value

    def __repr__(self):
        return self.__str__()
