from typing import Optional, Union


class Leaf:
    def __init__(self, label: str):
        self._label = label

    @property
    def label(self):
        return self._label

    def __str__(self):
        return self._label

    def __repr__(self):
        return f"Leaf({self._label})"


'''
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
'''


class Node:

    def __init__(self, feature: str, most_frequent_label: str):
        self.__feature = feature;
        self.__most_frequent_label = most_frequent_label
        self.__children: dict[str, Union[Node, Leaf]] = {}

    def add_child(self, branch_value: str, child):
        self.__children[branch_value] = child

    def children(self):
        yield from self.__children.items()

    def __iter__(self):
        yield from self.__children.values()

    @property
    def feature(self):
        return self.__feature

    @property
    def most_frequent_label(self):
        return self.__most_frequent_label

    # __str__ is kinda iffy

