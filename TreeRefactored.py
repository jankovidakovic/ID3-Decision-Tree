from typing import Optional, Union


class Leaf:
    # TODO - make a dataclass
    def __init__(self, label: str):
        self._label = label

    # taking the parent out forces an independent implementation of the str method
    @property
    def label(self):
        return self._label


class Node:
    # TODO - make a dataclass
    def __init__(self, feature_name: str, feature_value: str, most_frequent_value: str):
        self.__feature_name = feature_name
        self.__feature_value = feature_value
        self.__most_frequent_label = most_frequent_value
        self.__children: list[Union[Node, Leaf]] = []

    @property
    def feature_name(self):
        return self.__feature_name

    @feature_name.setter
    def feature_name(self, value: str):
        self.__feature_name = value

    @property
    def feature_value(self):
        return self.__feature_value

    @feature_value.setter
    def feature_value(self, value: str):
        self.__feature_value = value

    def __iter__(self):
        yield from self.__children

    def __str__(self):
        if self.__feature_name is None or self.__feature_value is None:
            return ""
        return f"{self.__feature_name}={self.__feature_value}"

    def add_child(self, child):
        self.__children.append(child)

    @property
    def most_frequent_label(self):
        return self.__most_frequent_label

# values that the node holds:
#   feature that the children are split by
#       -> kinda doesn't make sense for node to know this tho

# node:
#   if leaf:
#       value
#   if root:
#       mdf
#   if inner node:
#       mdf[parent] = None if root else mdf[parent]
#       value = None if root else value
#       mdf

# A NODE SHOULD CORRESPOND TO THE FEATURE
# ONLY LEAVES HAVE VALUES
# BRANCHES STORE VALUES
class NewNode:
    def __init__(self, feature_name: str):
        self.__feature_name = feature_name
        self.__children: dict = {}

    def add_child(self, child, branch_value):
        self.__children[branch_value] = child






