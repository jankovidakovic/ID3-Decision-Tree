import csv
import math
from typing import Union

from dataset import Dataset
from node import Node, Leaf


def load_dataset(dataset_path: str) -> Dataset:
    """parses the csv dataset at the given path into an instance of Dataset class

    :param dataset_path: path at which the dataset resides
    :return: an instance of Dataset class containing all the data
    """

    with open(dataset_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        dataset: Dataset
        for row in csv_reader:
            if dataset is None:  # first row of the csv file
                dataset = Dataset(row)
            else:
                dataset.add_example(row[:-1], row[-1])
        return dataset


def entropy(sample: list[str]) -> float:
    """Calculates the entropy of the given sample.

    Entropy is calculated with logarithm of base 2. Perfectly diverse sample, with equal number of
    occurrences of each class label, will have an entropy of 1. On the contrary, a sample with only
    a single class label will have an entropy of 0.
    
    :param sample: sample of the class label realizations
    :return: entropy of the sample
    """

    # calculate the probability of each example
    probs: dict[str, float] = {c: sample.count(c) / len(sample) for c in sample}
    e: float = 0  # entropy
    for probability in probs.values():
        probability: float

        e += probability * math.log2(probability)

    return -e


def accuracy(expected: list[str], actual: list[str]) -> float:
    """Computes the accuracy of the classification procedure, given expected and actual labels.

    Accuracy is calculated as the percentage of correctly classified examples.

    :param expected: sample of expected labels.
    :param actual: sample of actual labels obtained through some classification procedure
    :return: accuracy of classification
    """

    if len(expected) != len(actual):
        raise ValueError("Cannot compute accuracy if samples are not of the same length.")
    acc: float = 0
    for i, v in enumerate(actual):
        i: int
        v: str

        acc += 1 if expected[i] == v else 0
    acc /= len(actual)
    return acc


def format_branches(root: Union[Node, Leaf]) -> str:
    """Formats the branches of the given tree.

    Every path from the root of the tree to leaf nodes is stored in a string as a single line,
    in format of {distance_from_the_root}:{feature_name}={feature_value}. Nodes are whitespace-separated,
    and for leaf nodes only the label is stored in the string.

    :param root: root of the decision tree
    :return: string representation of all branches
    """

    if isinstance(root, Leaf):  # depth was limited to 0
        return root.label

    visited: dict[Union[Node, Leaf], bool] = {}  # tracks visited nodes
    curr_path: list[tuple[str, str]] = []  # (feature, branch_value), used to construct a single path
    format_str: str = "{}:{}={} "
    branches: list[str] = []  # paths within the tree

    def dfs(node: Union[Node, Leaf]):
        """Traverses the tree in a depth-wise manner.

        For each unvisited node, traverses the children nodes recursively, constructing the parts of the
        current path along the way. For each unvisited leaf, creates a string of the path from the root to
        the leaf and stores the string.

        :param node: current node for which the subtree is to be traversed
        """

        if node in visited:
            return  # only visited nodes
        visited[node] = True  # mark as visited
        if isinstance(node, Leaf):  # create output entry
            curr_branch: str = format_str * len(curr_path)
            format_list: list[tuple[int, str, str]] = [(i + 1, v[0], v[1]) for i, v in enumerate(curr_path)]
            curr_branch: str = curr_branch.format(*[i for tpl in format_list for i in tpl])  # this is pure magic
            curr_branch += node.label
            branches.append(curr_branch)  # store the path
        else:  # not a leaf -> inner node
            for feature_value, child_node in node.children():  # for each child
                feature_value: str
                child_node: Union[Node, Leaf]

                curr_path.append((node.feature, feature_value))  # temporary extend the path
                dfs(child_node)  # traverse subtree recursively
                curr_path.pop()  # remove from the path

    dfs(root)

    return "\n".join(branches)


def format_prediction_params(params: dict) -> str:
    """Formats the prediction parameters stored in the given dictionary.

    Creates a string representation containing branches of the trained decision tree,
    predictions on the test set, accuracy of prediction and the confusion matrix.

    :param params: prediction parameters. Expected to have the following elements:
        "predictions" - predicted labels
        "branches" - string representation of branches of the decision tree
        "accuracy" - accuracy of the predictions
        "confusion matrix" - confusion matrix
    :return: string representation in the form of
    [BRANCHES]:
    decision_tree_branches
    [PREDICTIONS]: predicted_labels
    [ACCURACY]: accuracy_to_5_decimal_places
    [CONFUSION_MATRIX]:
    space_separated_confusion_matrix
    """

    predictions: str = " ".join(p for p in params["predictions"])
    formatted_params: str = f"""[BRANCHES]:
{params["branches"]}
[PREDICTIONS]: {predictions}
[ACCURACY]: {params["accuracy"]:.5f}
[CONFUSION_MATRIX]: 
{params["confusion_matrix"]}
"""
    return formatted_params
