from typing import Union, Optional

from confusion_matrix import ConfusionMatrix
from dataset import Dataset
from node import Node, Leaf
from utils import accuracy, format_branches


class DecisionTree:
    """Model of a decision tree. It can be trained with some training dataset and uses the ID3 algorithm
    for constructing the tree structure. Then, a test dataset can be provided to obtain predictions of
    class labels, according to the trained tree structure.

    """

    def __init__(self, max_depth: Optional[int] = None):
        """Initializes the decision tree with the given maximum depth.

        :param max_depth: Maximum depth of the decision tree that can be reached during the training procedure.
        All ambiguities that aren't clearly branched at the point of maximum depth will be resolved with a leaf
        node that stores the most frequent value of the classification label for the subset of the dataset
        that corresponds to the last node before the maximum depth limit.
        """

        self.__max_depth: int = max_depth
        self.__root: Optional[Node] = None
        self.__branches: Optional[str] = None

    def fit(self, data: Dataset):
        """Learns the classification procedure on the provided dataset.

        Constructs the decision tree using the ID3 algorithm and stores the tree for future predictions.

        :param data: dataset that the decision tree will be fitted to
        :return: an instance of self
        """

        self.__root: Union[Node, Leaf] = self.__id3(data, data, 0)
        self.__branches: str = format_branches(self.__root)

        return self

    def __id3(self, dataset: Dataset, parent_dataset: Dataset, depth: int) -> Union[Node, Leaf]:
        """Performs the ID3 machine learning algorithm and returns the constructed decision tree.

        ID3 algorithm constructs the decision tree in which each node corresponds to some feature, and
        branches leading to the child nodes correspond to distinct values of the feature in the provided
        dataset. When constructing a new node, the dataset is grouped by its most discriminatory feature,
        which provides most information gain on the subsequent split. Leaf nodes of the tree correspond
        to class labels. if there is no depth limit, the id3 algorithm is guaranteed to construct a tree
        which perfectly classifies all examples from the given dataset, provided there are no examples
        with equal features but different labels.

        :param dataset: dataset for which the subtree of the decision tree is constructed.
        :param parent_dataset: dataset of the parent node in the decision tree. If the current execution of the
        method is the first one, parent dataset should be equal to the training dataset (root node has no parent)
        :param depth: depth of the node to be constructed. If called on the whole training dataset, depth should be 0
        :return: instance of Node corresponding to the root of the decision tree for the given dataset. Alternatively,
                returns a Leaf node with predicted class label if the depth limit is reached, or the dataset is
                empty, or all examples in the dataset are classified with the same value.
        """

        if self.__max_depth is None or depth < self.__max_depth:  # depth limit not reached
            if len(dataset) == 0:  # empty dataset
                return Leaf(parent_dataset.most_frequent_label)  # most frequent label of the parent
            elif len(dataset.label_space) == 1 or len(dataset.feature_names) == 0:
                # all examples have the same label or there is no features left in the dataset
                return Leaf(dataset.most_frequent_label)
            else:
                mdf: str = dataset.most_discriminatory_feature
                sub_datasets: dict[str, Dataset] = dataset.group_by_feature(mdf)
                node: Node = Node(mdf, dataset.most_frequent_label)
                # for each child, create the corresponding decision subtree
                for feature_value, sub_dataset in sub_datasets.items():
                    feature_value: str
                    sub_dataset: Dataset

                    child_node: Union[Node, Leaf] = self.__id3(sub_dataset, dataset, depth + 1)  # create the subtree
                    node.add_child(feature_value, child_node)  # add as a child
                return node
        else:  # depth limit reached
            return Leaf(dataset.most_frequent_label)

    def predict(self, dataset: Dataset) -> dict[str, Union[str, list[str], float, ConfusionMatrix]]:
        """predicts the class labels of the given test set, based on a previously fitted model.

        :param dataset: dataset consisting of
        :return: list of predicted values
        """

        prediction_params: dict[str, Union[str, list[str], float, ConfusionMatrix]] = {"branches": self.__branches}
        predicted_values: list[str] = []
        for example, label in dataset:
            example: dict[str, str]
            label: str

            # predict the example
            predicted_values.append(self.__label_example(example, self.__root))

        prediction_params["predictions"]: list[str] = predicted_values
        prediction_params["accuracy"] = accuracy(dataset.label_sample, predicted_values)
        prediction_params["confusion_matrix"]: ConfusionMatrix = ConfusionMatrix(
                dataset.label_space, dataset.label_sample, predicted_values)

        return prediction_params

    def __label_example(self, example: dict[str, str], node: Union[Node, Leaf]) -> str:
        """Labels the given example using the previously trained decision tree.

        Decision tree traversal procedure follows along the branches which feature values match the ones
        in the given example. If some feature in the given example has a value that was unseen in the
        training procedure, the example is classified with the most frequent value occurring in the leaves
        of the subtree.

        :param example: example to be classified
        :param node: node from which the decision tree is traversed for classification. Classification
                procedure should start from the root of the decision tree.
        :return: predicted class label
        """

        if isinstance(node, Leaf):
            return node.label
        for branch_value, child_node in node.children():
            branch_value: str
            child_node: Union[Node, Leaf]

            if node.feature in example and example[node.feature] == branch_value:
                return self.__label_example(example, child_node)
        # no child nodes correspond to the observed example -> unseen value
        return node.most_frequent_label
