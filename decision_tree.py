from typing import Union, Optional

from confusion_matrix import ConfusionMatrix
from dataset import Dataset
from node import Node, Leaf


class DecisionTree:
    def __init__(self, max_depth: Optional[int] = None):
        self._max_depth = max_depth
        # initialize with random parameters
        self._root: Optional[Node] = None
        self._branches: str = ""

    def fit(self, data: Dataset):
        """performs model learning on the given dataset.

        :param data: dataset that the decision tree will be fitted to
        :return: an instance of self
        """

        # self._root = Node.Root(data)
        # self.__id3(self._root, 0)

        self._root = self.__id3_returnsnode(data, data, 0)

        self.__save_fit(self._root)

        return self

    def __save_fit(self, node: Union[Node, Leaf]) -> None:
        if isinstance(node, Leaf):
            self._branches += str(node) + "\n"
        else:
            for child in node:  # TODO - check if something breaks
                self.__save_fit(child)

    def __id3(self, node: Node, depth: int):
        # TODO - make it return node
        if self._max_depth is None or depth < self._max_depth:
            if len(node.dataset) == 0:  # empty dataset
                # return the leaf node containing the most frequent label of the parent
                node.add_child(Leaf(node, node.parent.dataset.most_frequent_label()))
            elif len(node.dataset.label_space) == 1 or len(node.dataset.feature_names) == 0:
                # if there is no features in the dataset, it consists of only the labels
                node.add_child(Leaf(node, node.dataset.most_frequent_label()))
            else:
                # choose the most discriminatory feature
                most_discriminatory_feature: str = node.dataset.most_discriminatory_feature()
                node.subtree_feature_name = most_discriminatory_feature
                # split the dataset by the given feature
                sub_datasets: dict[str, Dataset] = node.dataset.group_by_feature(most_discriminatory_feature)
                #   key = feature realization, value = dataset
                # create children as split datasets
                for feature_value, sub_dataset in sub_datasets.items():
                    child_node = Node(node, sub_dataset, feature_value)
                    self.__id3(child_node, depth + 1)
                    node.add_child(child_node)

        else:
            # create a leaf node representing the most frequent observed value
            node.add_child(Leaf(node, node.dataset.most_frequent_label()))

    def __id3_returnsnode(self, dataset: Dataset, parent_dataset: Dataset, depth: int) -> Union[Node, Leaf]:
        if self._max_depth is None or depth < self._max_depth:
            if len(dataset) == 0:
                return Leaf(parent_dataset.most_frequent_label())
            elif len(dataset.label_space) == 1 or len(dataset.feature_names) == 0:
                return Leaf(dataset.most_frequent_label())
            else:
                mdf: str = dataset.most_discriminatory_feature()
                sub_datasets: dict[str, Dataset] = dataset.group_by_feature(mdf)
                node: Node = Node(mdf, dataset.most_frequent_label())
                for feature_value, sub_dataset in sub_datasets.items():
                    child_node = self.__id3_returnsnode(sub_dataset, dataset, depth + 1)
                    node.add_child(feature_value, child_node)
                return node
        else:
            return Leaf(dataset.most_frequent_label())

    def predict(self, dataset: Dataset) -> dict:
        """predicts the class labels of the given test set, based on a previously fitted model.

        :param dataset: dataset consisting of
        :return: list of predicted values
        """
        prediction_params: dict = {"branches": self._branches}
        predicted_values: list[str] = []
        for example, label in dataset:
            # predict the example
            predicted_values.append(self.__label_example_refactored(example, self._root))

        prediction_params["predictions"] = predicted_values

        # TODO - update the prediction matrix
        # accuracy
        accuracy: float = 0
        for i, v in enumerate(predicted_values):
            accuracy += 1 if dataset.label_sample()[i] == v else 0
        accuracy /= len(predicted_values)
        prediction_params["accuracy"] = accuracy
        prediction_params["confusion_matrix"] = ConfusionMatrix(
            dataset.label_space, dataset.label_sample(), predicted_values)

        return prediction_params

    def __example_label(self, example: dict[str, str], node: Union[Node, Leaf]) -> str:
        if isinstance(node, Leaf):
            return node.label
        for child in node:
            if isinstance(child, Leaf) or \
                    node.subtree_feature_name in example and \
                    example[node.subtree_feature_name] == child.parent_feature_value:
                return self.__example_label(example, child)
        # if it comes to this point, the example contains some value unseen in the training process
        # -> return the most frequent label of the dataset
        return node.dataset.most_frequent_label()

    def __label_example_refactored(self, example: dict[str, str], node: Union[Node, Leaf]) -> str:
        if isinstance(node, Leaf):
            return node.label
        for branch_value, child_node in node.children():
            if node.feature in example and example[node.feature] == branch_value:
                return self.__label_example_refactored(example, child_node)
        return node.most_frequent_label
