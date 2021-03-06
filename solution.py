import sys
from typing import Union

import utils
from confusion_matrix import ConfusionMatrix
from dataset import Dataset
from decision_tree import DecisionTree


def main() -> None:
    """Fits the decision tree, predicts class labels for unseen data and prints the result.

    Expected command line arguments:
        first argument should be the path to the training dataset
        second argument should be the path to the test dataset
        third argument, if provided, is considered to be the depth limit of the decision tree

    """

    train_dataset_path, test_dataset_path = sys.argv[1:3]
    depth_limit: int = int(sys.argv[3]) if len(sys.argv) > 3 else None

    train_dataset: Dataset = utils.load_dataset(train_dataset_path)
    test_dataset: Dataset = utils.load_dataset(test_dataset_path)

    decision_tree: DecisionTree = DecisionTree(depth_limit)
    decision_tree = decision_tree.fit(train_dataset)
    predictions: dict[str, Union[str, list[str], float, ConfusionMatrix]] = decision_tree.predict(test_dataset)

    print(utils.format_prediction_params(predictions))


if __name__ == '__main__':
    main()
