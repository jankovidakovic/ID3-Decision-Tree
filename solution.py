import sys

import utils

from decision_tree import DecisionTree


def main():
    """Fits the decision tree, predicts class labels for unseen data and prints the result.

    Expected command line arguments:
        first argument should be the path to the training dataset
        second argument should be the path to the test dataset
        third argument, if provided, is considered to be the depth limit of the decision tree

    """

    train_dataset_path, test_dataset_path = sys.argv[1:3]
    depth_limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    train_dataset = utils.load_dataset(train_dataset_path)
    test_dataset = utils.load_dataset(test_dataset_path)

    decision_tree = DecisionTree(depth_limit)
    decision_tree = decision_tree.fit(train_dataset)
    predictions = decision_tree.predict(test_dataset)

    print(utils.format_prediction_params(predictions))


if __name__ == '__main__':
    main()
