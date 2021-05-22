import csv
import math

from dataset import Dataset


def load_dataset(dataset_path: str) -> Dataset:
    """parses the csv dataset at the given path into an instance of Dataset class

    :param dataset_path: path at which the dataset resides
    :return: an instance of Dataset class containing all the data
    """
    with open(dataset_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        dataset = None
        for row in csv_reader:
            if dataset is None:  # first row of the csv file
                dataset = Dataset(row)
            else:
                dataset.add_example(row[:-1], row[-1])
        return dataset


def entropy(examples: list[str], classes: list[str]) -> float:
    if len(examples) == 0:
        raise ValueError("Empty example dataset")

    # calculate the probability of each example
    probs: dict[str, float] = {c: examples.count(c) / len(examples) for c in classes}
    e: float = 0  # entropy
    for c in classes:
        if probs[c] > 0:
            e += probs[c] * math.log2(probs[c])

    return -e  # tested - works


def format_prediction_params(params: dict) -> str:
    predictions = " ".join(p for p in params["predictions"])
    formatted_params: str = f"""[BRANCHES]:
{params["branches"]}[PREDICTIONS]: {predictions}
[ACCURACY]: {params["accuracy"]:.5f}
[CONFUSION_MATRIX]: 
{params["confusion_matrix"]}
"""
    return formatted_params
