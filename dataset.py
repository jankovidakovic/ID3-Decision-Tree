import utils


class Dataset:
    """Model of a dataset which stores a list of classified examples, each having a feature map and a class label.

    """

    def __init__(self, features: list[str]) -> None:
        """Creates a new instance of the dataset that will store examples with given features.

        :param features: Feature names of the dataset. Last feature of the list is considered to be the class label.
        """

        self.__feature_names: list[str] = features[:-1]
        self.__class_label: str = features[-1]
        self.__data: list[tuple[dict[str, str], str]] = []

    @property
    def most_frequent_label(self) -> str:
        """Returns the most frequent class label within the dataset examples.

        If there are multiple labels of the same frequency, the alphabetical order is considered, and the
        lexicographically smallest label is returned.

        :return: Most frequent label of the dataset.
        """

        max_count: int = self.label_sample.count(
            max(self.label_sample, key=lambda label: self.label_sample.count(label)))
        max_count_labels: list[str] = list(
            filter(lambda label:
                   self.label_sample.count(label) == max_count, self.label_sample))
        max_count_labels.sort()
        return max_count_labels[0]

    @property
    def label_sample(self) -> list[str]:
        """Returns a list of all labels from the dataset.

        :return: Labels from the dataset
        """

        return [x[1] for x in self.__data]

    @property
    def label_space(self) -> set[str]:
        """Returns all the distinct values of the class label occurring within the dataset.

        :return: distinct values of class labels
        """

        return set(self.label_sample)  # this is kinda slow but ok

    @property
    def feature_names(self) -> list[str]:
        """Returns the feature names.

        :return: feature names
        """

        return self.__feature_names

    def add_example(self, feature_values: list[str], label: str) -> None:
        """Adds the given example into the dataset.

        Feature values are sequentially matched against feature names of the dataset.

        :param feature_values: Feature realizations of the example
        :param label: class label of the example
        """

        self.__data.append((dict(zip(self.__feature_names, feature_values)), label))

    def __add_example(self, feature_map: dict[str, str], label: str) -> None:
        self.__data.append((feature_map, label))

    def group_by_feature(self, feature_name: str) -> dict:
        """groups the dataset by distinct values of a feature defined by the given feature name.

        :param feature_name: name of the feature to group by
        :return: dict[str, Dataset] - dataset grouped by the given feature's values
        """

        if feature_name not in self.__feature_names:
            raise ValueError("Feature " + feature_name + " is not a part of the dataset")
        grouped: dict[str, Dataset] = {}

        # a single dictionary pass:
        for feature_map, label in self.__data:  # for (feature_map, label) in self._data:
            feature_map: dict[str, str] = dict(feature_map)
            label: str
            feature_value = feature_map[feature_name]

            #   create new dictionary of features, retaining all but the feature by which the grouping is done
            new_feature_map: dict[str, str] = feature_map.copy()
            del new_feature_map[feature_name]
            #   if a given feature tuple already has a dataset:
            if feature_value in grouped:
                dataset: Dataset = grouped[feature_value]
                #   add the (new_features, label) into the dataset
                dataset.__add_example(new_feature_map, label)
            else:
                dataset: Dataset = Dataset([*new_feature_map.keys(), self.__class_label])
                dataset.__add_example(new_feature_map, label)
                grouped[feature_value] = dataset

        return grouped

    def filter_by_feature(self, feature_name: str, feature_value: str):
        """Returns a new dataset containing only the examples for which the feature with the given name
        has the given value.

        :param feature_name:  feature name
        :param feature_value: feature value
        :return:  dataset of examples for which the feature with the given name has the given value
        """

        filtered_dataset: dict[str, Dataset] = self.group_by_feature(feature_name)
        if feature_value not in filtered_dataset:
            return {}
        return filtered_dataset[feature_value]

    def __str__(self) -> str:
        return "Features: " + str(self.__feature_names) + "\n" + \
               "Class label: " + self.__class_label + "\n" + "Data: " + str(self.__data)

    def __repr__(self):
        return self.__str__()

    @property
    def entropy(self) -> float:
        """Returns the Shannon entropy of the dataset, according to the distribution of the class label values.

        :return: Entropy of the dataset
        """

        return utils.entropy(self.label_sample)

    def information_gain(self, feature_name: str) -> float:
        """Returns the expected information gain from grouping the dataset by values of the feature with the given name.

        Information gain of some feature is calculated as the difference between the entropy of the whole dataset,
        and the sum of entropies from the subsequent datasets grouped by the feature value, scaled by the ratio of
        examples of the subsequent datasets and the whole dataset.

        :param feature_name: name of the feature of which the information gain should be obtained
        :return: information gain from grouping the dataset by the given feature
        """

        groups: dict[str, Dataset] = self.group_by_feature(feature_name)
        ig: float = self.entropy
        for dataset in groups.values():
            ig -= dataset.entropy * len(dataset.__data) / len(self.__data)
        return ig

    @property
    def most_discriminatory_feature(self) -> str:
        """Returns the name of the most discriminatory feature of the dataset.

        Most discriminatory feature of the dataset is the one for which the information gain is the highest.
        If there are multiple such features, the one returned is lexicographically smallest.

        :return: feature name of the most discriminatory feature
        """
        ig_argmax: str = max(self.__feature_names, key=lambda feature: self.information_gain(feature))
        max_ig: float = self.information_gain(ig_argmax)
        max_ig_features: list[str] = list(filter(lambda f: self.information_gain(f) == max_ig, self.__feature_names))
        max_ig_features.sort()  # alphabetical sorting
        return max_ig_features[0]

    def __iter__(self) -> tuple[dict[str, str], str]:
        yield from self.__data

    def __len__(self) -> int:
        return len(self.__data)
