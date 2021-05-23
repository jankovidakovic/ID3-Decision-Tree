import utils


class Dataset:
    """Model of a dataset which stores a list of classified examples, each having a feature map and a class label.

    """

    def __init__(self, features):
        """Creates a new instance of the dataset that will store examples with given features.

        :param features: Feature names of the dataset. Last feature of the list is considered to be the class label.
        """

        self.__feature_names = features[:-1]
        self.__class_label= features[-1]
        self.__data = []

    @property
    def most_frequent_label(self):
        """Returns the most frequent class label within the dataset examples.

        If there are multiple labels of the same frequency, the alphabetical order is considered, and the
        lexicographically smallest label is returned.

        :return: Most frequent label of the dataset.
        """

        label_sample = self.label_sample  # labels present within the dataset
        max_count = label_sample.count(max(label_sample, key=lambda label: label_sample.count(label)))
        max_count_labels = list(filter(lambda label: label_sample.count(label) == max_count, label_sample))
        max_count_labels.sort()
        return max_count_labels[0]

    @property
    def label_sample(self):
        """Returns a list of all labels from the dataset.

        :return: Labels from the dataset
        """

        return [x[1] for x in self.__data]

    @property
    def label_space(self):
        """Returns all the distinct values of the class label occurring within the dataset.

        :return: distinct values of class labels
        """

        return set(self.label_sample)  # this is kinda slow but ok

    @property
    def feature_names(self):
        """Returns the feature names.

        :return: feature names
        """

        return self.__feature_names

    def add_example(self, feature_values, label):
        """Adds the given example into the dataset.

        Feature values are sequentially matched against feature names of the dataset.

        :param feature_values: Feature realizations of the example
        :param label: class label of the example
        """

        self.__data.append((dict(zip(self.__feature_names, feature_values)), label))

    def __add_example(self, feature_map, label) -> None:
        self.__data.append((feature_map, label))

    def group_by_feature(self, feature_name):
        """groups the dataset by distinct values of a feature defined by the given feature name.

        :param feature_name: name of the feature to group by
        :return: dict[str, Dataset] - dataset grouped by the given feature's values
        """

        if feature_name not in self.__feature_names:
            raise ValueError("Feature " + feature_name + " is not a part of the dataset")
        grouped= {}

        # a single dictionary pass:
        for feature_map, label in self.__data:  # for (feature_map, label) in self._data:
            # key = dictionary of features (key=name, value=value), value = label
            feature_map = dict(feature_map)

            # extract the wanted feature entry
            feature_value = feature_map[feature_name]

            #   create new dictionary of features, retaining all but the feature by which the grouping is done
            new_feature_map = feature_map.copy()
            del new_feature_map[feature_name]
            #   if a given feature tuple already has a dataset:
            if feature_value in grouped:
                dataset = grouped[feature_value]
                #   add the (new_features, label) into the dataset
                dataset.__add_example(new_feature_map, label)
            else:
                dataset = Dataset([*new_feature_map.keys(), self.__class_label])
                dataset.__add_example(new_feature_map, label)
                grouped[feature_value] = dataset

        return grouped

    def filter_by_feature(self, feature_name, feature_value):
        """Returns a new dataset containing only the examples for which the feature with the given name
        has the given value.

        :param feature_name:  feature name
        :param feature_value: feature value
        :return:  dataset of examples for which the feature with the given name has the given value
        """

        filtered_dataset = self.group_by_feature(feature_name)
        if feature_value not in filtered_dataset:
            return {}
        return filtered_dataset[feature_value]

    def __str__(self):
        return "Features: " + str(self.__feature_names) + "\n" + \
               "Class label: " + self.__class_label + "\n" + "Data: " + str(self.__data)

    def __repr__(self):
        return self.__str__()

    @property
    def entropy(self):
        """Returns the Shannon entropy of the dataset, according to the distribution of the class label values.

        :return: Entropy of the dataset
        """

        return utils.entropy(self.label_sample)

    def information_gain(self, feature_name):
        """Returns the expected information gain from grouping the dataset by values of the feature with the given name.

        Information gain of some feature is calculated as the difference between the entropy of the whole dataset,
        and the sum of entropies from the subsequent datasets grouped by the feature value, scaled by the ratio of
        examples of the subsequent datasets and the whole dataset.

        :param feature_name: name of the feature of which the information gain should be obtained
        :return: information gain from grouping the dataset by the given feature
        """

        groups = self.group_by_feature(feature_name)
        ig = self.entropy
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
        ig_argmax = max(self.__feature_names, key=lambda feature: self.information_gain(feature))
        max_ig = self.information_gain(ig_argmax)
        max_ig_features = list(filter(lambda f: self.information_gain(f) == max_ig, self.__feature_names))
        max_ig_features.sort()  # alphabetical sorting
        return max_ig_features[0]

    def __iter__(self):
        yield from self.__data

    def __len__(self):
        return len(self.__data)
