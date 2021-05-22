import math

class Dataset:
    def __init__(self, features: list[str], **kwargs):
        self._feature_names: list[str] = features[:-1]
        # self._label_values: set[str] = kwargs["label_values"] if "label_values" in kwargs else {}
        self._label_space: set[str] = set()
        self._label_name: str = features[-1]
        # self._data: dict[frozenset, str] = kwargs["data"] if "data" in kwargs else {}
        self._data: list[tuple[dict[str, str], str]] = kwargs["data"] if "data" in kwargs else []
        # what about having the value as a set of strings?
        #       that could work i guess?
        # i cri - dict is not hashable
        # TODO - what if the data is not consistent with labels and features
        '''
        list of tuples (separate features and labels)
            dictionary of features: key = feature name, value = feature_value
            label: str, representing the output value
        '''
        # TODO - what else?

    def most_frequent_label(self):
        label_sample = self.label_sample()
        max_count = label_sample.count(max(label_sample, key=lambda label: label_sample.count(label)))
        max_count_labels = list(filter(lambda label: label_sample.count(label) == max_count, label_sample))
        max_count_labels.sort()
        return max_count_labels[0]

    def label_sample(self):
        return [x[1] for x in self._data]

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_space(self):
        return self._label_space

    def add_example(self, feature_values: list[str], label: str) -> None:
        old_len = len(self._data)
        self._data.append((dict(zip(self._feature_names, feature_values)), label))
        # self._data[frozenset(zip(self._feature_names, feature_values))] = label
        self.add_label_value(label)
        # TODO - BIG PROBLEM - it overrides old values
        # if len(self._data) == old_len:
        #    print(f"Value overridden for {dict(zip(self._feature_names, feature_values))}")

    def __add_example(self, feature_map: dict[str, str], label: str) -> None:
        # self._data[frozenset(feature_map.items())] = label  # doable with a list
        self._data.append((feature_map, label))
        self.add_label_value(label)  # doable with a list

    def add_label_value(self, label_value: str) -> None:
        self._label_space.add(label_value)  # TODO - check here for performance issues

    def group_by_feature(self, feature_name: str):
        """groups the dataset by distinct values of a feature defined by the given feature name.

        :param feature_name: name of the feature to group by
        :return: dict[str, Dataset] - dataset grouped by the given feature's values
        """
        if feature_name not in self._feature_names:
            raise ValueError("Feature " + feature_name + " is not a part of the dataset")
        new_features = list(filter(lambda f: f != feature_name, self._feature_names))
        # new_datasets: list[Dataset] = []
        # for key, value in self._data.items():
        grouped: dict[str, Dataset] = {}

        # a single dictionary pass:
        for feature_map, label in self._data:  # for (feature_map, label) in self._data:
            # key = dictionary of features (key=name, value=value), value = label
            feature_map = dict(feature_map)

            # extract the wanted feature entry
            feature_value = feature_map[feature_name]

            #   create new dictionary of features, retaining all but the feature by which the grouping is done
            new_feature_map: dict[str, str] = feature_map.copy()
            del new_feature_map[feature_name]
            #   if a given feature tuple already has a dataset:
            if feature_value in grouped:
                dataset = grouped[feature_value]
                #   add the (new_features, label) into the dataset
                dataset.__add_example(new_feature_map, label)
            else:
                dataset = Dataset([*new_feature_map.keys(), self._label_name])
                # TODO - what if the dataset contains only the labels
                dataset.__add_example(new_feature_map, label)
                grouped[feature_value] = dataset

        #   otherwise - create a dataset with (data = {new_features: label}, labels = labels)

        return grouped

        # TODO - implement splitting as a composite filter?

    def filter_by_feature(self, feature_name: str, feature_value: str):
        # returns the dataset
        filtered_dataset = self.group_by_feature(feature_name)
        if feature_value not in filtered_dataset:
            return {}
        return filtered_dataset[feature_value]

    def __str__(self):
        return "Features: " + str(self._feature_names) + "\n" + \
               "Class label: " + self._label_name + "\n" + "Data: " + str(self._data)

    def __repr__(self):
        return self.__str__()

    def entropy(self) -> float:
        def label_likelihood(label: str, sample: list[str]):
            return sample.count(label) / len(sample)

        entropy_value = 0
        label_sample = self.label_sample()
        for label in self._label_space:
            prob = label_likelihood(label, label_sample)
            if prob != 0:
                entropy_value += prob * math.log2(prob)
        return -entropy_value

    def information_gain(self, feature_name: str):
        groups = self.group_by_feature(feature_name)
        ig = self.entropy()
        for dataset in groups.values():
            ig -= dataset.entropy() * len(dataset._data) / len(self._data)
        return ig

    def most_discriminatory_feature(self) -> str:
        ig_argmax = max(self._feature_names, key=lambda feature: self.information_gain(feature))
        max_ig = self.information_gain(ig_argmax)
        max_ig_features = list(filter(lambda f: self.information_gain(f) == max_ig, self._feature_names))
        max_ig_features.sort()  # alphabetical sorting
        return max_ig_features[0]

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)
