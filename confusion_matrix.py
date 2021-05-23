
class ConfusionMatrix:
    """Model of a confusion matrix, represented as a two-dimensional list of integer values

    """

    def __init__(self, label_space, expected, actual):
        """Initializes the confusion matrix to match the given parameters.

        Confusion matrix presents a more detailed view into some classification outcome. It accumulates
        information about classification correctness in a way that rows of the matrix correspond to
        the expected labels, and columns correspond to actual labels obtained with a classification procedure.

        :param label_space: list of all possible values that the predicted label can have
        :param expected: correct labels of some dataset
        :param actual: labels obtained with some classification procedure
        """

        self.__conf_mat = [[0 for _ in range(len(label_space))] for _ in range(len(label_space))]
        # sort the values alphabetically
        sorted_labels = sorted(label_space)
        for i, v in enumerate(expected):
            self.__conf_mat[sorted_labels.index(v)][sorted_labels.index(actual[i])] += 1

    def __str__(self):
        """Represents the instance of confusion matrix as a string.

        String representation consists of multiple lines, each line corresponding to a row
        in the matrix and consisting of whitespace-separated column values.

        :return: string representation of the confusion matrix
        """

        conf_mat_str = ""
        for i in range(len(self.__conf_mat)):  # row
            for j in range(len(self.__conf_mat)):  # col
                conf_mat_str += str(self.__conf_mat[i][j])
                if j != len(self.__conf_mat) - 1:
                    conf_mat_str += " "
            if i != len(self.__conf_mat) - 1:
                conf_mat_str += "\n"

        return conf_mat_str

    def __repr__(self):
        return self.__str__()
