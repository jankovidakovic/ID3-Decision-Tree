
class ConfusionMatrix:
    def __init__(self, label_space: list[str], expected: list[str], actual: list[str]):
        self.__conf_mat: list[list[int]] = [[0 for _ in range(len(label_space))] for _ in range(len(label_space))]

        # sort the values alphabetically
        sorted_labels: list[str] = sorted(label_space)  # i hope this doesn't modify the label space in-place
        for i, v in enumerate(expected):
            self.__conf_mat[sorted_labels.index(v)][sorted_labels.index(actual[i])] += 1

    def __str__(self):
        conf_mat_str = ""
        for i in range(len(self.__conf_mat)):
            for j in range(len(self.__conf_mat)):
                conf_mat_str += str(self.__conf_mat[i][j])
                if j != len(self.__conf_mat) - 1:
                    conf_mat_str += " "
            if i != len(self.__conf_mat) - 1:
                conf_mat_str += "\n"

        return conf_mat_str

    def __repr__(self):
        return self.__str__()
