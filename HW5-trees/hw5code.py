import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    if np.unique(feature_vector).shape[0] == 1:
        return np.empty(1), np.empty(1), None, -np.inf

    feat_args_sorted = feature_vector.argsort()
    feat_sorted = feature_vector[feat_args_sorted]
    target_sorted = target_vector[feat_args_sorted]

    thresholds = (feat_sorted[:-1] + feat_sorted[1:]) / 2
    feat_sorted_unique = np.unique(feat_sorted)
    thresholds_unique = (feat_sorted_unique[:-1] + feat_sorted_unique[1:]) / 2

    needed_gini = np.in1d(thresholds, thresholds_unique)

    pos_total = target_sorted.sum()
    neg_total = target_sorted.shape[0] - pos_total

    pos_count = target_sorted.cumsum()[:-1]

    l_count = np.arange(1, pos_count.shape[0] + 1, 1)
    r_count = np.arange(pos_count.shape[0], 0, -1)
    neg_count = l_count - pos_count

    pos_ratio_l = pos_count / l_count
    pos_ratio_r = (pos_total - pos_count) / r_count

    neg_ratio_l = neg_count / l_count
    neg_ratio_r = (neg_total - neg_count) / r_count

    H_l = 1 - pos_ratio_l ** 2 - neg_ratio_l ** 2
    H_r = 1 - pos_ratio_r ** 2 - neg_ratio_r ** 2

    y_l_ratio = l_count / feat_sorted.shape[0]
    y_r_ratio = y_l_ratio[::-1]

    ginis = -y_l_ratio * H_l - y_r_ratio * H_r
    ginis = ginis[needed_gini]
    best_gini_idx = np.argmax(ginis)

    return thresholds_unique, ginis, thresholds_unique[best_gini_idx], ginis[best_gini_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature_best = node["feature_split"]

        if self._feature_types[feature_best] == "real":
            threshold_best = node["threshold"]
            if x[feature_best] < threshold_best:
                return self._predict_node(x, node["left_child"])

            return self._predict_node(x, node["right_child"])

        elif self._feature_types[feature_best] == "categorical":
            threshold_best = node["categories_split"]
            if x[feature_best] in threshold_best:
                return self._predict_node(x, node["left_child"])

            return self._predict_node(x, node["right_child"])
        else:
            raise ValueError('Unknown feature type')

    def fit(self, X, y):
        if type(y) != np.ndarray and type(y) != np.array:
            y = y.values
        if type(X) != np.ndarray and type(X) != np.array:
            X = X.values

        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        if type(X) != np.ndarray and type(X) != np.array:
            X = X.values

        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        out = {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }

        return out
