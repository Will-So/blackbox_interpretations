from blackbox_interpretations.best_tree import get_best_tree, visualize_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, Y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, random_state=555)


def test_get_best_tree():
    model = RandomForestClassifier(random_state=555).fit(X, Y)

    best_tree_number, best_tree = get_best_tree(model, X)

    assert best_tree_number == 2

    graph = visualize_tree(best_tree, ['feature_1', 'feature_2'], ['0', '1'])

    assert type(graph) == 'graphviz.files.Source'