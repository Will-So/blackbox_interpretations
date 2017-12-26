"""
Takes an already fitted model with an ensemble of trees and finds the tree
that most closely resembles the results of the overall model.
"""
from sklearn import tree
import graphviz


def get_best_tree(model, X, keep_scores=False):
    """
    Given a model of ensembled trees with an `estimators_` attribute,
    finds the tree that most closely resembles

    Parameters
    ----------
    model
    X

    Returns
    -------

    """
    overall_prediction = model.predict(X)

    predictions = dict()
    scores = dict()

    best_score, best_tree_number = -999, -999

    for tree_num, tree in enumerate(model.estimators_):
        predictions[tree_num] = tree.predict(X)
        new_score = tree.score(X, overall_prediction)
        scores[tree_num] = new_score

        if new_score > best_score:
            best_score = new_score
            best_tree_number = tree_num

    nearest_tree = model.estimators_[best_tree_number]

    if keep_scores:
        return best_tree_number, nearest_tree, scores
    return best_tree_number, nearest_tree


def visualize_tree(best_tree, feature_names, class_names):
    """Visualizes the best tree with Graphviz

    Parameters
    ----------
    best_tree tree object to be plotted
    feature_names names of the features
    class_names Names of the classes

    Returns
    ---
    A graphviz object with the visualized trees.
    """
    dot_data = tree.export_graphviz(best_tree, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)

    return graph