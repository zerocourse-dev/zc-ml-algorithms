"""Tests for ML Algorithm Implementations.

Students: DO NOT modify this file. Your task is to make all these tests pass
by implementing the classes in lib/ml_algorithms.py.
"""

import math
import pytest

from lib.ml_algorithms import (
    LinearRegression,
    LogisticRegression,
    DecisionTree,
    DecisionTreeNode,
)


# ── Helper: generate simple datasets ──────────────────────────────────

def make_linear_data():
    """y = 2*x1 + 3*x2 + 1 (no noise)."""
    X = [[1, 1], [2, 1], [3, 2], [4, 3], [5, 5], [1, 3], [2, 2], [3, 1]]
    y = [2*x[0] + 3*x[1] + 1 for x in X]
    return X, y


def make_separable_data():
    """Linearly separable binary classification data."""
    X = [[1, 1], [1, 2], [2, 1], [2, 2],
         [5, 5], [5, 6], [6, 5], [6, 6]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    return X, y


def make_xor_data():
    """XOR-like data (not linearly separable)."""
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    return X, y


# ── LinearRegression ──────────────────────────────────────────────────

class TestLinearRegressionInit:
    def test_stores_hyperparameters(self):
        lr = LinearRegression(learning_rate=0.05, n_iterations=500)
        assert lr.learning_rate == 0.05
        assert lr.n_iterations == 500

    def test_default_hyperparameters(self):
        lr = LinearRegression()
        assert lr.learning_rate == 0.01
        assert lr.n_iterations == 1000


class TestLinearRegressionFit:
    def test_fit_returns_self(self):
        lr = LinearRegression()
        result = lr.fit([[1], [2], [3]], [2, 4, 6])
        assert result is lr

    def test_fit_creates_weights(self):
        lr = LinearRegression()
        lr.fit([[1], [2], [3]], [2, 4, 6])
        assert hasattr(lr, "weights")
        assert hasattr(lr, "bias")

    def test_fit_simple_line(self):
        """y = 2x, should learn slope ~2, intercept ~0."""
        lr = LinearRegression(learning_rate=0.01, n_iterations=2000)
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]
        lr.fit(X, y)
        preds = lr.predict([[6]])
        assert abs(preds[0] - 12.0) < 0.5


class TestLinearRegressionPredict:
    def test_predict_length(self):
        lr = LinearRegression(n_iterations=1000)
        lr.fit([[1], [2], [3]], [1, 2, 3])
        preds = lr.predict([[4], [5]])
        assert len(preds) == 2

    def test_predict_multidimensional(self):
        """y = 2*x1 + 3*x2 + 1"""
        lr = LinearRegression(learning_rate=0.005, n_iterations=5000)
        X, y = make_linear_data()
        lr.fit(X, y)
        pred = lr.predict([[1, 1]])[0]
        expected = 2*1 + 3*1 + 1  # 6
        assert abs(pred - expected) < 1.0


class TestLinearRegressionScore:
    def test_r2_perfect_fit(self):
        """Perfect linear data should give R² close to 1."""
        lr = LinearRegression(learning_rate=0.01, n_iterations=2000)
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]
        lr.fit(X, y)
        r2 = lr.score(X, y)
        assert r2 > 0.99

    def test_r2_range(self):
        """R² should be between 0 and 1 for reasonable data."""
        lr = LinearRegression(learning_rate=0.005, n_iterations=3000)
        X, y = make_linear_data()
        lr.fit(X, y)
        r2 = lr.score(X, y)
        assert 0 <= r2 <= 1.0


# ── LogisticRegression ────────────────────────────────────────────────

class TestLogisticRegressionInit:
    def test_stores_hyperparameters(self):
        lg = LogisticRegression(learning_rate=0.05, n_iterations=500)
        assert lg.learning_rate == 0.05
        assert lg.n_iterations == 500


class TestLogisticRegressionSigmoid:
    def test_sigmoid_zero(self):
        lg = LogisticRegression()
        result = lg._sigmoid([0.0])
        assert abs(result[0] - 0.5) < 1e-10

    def test_sigmoid_large_positive(self):
        lg = LogisticRegression()
        result = lg._sigmoid([100.0])
        assert result[0] > 0.99

    def test_sigmoid_large_negative(self):
        lg = LogisticRegression()
        result = lg._sigmoid([-100.0])
        assert result[0] < 0.01

    def test_sigmoid_symmetry(self):
        """sigmoid(x) + sigmoid(-x) = 1."""
        lg = LogisticRegression()
        pos = lg._sigmoid([2.0])[0]
        neg = lg._sigmoid([-2.0])[0]
        assert abs(pos + neg - 1.0) < 1e-10


class TestLogisticRegressionFit:
    def test_fit_returns_self(self):
        lg = LogisticRegression()
        result = lg.fit([[1, 2], [3, 4]], [0, 1])
        assert result is lg

    def test_fit_separable_data(self):
        """Should achieve high accuracy on separable data."""
        lg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        X, y = make_separable_data()
        lg.fit(X, y)
        acc = lg.score(X, y)
        assert acc >= 0.875  # at least 7/8 correct


class TestLogisticRegressionPredict:
    def test_predict_returns_binary(self):
        lg = LogisticRegression(learning_rate=0.1, n_iterations=500)
        X, y = make_separable_data()
        lg.fit(X, y)
        preds = lg.predict(X)
        assert all(p in (0, 1) for p in preds)

    def test_predict_length(self):
        lg = LogisticRegression()
        lg.fit([[1], [2]], [0, 1])
        preds = lg.predict([[3], [4], [5]])
        assert len(preds) == 3


class TestLogisticRegressionProba:
    def test_proba_range(self):
        """All probabilities should be in [0, 1]."""
        lg = LogisticRegression(learning_rate=0.1, n_iterations=500)
        X, y = make_separable_data()
        lg.fit(X, y)
        probas = lg.predict_proba(X)
        for p in probas:
            assert 0.0 <= p <= 1.0

    def test_proba_separable_confidence(self):
        """For well-separated data, class 1 samples should have high probability."""
        lg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        X, y = make_separable_data()
        lg.fit(X, y)
        probas = lg.predict_proba(X)
        # Last 4 are class 1 — should have high probability
        for p in probas[4:]:
            assert p > 0.5


class TestLogisticRegressionScore:
    def test_score_perfect_separable(self):
        lg = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        X, y = make_separable_data()
        lg.fit(X, y)
        assert lg.score(X, y) == 1.0


# ── DecisionTree ──────────────────────────────────────────────────────

class TestDecisionTreeNode:
    def test_leaf_node(self):
        leaf = DecisionTreeNode(value=1)
        assert leaf.is_leaf()
        assert leaf.value == 1

    def test_internal_node(self):
        node = DecisionTreeNode(feature_index=0, threshold=2.5)
        assert not node.is_leaf()
        assert node.feature_index == 0
        assert node.threshold == 2.5


class TestDecisionTreeGini:
    def test_gini_pure(self):
        dt = DecisionTree()
        # Need to access _gini before fit — temporarily create tree
        dt.root = None
        assert dt._gini([0, 0, 0, 0]) == 0.0

    def test_gini_balanced_binary(self):
        dt = DecisionTree()
        dt.root = None
        gini = dt._gini([0, 0, 1, 1])
        assert abs(gini - 0.5) < 1e-10

    def test_gini_multiclass(self):
        dt = DecisionTree()
        dt.root = None
        gini = dt._gini([0, 1, 2])
        # 1 - (1/3)^2 - (1/3)^2 - (1/3)^2 = 1 - 3*(1/9) = 2/3
        assert abs(gini - 2/3) < 1e-10


class TestDecisionTreeFit:
    def test_fit_returns_self(self):
        dt = DecisionTree()
        result = dt.fit([[1], [2], [3]], [0, 0, 1])
        assert result is dt

    def test_fit_creates_root(self):
        dt = DecisionTree()
        dt.fit([[1], [2], [3]], [0, 0, 1])
        assert dt.root is not None

    def test_fit_pure_data(self):
        """All same label → single leaf node."""
        dt = DecisionTree()
        dt.fit([[1], [2], [3]], [1, 1, 1])
        assert dt.root.is_leaf()
        assert dt.root.value == 1


class TestDecisionTreePredict:
    def test_predict_simple_split(self):
        """Should split cleanly on a single feature."""
        dt = DecisionTree(max_depth=1)
        X = [[1], [2], [3], [4], [5], [6]]
        y = [0, 0, 0, 1, 1, 1]
        dt.fit(X, y)
        preds = dt.predict([[1], [6]])
        assert preds[0] == 0
        assert preds[1] == 1

    def test_predict_length(self):
        dt = DecisionTree()
        dt.fit([[1], [2], [3]], [0, 0, 1])
        preds = dt.predict([[1], [2], [3], [4]])
        assert len(preds) == 4

    def test_predict_separable_data(self):
        dt = DecisionTree(max_depth=5)
        X, y = make_separable_data()
        dt.fit(X, y)
        assert dt.score(X, y) == 1.0

    def test_predict_xor(self):
        """Decision tree can learn XOR (non-linear)."""
        dt = DecisionTree(max_depth=3)
        X, y = make_xor_data()
        dt.fit(X, y)
        assert dt.score(X, y) == 1.0


class TestDecisionTreeScore:
    def test_score_range(self):
        dt = DecisionTree()
        X, y = make_separable_data()
        dt.fit(X, y)
        score = dt.score(X, y)
        assert 0 <= score <= 1.0

    def test_perfect_accuracy(self):
        dt = DecisionTree(max_depth=10)
        X = [[i] for i in range(20)]
        y = [0 if i < 10 else 1 for i in range(20)]
        dt.fit(X, y)
        assert dt.score(X, y) == 1.0


class TestDecisionTreeDepth:
    def test_single_leaf_depth(self):
        dt = DecisionTree()
        dt.fit([[1], [2]], [0, 0])
        assert dt.tree_depth() == 0

    def test_max_depth_constraint(self):
        dt = DecisionTree(max_depth=2)
        X = [[i] for i in range(100)]
        y = [i % 4 for i in range(100)]
        dt.fit(X, y)
        assert dt.tree_depth() <= 2

    def test_depth_grows_with_data(self):
        dt1 = DecisionTree(max_depth=10)
        dt1.fit([[1], [2]], [0, 1])

        dt2 = DecisionTree(max_depth=10)
        X = [[i] for i in range(20)]
        y = [i % 3 for i in range(20)]
        dt2.fit(X, y)

        assert dt2.tree_depth() >= dt1.tree_depth()


# ── Integration Tests ─────────────────────────────────────────────────

class TestIntegration:
    def test_linear_regression_on_harder_data(self):
        """Multiple features, should still fit well."""
        lr = LinearRegression(learning_rate=0.001, n_iterations=5000)
        X = [[i, i*2, i*0.5] for i in range(1, 11)]
        y = [3*x[0] + 1*x[1] - 2*x[2] + 5 for x in X]
        lr.fit(X, y)
        r2 = lr.score(X, y)
        assert r2 > 0.95

    def test_decision_tree_multiclass(self):
        """Three classes, should classify correctly."""
        dt = DecisionTree(max_depth=5)
        X = [[1], [2], [3], [10], [11], [12], [20], [21], [22]]
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        dt.fit(X, y)
        assert dt.score(X, y) == 1.0

    def test_logistic_regression_decision_boundary(self):
        """Points near the boundary should have probabilities near 0.5."""
        lg = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        X, y = make_separable_data()
        lg.fit(X, y)
        # A point between the clusters
        proba = lg.predict_proba([[3.5, 3.5]])[0]
        assert 0.2 < proba < 0.8
