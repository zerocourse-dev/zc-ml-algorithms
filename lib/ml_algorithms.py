# ML Algorithm Implementations
#
# Implement three classic ML algorithms from scratch:
# 1. LinearRegression — gradient descent optimization
# 2. LogisticRegression — binary classification with sigmoid
# 3. DecisionTree — CART with Gini impurity
#
# No sklearn allowed for the algorithms themselves — only for
# data loading and evaluation in tests.
#
# Hint: Start with LinearRegression (simplest), then LogisticRegression
# (adds sigmoid + binary cross-entropy), then DecisionTree (recursive splitting).

import math


# ── Linear Regression ──────────────────────────────────────────────────

class LinearRegression:
    """Linear regression with gradient descent.

    Learns weights w and bias b to minimize mean squared error:
    MSE = (1/n) * sum((y_pred - y_true)^2)

    Examples:
        >>> lr = LinearRegression(learning_rate=0.01, n_iterations=1000)
        >>> lr.fit([[1], [2], [3]], [2, 4, 6])
        >>> lr.predict([[4]])  # approximately [8.0]
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Initialize hyperparameters.

        @param learning_rate: float — step size for gradient descent
        @param n_iterations: int — number of gradient descent steps
        """
        raise NotImplementedError("Implement LinearRegression.__init__")

    def fit(self, X, y):
        """Train the model using gradient descent.

        Initialize weights to zeros. For each iteration:
        1. Compute predictions: y_pred = X @ w + b
        2. Compute gradients: dw = (2/n) * X.T @ (y_pred - y), db = (2/n) * sum(y_pred - y)
        3. Update: w -= lr * dw, b -= lr * db

        @param X: list of lists — training features, shape (n_samples, n_features)
        @param y: list — training targets, length n_samples
        @return: self
        """
        raise NotImplementedError("Implement LinearRegression.fit")

    def predict(self, X):
        """Predict target values.

        @param X: list of lists — input features
        @return: list of float — predictions
        """
        raise NotImplementedError("Implement LinearRegression.predict")

    def score(self, X, y):
        """Compute R² (coefficient of determination).

        R² = 1 - SS_res / SS_tot
        where SS_res = sum((y - y_pred)²) and SS_tot = sum((y - y_mean)²)

        @param X: list of lists — input features
        @param y: list — true target values
        @return: float — R² score (1.0 is perfect)
        """
        raise NotImplementedError("Implement LinearRegression.score")


# ── Logistic Regression ────────────────────────────────────────────────

class LogisticRegression:
    """Logistic regression for binary classification.

    Uses the sigmoid function to output probabilities, trained with
    gradient descent on binary cross-entropy loss.

    Examples:
        >>> lg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        >>> lg.fit([[1, 2], [2, 3], [3, 1], [4, 2]], [0, 0, 1, 1])
        >>> lg.predict([[3, 2]])  # [1]
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        """Initialize hyperparameters.

        @param learning_rate: float — step size for gradient descent
        @param n_iterations: int — number of gradient descent steps
        """
        raise NotImplementedError("Implement LogisticRegression.__init__")

    def _sigmoid(self, z):
        """Compute sigmoid function element-wise.

        sigmoid(z) = 1 / (1 + exp(-z))
        Clip z to [-500, 500] to avoid overflow.

        @param z: list of float — input values
        @return: list of float — sigmoid outputs in (0, 1)
        """
        raise NotImplementedError("Implement LogisticRegression._sigmoid")

    def fit(self, X, y):
        """Train using gradient descent on binary cross-entropy.

        Gradients:
        dw = (1/n) * X.T @ (y_pred - y)
        db = (1/n) * sum(y_pred - y)

        @param X: list of lists — training features
        @param y: list — binary labels (0 or 1)
        @return: self
        """
        raise NotImplementedError("Implement LogisticRegression.fit")

    def predict_proba(self, X):
        """Predict probability of class 1.

        @param X: list of lists — input features
        @return: list of float — probabilities in [0, 1]
        """
        raise NotImplementedError("Implement LogisticRegression.predict_proba")

    def predict(self, X, threshold=0.5):
        """Predict class labels.

        @param X: list of lists — input features
        @param threshold: float — decision boundary (default 0.5)
        @return: list of int — predicted labels (0 or 1)
        """
        raise NotImplementedError("Implement LogisticRegression.predict")

    def score(self, X, y):
        """Compute classification accuracy.

        @param X: list of lists — input features
        @param y: list — true labels
        @return: float — accuracy (fraction of correct predictions)
        """
        raise NotImplementedError("Implement LogisticRegression.score")


# ── Decision Tree ──────────────────────────────────────────────────────

class DecisionTreeNode:
    """A node in the decision tree.

    Internal nodes store a feature index and threshold for splitting.
    Leaf nodes store a predicted class label.
    """

    def __init__(self, feature_index=None, threshold=None, left=None,
                 right=None, value=None):
        """Create a tree node.

        @param feature_index: int or None — feature to split on (internal nodes)
        @param threshold: float or None — split threshold (internal nodes)
        @param left: DecisionTreeNode or None — left child (samples <= threshold)
        @param right: DecisionTreeNode or None — right child (samples > threshold)
        @param value: int or None — predicted class (leaf nodes only)
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """Check if this node is a leaf (has a predicted value)."""
        return self.value is not None


class DecisionTree:
    """Decision tree classifier using CART algorithm with Gini impurity.

    Builds a binary tree by recursively finding the best feature/threshold
    split that minimizes weighted Gini impurity.

    Examples:
        >>> dt = DecisionTree(max_depth=3, min_samples_split=2)
        >>> dt.fit([[1], [2], [3], [4]], [0, 0, 1, 1])
        >>> dt.predict([[2.5]])  # [0] or [1] depending on split
    """

    def __init__(self, max_depth=10, min_samples_split=2):
        """Initialize hyperparameters.

        @param max_depth: int — maximum tree depth
        @param min_samples_split: int — minimum samples required to split a node
        """
        raise NotImplementedError("Implement DecisionTree.__init__")

    def _gini(self, y):
        """Compute Gini impurity for a set of labels.

        Gini = 1 - sum(p_i^2) where p_i is the proportion of class i.

        @param y: list — class labels
        @return: float — Gini impurity (0 = pure, max ~0.5 for binary)
        """
        raise NotImplementedError("Implement DecisionTree._gini")

    def _best_split(self, X, y):
        """Find the best feature and threshold to split on.

        For each feature, try each unique value as a threshold.
        Choose the split that gives the lowest weighted Gini impurity.

        @param X: list of lists — features
        @param y: list — labels
        @return: dict with 'feature_index', 'threshold', 'left_indices', 'right_indices'
                 or None if no valid split exists
        """
        raise NotImplementedError("Implement DecisionTree._best_split")

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree.

        Base cases (create leaf):
        - depth >= max_depth
        - len(y) < min_samples_split
        - all labels are the same (pure node)

        Otherwise, find best split and recurse on left/right subsets.

        @param X: list of lists — features
        @param y: list — labels
        @param depth: int — current depth
        @return: DecisionTreeNode
        """
        raise NotImplementedError("Implement DecisionTree._build_tree")

    def _majority_class(self, y):
        """Return the most common class label.

        @param y: list — class labels
        @return: the most frequent label
        """
        raise NotImplementedError("Implement DecisionTree._majority_class")

    def fit(self, X, y):
        """Build the decision tree from training data.

        @param X: list of lists — training features
        @param y: list — training labels
        @return: self
        """
        raise NotImplementedError("Implement DecisionTree.fit")

    def _predict_one(self, node, sample):
        """Predict the class for a single sample by traversing the tree.

        @param node: DecisionTreeNode — current node
        @param sample: list — single sample's features
        @return: predicted class label
        """
        raise NotImplementedError("Implement DecisionTree._predict_one")

    def predict(self, X):
        """Predict class labels for multiple samples.

        @param X: list of lists — input features
        @return: list — predicted class labels
        """
        raise NotImplementedError("Implement DecisionTree.predict")

    def score(self, X, y):
        """Compute classification accuracy.

        @param X: list of lists — input features
        @param y: list — true labels
        @return: float — accuracy
        """
        raise NotImplementedError("Implement DecisionTree.score")

    def tree_depth(self):
        """Return the actual depth of the built tree.

        @return: int — depth (0 for a single leaf)
        """
        raise NotImplementedError("Implement DecisionTree.tree_depth")

    def _compute_depth(self, node):
        """Recursively compute depth of subtree rooted at node.

        @param node: DecisionTreeNode
        @return: int
        """
        raise NotImplementedError("Implement DecisionTree._compute_depth")
