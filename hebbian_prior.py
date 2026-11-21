"""Supervised and competitive category/feature co-activation learning.

Supervised training clamps category activity to the supplied label. Competitive
training computes anonymous unit activity from feature likelihood and recent
usage. Normalized connections supply the categorical probabilities used by the
model. Both variants receive feature indices from the supplied vocabulary.
"""
import numpy as np
from scipy.special import logsumexp


class SupervisedHebbianPrior:
    """Accumulate local Hebbian increments, then normalize for probabilistic readout.

    With rate=1 and identical initial pseudocounts, one pass gives the same
    category prior as batch counting. Repeated passes count examples again.
    Learning consumes feature indices and labels, without enumerating scenes.
    """

    def __init__(self, cardinalities, n_categories, smoothing=1., learning_rate=1.):
        values = np.asarray(cardinalities, dtype=float)
        if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
                or (values < 1).any() or (values != np.floor(values)).any()):
            raise ValueError("cardinalities must be positive integer sizes.")
        if not np.isfinite(n_categories) or n_categories < 1 or int(n_categories) != n_categories:
            raise ValueError("n_categories must be a positive integer.")
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be finite and positive.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        self.cardinalities = tuple(values.astype(int))
        self.n_categories = int(n_categories)
        self.smoothing = float(smoothing)
        self.learning_rate = float(learning_rate)
        self.feature_weights = [
            np.full((self.n_categories, size), self.smoothing)
            for size in self.cardinalities
        ]
        self.category_weights = np.full(self.n_categories, self.smoothing)
        self.n_samples_seen = 0

    def _states(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.cardinalities):
            raise ValueError("states must have one column per feature.")
        if (not np.isfinite(states).all() or (states != np.floor(states)).any()
                or (states < 0).any() or (states >= self.cardinalities).any()):
            raise ValueError("Feature indices must be integers within their codebooks.")
        return states.astype(int)

    def _labels(self, labels, count):
        labels = np.asarray(labels, dtype=float)
        if (labels.shape != (count,) or not np.isfinite(labels).all()
                or (labels != np.floor(labels)).any() or (labels < 0).any()
                or (labels >= self.n_categories).any()):
            raise ValueError("Supply one valid integer category label per example.")
        return labels.astype(int)

    def partial_fit(self, states, labels):
        """Learn a chunk in order. Invalid input is rejected before any update."""
        states = self._states(states)
        labels = self._labels(labels, len(states))
        for state, category in zip(states, labels):
            # One-hot pre/post activity leaves just one active synapse per
            # feature. This sparse increment equals eta * post * pre.
            for feature, value in enumerate(state):
                self.feature_weights[feature][category, value] += self.learning_rate
            self.category_weights[category] += self.learning_rate
            self.n_samples_seen += 1
        return self

    def update(self, state, category):
        """Learn one complete feature observation and its supplied category."""
        return self.partial_fit(np.asarray(state)[None, :], np.asarray([category]))

    @property
    def feature_probabilities(self):
        return [weights / weights.sum(axis=1, keepdims=True)
                for weights in self.feature_weights]

    @property
    def category_probabilities(self):
        return self.category_weights / self.category_weights.sum()

    def class_log_probabilities(self, states):
        """Return log P(state | category), shape (categories, candidate states)."""
        states = self._states(states)
        scores = np.zeros((self.n_categories, len(states)))
        for feature, probabilities in enumerate(self.feature_probabilities):
            scores += np.log(probabilities[:, states[:, feature]])
        return scores

    def log_prior(self, states):
        """Mix categories, then normalize over the supplied candidate support."""
        scores = logsumexp(
            self.class_log_probabilities(states)
            + np.log(self.category_probabilities[:, None]), axis=0)
        if not len(scores):
            raise ValueError("At least one candidate is required to read a prior.")
        return scores - logsumexp(scores)


class CompetitiveHebbianPrior(SupervisedHebbianPrior):
    """Learn anonymous feature prototypes through soft competitive co-activation.

    No category labels enter training. A unit's activation depends on its
    feature likelihood, learned frequency and a homeostatic usage penalty.
    Activity-weighted Hebbian potentiation and normalization keep each
    feature group on the probability simplex. This stochastic competitive
    rule has no guarantee of likelihood improvement or semantic categories.
    """

    def __init__(self, cardinalities, n_units=3, learning_rate=.03,
                 temperature=.7, homeostasis=2., usage_rate=.02, seed=0):
        super().__init__(cardinalities, n_units, learning_rate=learning_rate)
        if learning_rate > 1:
            raise ValueError("Competitive learning_rate must be at most 1.")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not np.isfinite(homeostasis) or homeostasis < 0:
            raise ValueError("homeostasis must be finite and nonnegative.")
        if not np.isfinite(usage_rate) or not 0 < usage_rate <= 1:
            raise ValueError("usage_rate must be in (0, 1].")
        self.temperature = float(temperature)
        self.homeostasis = float(homeostasis)
        self.usage_rate = float(usage_rate)
        self.n_units = self.n_categories
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Random positive normalized weights break symmetry without examples
        # or labels. No training example is a privileged initialization.
        self.feature_weights = [
            rng.dirichlet(np.ones(size), size=self.n_units)
            for size in self.cardinalities
        ]
        self.usage = np.full(self.n_units, 1 / self.n_units)
        self.activity_totals = np.zeros(self.n_units)

    def predict_proba(self, states):
        """Read P(unit | complete features) under the learned mixture, no updates.

        Homeostatic inhibition and training temperature do not enter this
        probabilistic readout. They influence which units learn during training.
        """
        scores = self.class_log_probabilities(states).T + np.log(self.category_probabilities)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

    def _learn(self, state):
        scores = (self.class_log_probabilities(state[None, :])[:, 0]
                  + np.log(self.category_probabilities)
                  - self.homeostasis * np.log(np.maximum(self.n_units * self.usage, 1e-12)))
        scores = (scores - scores.max()) / self.temperature
        activity = np.exp(scores - logsumexp(scores))
        for feature, value in enumerate(state):
            target = np.eye(self.cardinalities[feature])[value]
            weights = self.feature_weights[feature]
            # eta * post * (pre - weight): Hebbian drive with an
            # activity-dependent normalization term, applied per feature group.
            weights += self.learning_rate * activity[:, None] * (target - weights)
            np.maximum(weights, 1e-12, out=weights)
            weights /= weights.sum(axis=1, keepdims=True)
        self.category_weights += activity
        self.activity_totals += activity
        self.usage += self.usage_rate * (activity - self.usage)
        self.n_samples_seen += 1
        return activity.copy()

    def update(self, state):
        """Learn one unlabeled feature observation and return its unit activities."""
        state = self._states(np.asarray(state)[None, :])[0]
        return self._learn(state)

    def partial_fit(self, states):
        """Learn an unlabeled chunk in order, validating it before updates."""
        states = self._states(states)
        for state in states:
            self._learn(state)
        return self

    def fit(self, states, epochs=20, shuffle_seed=0):
        """Continue learning for shuffled passes. Unit initialization stays fixed."""
        states = self._states(states)
        if not np.isfinite(epochs) or epochs < 1 or int(epochs) != epochs:
            raise ValueError("epochs must be a positive integer.")
        rng = np.random.default_rng(shuffle_seed)
        for _ in range(int(epochs)):
            self.partial_fit(states[rng.permutation(len(states))])
        return self
