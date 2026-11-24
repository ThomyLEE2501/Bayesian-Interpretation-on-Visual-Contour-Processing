"""Competitive activity, label independence and completion integration checks."""
import unittest
import numpy as np
from scipy.special import logsumexp
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour_models.hebbian_category_priors import CompetitiveHebbianPrior
from contour_models.face_recipes_and_observations import CARDINALITIES, fit_priors, make_library, sample_training
from contour_models.vsa_inference import posterior, mean_field


class CompetitivePriorTests(unittest.TestCase):
    def test_one_update_matches_competition_and_normalized_hebbian_rule(self):
        model = CompetitiveHebbianPrior((2, 3), learning_rate=.1, temperature=.8,
                                        homeostasis=0, seed=4)
        state = np.array([1, 2])
        initial = [w.copy() for w in model.feature_weights]
        logits = model.class_log_probabilities(state[None, :])[:, 0]
        logits = (logits + np.log(model.category_probabilities)) / .8
        expected_activity = np.exp(logits - logsumexp(logits))
        activity = model.update(state)
        np.testing.assert_allclose(activity, expected_activity, atol=1e-12)
        for j, w in enumerate(initial):
            target = np.eye(model.cardinalities[j])[state[j]]
            expected = w + .1 * activity[:, None] * (target-w)
            np.testing.assert_allclose(model.feature_weights[j], expected, atol=1e-12)
        self.assertEqual(model.n_samples_seen, 1)
        self.assertAlmostEqual(model.activity_totals.sum(), 1)

    def test_reproducible_streaming_chunks(self):
        states = np.array([[0, 1], [1, 2], [1, 1], [0, 0]])
        first = CompetitiveHebbianPrior((2, 3), seed=8).partial_fit(states)
        second = CompetitiveHebbianPrior((2, 3), seed=8)
        second.partial_fit(states[:2]).partial_fit(states[2:])
        for a, b in zip(first.feature_weights, second.feature_weights):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(first.usage, second.usage)

    def test_readout_is_normalized_positive_and_does_not_learn(self):
        model = CompetitiveHebbianPrior((2, 3), seed=5)
        model.fit([[0, 1], [1, 2], [1, 1]], epochs=3)
        before = [w.copy() for w in model.feature_weights]
        seen = model.n_samples_seen
        states = np.array([[0, 1], [1, 2], [1, 1]])
        activity = model.predict_proba(states)
        np.testing.assert_allclose(activity.sum(axis=1), 1)
        self.assertTrue((activity > 0).all())
        self.assertAlmostEqual(np.exp(model.log_prior(states)).sum(), 1)
        for a, b in zip(before, model.feature_weights):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(model.n_samples_seen, seen)

    def test_random_initialization_breaks_unit_symmetry(self):
        a = CompetitiveHebbianPrior((2, 3), seed=2)
        b = CompetitiveHebbianPrior((2, 3), seed=3)
        self.assertFalse(np.array_equal(a.feature_weights[0][0], a.feature_weights[0][1]))
        self.assertFalse(np.array_equal(a.feature_weights[0], b.feature_weights[0]))

    def test_invalid_chunk_does_not_partially_update(self):
        model = CompetitiveHebbianPrior((2, 3), seed=9)
        before = [w.copy() for w in model.feature_weights]
        with self.assertRaises(ValueError):
            model.partial_fit([[0, 1], [9, 2]])
        self.assertEqual(model.n_samples_seen, 0)
        np.testing.assert_array_equal(model.activity_totals, np.zeros(3))
        for a, b in zip(before, model.feature_weights):
            np.testing.assert_array_equal(a, b)

    def test_invalid_competition_parameters(self):
        for options in ({'learning_rate': 1.1}, {'temperature': 0},
                        {'homeostasis': -1}, {'usage_rate': 0}, {'temperature': np.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CompetitiveHebbianPrior((2, 3), **options)

    def test_inhibition_reduces_relative_activity_of_overused_unit(self):
        model = CompetitiveHebbianPrior((2,), n_units=2, seed=3)
        model.feature_weights[0][:] = [.5, .5]
        model.usage[:] = [.9, .1]
        activity = model.update([0])
        self.assertGreater(activity[1], activity[0])

    def test_specializes_on_separable_unlabeled_patterns(self):
        patterns = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        states = np.repeat(patterns, 30, axis=0)
        model = CompetitiveHebbianPrior((3, 3, 3), seed=2)
        model.fit(states, epochs=12, shuffle_seed=1)
        predictions = model.predict_proba(patterns).argmax(axis=1)
        self.assertEqual(len(np.unique(predictions)), 3)
        for state, unit in zip(patterns, predictions):
            for j, value in enumerate(state):
                self.assertEqual(int(model.feature_probabilities[j][unit].argmax()), value)

    def test_training_api_does_not_accept_labels(self):
        model = CompetitiveHebbianPrior((2, 3))
        with self.assertRaises(TypeError):
            model.partial_fit([[0, 1]], [0])
        with self.assertRaises(TypeError):
            model.update([0, 1], 0)


class CompetitiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = make_library(size=24)
        cls.ids, cls.labels = sample_training(cls.library, n_per_class=15)
        cls.options = dict(category_learning='competitive', competitive_epochs=3,
                           competitive_units=4, competitive_seed=12)
        cls.bank = fit_priors(cls.library, cls.ids, None, **cls.options)

    def test_supplied_labels_are_ignored_and_unit_count_is_configurable(self):
        changed = fit_priors(self.library, self.ids,
                             np.full(len(self.ids), 'unused'), **self.options)
        self.assertEqual(self.bank.class_logp.shape, (4, len(self.library.states)))
        for name in self.bank.logpriors:
            np.testing.assert_array_equal(self.bank.logpriors[name], changed.logpriors[name])

    def test_no_generator_teacher_probabilities_are_used(self):
        library = make_library(size=24)
        library.teacher_logp[:] = np.nan
        changed = fit_priors(library, self.ids, None, **self.options)
        np.testing.assert_array_equal(self.bank.logpriors['category'], changed.logpriors['category'])

    def test_held_out_training_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_priors(self.library, [self.library.test_ids[0]], None, **self.options)

    def test_prior_works_with_full_candidate_and_iterative_inference(self):
        logprior = self.bank.logpriors['category']
        self.assertTrue(np.isfinite(logprior).all())
        self.assertAlmostEqual(np.exp(logprior).sum(), 1)
        loglik = np.linspace(-2, 0, len(logprior))
        self.assertAlmostEqual(posterior(loglik, logprior).sum(), 1)
        result = mean_field(loglik, logprior, self.library.states, CARDINALITIES, sweeps=4)
        self.assertAlmostEqual(result['joint'].sum(), 1)
        self.assertTrue((np.diff(result['free_energy']) <= 1e-8).all())


if __name__ == '__main__':
    unittest.main()
