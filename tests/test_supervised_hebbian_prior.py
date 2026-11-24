"""Tests of co-activation learning, baseline equivalence and inference integration."""
import itertools
import unittest
import numpy as np
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour_models.hebbian_category_priors import SupervisedHebbianPrior
from contour_models.face_recipes_and_observations import (CARDINALITIES, CLASS_NAMES, fit_priors, make_library,
                           sample_training, sample_trials)
from contour_models.vsa_inference import PixelVSA, posterior, mean_field


class HebbianPriorTests(unittest.TestCase):
    def test_one_observation_strengthens_only_active_connections(self):
        model = SupervisedHebbianPrior((2, 3), 2, smoothing=1, learning_rate=.5)
        model.update([1, 2], 0)
        np.testing.assert_array_equal(model.feature_weights[0], [[1, 1.5], [1, 1]])
        np.testing.assert_array_equal(model.feature_weights[1], [[1, 1, 1.5], [1, 1, 1]])
        np.testing.assert_array_equal(model.category_weights, [1.5, 1])
        self.assertEqual(model.n_samples_seen, 1)

    def test_chunking_and_order_preserve_additive_learning(self):
        states = np.array([[1, 2], [0, 1], [0, 2], [1, 2]])
        labels = np.array([0, 1, 0, 1])
        whole = SupervisedHebbianPrior((2, 3), 2).partial_fit(states, labels)
        chunks = SupervisedHebbianPrior((2, 3), 2)
        chunks.partial_fit(states[:2], labels[:2]).partial_fit(states[2:], labels[2:])
        reverse = SupervisedHebbianPrior((2, 3), 2).partial_fit(states[::-1], labels[::-1])
        for other in (chunks, reverse):
            for actual, expected in zip(other.feature_weights, whole.feature_weights):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(other.category_weights, whole.category_weights)
            self.assertEqual(other.n_samples_seen, 4)

    def test_initial_prior_and_unseen_categories_have_full_support(self):
        states = np.array(list(itertools.product(range(2), range(3))))
        model = SupervisedHebbianPrior((2, 3), 3)
        np.testing.assert_allclose(np.exp(model.log_prior(states)), np.full(6, 1/6))
        model.update([1, 2], 0)
        self.assertTrue(np.isfinite(model.log_prior(states)).all())
        self.assertAlmostEqual(np.exp(model.log_prior(states)).sum(), 1)
        np.testing.assert_allclose(model.feature_probabilities[0][2], [.5, .5])

    def test_invalid_chunk_does_not_partially_change_model(self):
        model = SupervisedHebbianPrior((2, 3), 2).update([1, 2], 0)
        before = [w.copy() for w in model.feature_weights]
        categories_before = model.category_weights.copy()
        for states, labels in (([[0, 1], [0, 99]], [0, 1]),
                               ([[0, 1], [0, 2]], [0, -1]),
                               ([[.5, 1]], [0]), ([[0, 1]], [np.nan]),
                               ([[0, 1]], [0, 1])):
            with self.subTest(states=states, labels=labels):
                with self.assertRaises(ValueError):
                    model.partial_fit(states, labels)
                for actual, expected in zip(model.feature_weights, before):
                    np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(model.category_weights, categories_before)
                self.assertEqual(model.n_samples_seen, 1)

    def test_invalid_learning_parameters_are_rejected(self):
        for kwargs in ({'smoothing': 0}, {'learning_rate': -1},
                       {'learning_rate': np.nan}, {'smoothing': np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SupervisedHebbianPrior((2, 3), 2, **kwargs)

    def test_repeated_examples_are_additional_learning(self):
        model = SupervisedHebbianPrior((2,), 2)
        model.update([1], 0)
        probability_once = model.feature_probabilities[0][0, 1]
        model.update([1], 0)
        self.assertGreater(model.feature_probabilities[0][0, 1], probability_once)
        self.assertEqual(model.n_samples_seen, 2)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = make_library(size=24)
        cls.ids, cls.labels = sample_training(cls.library, n_per_class=25)
        cls.batch = fit_priors(cls.library, cls.ids, cls.labels, category_learning='counts')
        cls.online = fit_priors(cls.library, cls.ids, cls.labels, category_learning='hebbian')

    def test_default_matches_batch_category_prior_and_other_priors(self):
        for name in self.batch.logpriors:
            np.testing.assert_allclose(self.online.logpriors[name], self.batch.logpriors[name],
                                       rtol=0, atol=1e-12)
        np.testing.assert_allclose(self.online.class_logp, self.batch.class_logp, atol=1e-12)
        np.testing.assert_array_equal(self.online.class_weights, self.batch.class_weights)

    def test_learning_rate_changes_effective_pseudocount(self):
        online = fit_priors(self.library, self.ids, self.labels, hebbian_learning_rate=.25)
        batch = fit_priors(self.library, self.ids, self.labels, smoothing=4,
                           category_learning='counts')
        np.testing.assert_allclose(online.logpriors['category'], batch.logpriors['category'], atol=1e-12)

    def test_full_candidate_and_iterative_inference_match_baseline(self):
        trial = sample_trials(self.library, n_per_visibility=1, visibilities=(.15,))[0]
        encoder = PixelVSA(24, dim=128, seed=4)
        for backend in ('pixels', 'vsa'):
            loglik = encoder.evidence_scores(self.library.images, trial['observed'],
                                            trial['mask'], backend=backend)
            np.testing.assert_allclose(posterior(loglik, self.online.logpriors['category']),
                                       posterior(loglik, self.batch.logpriors['category']), atol=1e-12)
            online = mean_field(loglik, self.online.logpriors['category'],
                                self.library.states, CARDINALITIES, seed=3, sweeps=8)
            batch = mean_field(loglik, self.batch.logpriors['category'],
                               self.library.states, CARDINALITIES, seed=3, sweeps=8)
            np.testing.assert_allclose(online['joint'], batch['joint'], atol=1e-12)

    def test_adapter_rejects_held_out_training_and_unknown_backend(self):
        with self.assertRaises(ValueError):
            fit_priors(self.library, [self.library.test_ids[0]], [0])
        with self.assertRaises(ValueError):
            fit_priors(self.library, self.ids, self.labels, category_learning='unknown')

    def test_fitting_does_not_read_teacher_probabilities(self):
        library = make_library(size=24)
        library.teacher_logp[:] = np.nan
        fitted = fit_priors(library, self.ids, self.labels)
        np.testing.assert_array_equal(fitted.logpriors['category'], self.online.logpriors['category'])


if __name__ == '__main__':
    unittest.main()
