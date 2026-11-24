"""Hebbian energy, clamped retrieval and prior-adapter regression checks."""
import itertools
import unittest
import numpy as np
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour_models.hopfield_exemplar_prior import HopfieldExemplarPrior
from contour_models.face_recipes_and_observations import CARDINALITIES, fit_priors, make_library, sample_training
from contour_models.vsa_inference import posterior, mean_field


class HopfieldPriorTests(unittest.TestCase):
    def setUp(self):
        self.model = HopfieldExemplarPrior((3, 3)).fit([[1, 2]])
        self.states = np.array(list(itertools.product(range(3), range(3))))

    def test_encoding_has_exactly_one_positive_unit_per_feature(self):
        encoded = self.model.encode(self.states)
        self.assertTrue(np.isin(encoded, [-1, 1]).all())
        for left, right in zip(self.model.offsets[:-1], self.model.offsets[1:]):
            np.testing.assert_array_equal((encoded[:, left:right] == 1).sum(axis=1), 1)

    def test_weights_and_energy_match_explicit_hebbian_formula(self):
        samples = np.array([[1, 2], [2, 1], [1, 2]])
        model = HopfieldExemplarPrior((3, 3)).fit(samples)
        patterns = model.encode(samples)
        expected = sum(np.outer(p, p) for p in patterns) / (len(patterns)*6)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.weights, expected)
        np.testing.assert_array_equal(model.weights.T, model.weights)
        np.testing.assert_array_equal(np.diag(model.weights), 0)
        energies = [-.5 * p @ expected @ p for p in model.encode(self.states)]
        np.testing.assert_allclose(model.energy(self.states), energies)

    def test_prior_normalization_and_zero_inverse_temperature(self):
        q = np.exp(self.model.log_prior(self.states))
        self.assertAlmostEqual(q.sum(), 1)
        self.assertTrue((q > 0).all())
        self.assertEqual(int(q.argmax()), 5)  # state [1, 2]
        flat = HopfieldExemplarPrior((3, 3), inverse_temperature=0).fit([[1, 2]])
        np.testing.assert_allclose(np.exp(flat.log_prior(self.states)), np.full(9, 1/9))

    def test_duplicate_corpus_is_invariant_but_relative_frequency_matters(self):
        a = HopfieldExemplarPrior((3, 3)).fit([[1, 2], [0, 0]])
        b = HopfieldExemplarPrior((3, 3)).fit([[1, 2], [0, 0], [1, 2], [0, 0]])
        c = HopfieldExemplarPrior((3, 3)).fit([[1, 2], [1, 2], [0, 0]])
        np.testing.assert_array_equal(a.weights, b.weights)
        self.assertFalse(np.array_equal(a.weights, c.weights))

    def test_single_memory_can_be_retrieved_from_partial_feature_cue(self):
        result = self.model.retrieve([1, -1], initial_state=[0, 0])
        np.testing.assert_array_equal(result['state'], [1, 2])
        self.assertTrue(result['converged'])
        self.assertTrue((np.diff(result['energy_trace']) <= 1e-12).all())

    def test_retrieval_ignores_unknown_values_and_preserves_known_values(self):
        mask = [True, False]
        a = self.model.retrieve([1, 0], known_mask=mask, seed=3)
        b = self.model.retrieve([1, np.nan], known_mask=mask, seed=3)
        np.testing.assert_array_equal(a['state_trace'], b['state_trace'])
        self.assertTrue((a['state_trace'][:, 0] == 1).all())
        full = self.model.retrieve([0, 1])
        np.testing.assert_array_equal(full['state'], [0, 1])

    def test_ties_retain_initial_state(self):
        model = HopfieldExemplarPrior((2,)).fit([[0], [1]])
        result = model.retrieve([-1], initial_state=[1])
        np.testing.assert_array_equal(result['state'], [1])
        self.assertTrue(result['converged'])

    def test_retrieval_is_reproducible_and_energy_decreases_with_group_updates(self):
        rng = np.random.default_rng(9)
        samples = rng.integers([3, 2, 3], size=(30, 3))
        model = HopfieldExemplarPrior((3, 2, 3)).fit(samples)
        before = model.weights.copy()
        for seed in range(8):
            result = model.retrieve([-1, 1, -1], random_order=True, seed=seed)
            again = model.retrieve([-1, 1, -1], random_order=True, seed=seed)
            np.testing.assert_array_equal(result['state_trace'], again['state_trace'])
            self.assertTrue((np.diff(result['energy_trace']) <= 1e-12).all())
            self.assertTrue((result['state_trace'][:, 1] == 1).all())
            self.assertTrue(result['converged'])
            # At convergence, no free single-feature change improves energy.
            for feature in (0, 2):
                candidates = np.tile(result['state'], (model.cardinalities[feature], 1))
                candidates[:, feature] = np.arange(model.cardinalities[feature])
                self.assertGreaterEqual(model.energy(candidates).min(), result['energy_trace'][-1]-1e-12)
        np.testing.assert_array_equal(before, model.weights)

    def test_invalid_fit_preserves_existing_memory(self):
        before = self.model.weights.copy()
        with self.assertRaises(ValueError):
            self.model.fit([[0, 1], [1, 99]])
        np.testing.assert_array_equal(self.model.weights, before)
        self.assertEqual(self.model.n_memories, 1)

    def test_invalid_inputs_and_unfitted_readouts(self):
        for action in (lambda: HopfieldExemplarPrior((2,), -1),
                       lambda: HopfieldExemplarPrior((2.5,)),
                       lambda: HopfieldExemplarPrior((2,)).fit(np.empty((0, 1))),
                       lambda: HopfieldExemplarPrior((2,)).energy([[1]]),
                       lambda: self.model.retrieve([1, -1], max_sweeps=0),
                       lambda: self.model.retrieve([9, -1]),
                       lambda: self.model.log_prior(np.empty((0, 2)))):
            with self.assertRaises(ValueError):
                action()

    def test_training_accepts_no_category_labels(self):
        with self.assertRaises(TypeError):
            self.model.fit([[1, 2]], [0])


class HopfieldIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = make_library(size=24)
        cls.ids, cls.labels = sample_training(cls.library, n_per_class=20)
        cls.kernel = fit_priors(cls.library, cls.ids, cls.labels)
        cls.hopfield = fit_priors(cls.library, cls.ids, cls.labels, exemplar_learning='hopfield')

    def test_only_exemplar_prior_changes_and_original_reference_remains(self):
        self.assertIsNone(self.kernel.exemplar_model)
        self.assertIsInstance(self.hopfield.exemplar_model, HopfieldExemplarPrior)
        for name in ('uniform', 'generic', 'category'):
            np.testing.assert_array_equal(self.kernel.logpriors[name], self.hopfield.logpriors[name])
        self.assertFalse(np.allclose(self.kernel.logpriors['exemplar'], self.hopfield.logpriors['exemplar']))
        np.testing.assert_allclose(self.hopfield.logpriors['exemplar'],
                                   self.hopfield.exemplar_model.log_prior(self.library.states))

    def test_exemplar_learning_is_independent_of_supplied_category_labels(self):
        changed = fit_priors(self.library, self.ids, self.labels[::-1], exemplar_learning='hopfield')
        np.testing.assert_array_equal(self.hopfield.logpriors['exemplar'], changed.logpriors['exemplar'])
        np.testing.assert_array_equal(self.hopfield.exemplar_model.weights, changed.exemplar_model.weights)

    def test_rejects_heldout_training_and_invalid_exemplar_backend(self):
        with self.assertRaises(ValueError):
            fit_priors(self.library, [self.library.test_ids[0]], [0], exemplar_learning='hopfield')
        with self.assertRaises(ValueError):
            fit_priors(self.library, self.ids, self.labels, exemplar_learning='bad')

    def test_full_candidate_and_iterative_inference_accept_energy_prior(self):
        logprior = self.hopfield.logpriors['exemplar']
        loglik = np.linspace(-3, 0, len(logprior))
        self.assertAlmostEqual(posterior(loglik, logprior).sum(), 1)
        result = mean_field(loglik, logprior, self.library.states, CARDINALITIES, sweeps=5)
        self.assertAlmostEqual(result['joint'].sum(), 1)
        self.assertTrue((np.diff(result['free_energy']) <= 1e-8).all())


if __name__ == '__main__':
    unittest.main()
