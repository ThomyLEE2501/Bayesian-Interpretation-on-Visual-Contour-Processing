"""Hebbian exemplar energy and constrained recurrent feature retrieval."""
import numpy as np
from scipy.special import logsumexp


class HopfieldExemplarPrior:
    """Store bipolar feature patterns with symmetric Hebbian connections.

    A feature group has one +1 entry and -1 at every other value. Training
    averages outer products over all supplied examples and divides by the
    number of bits. Repeated examples retain their frequency weighting, while
    duplicating the whole dataset leaves the weights unchanged.

    Retrieval updates an entire feature group at a time so states remain valid
    feature configurations. This is a categorical, Hopfield-style restriction
    of the binary energy, with known feature values optionally clamped.
    """

    def __init__(self, cardinalities, inverse_temperature=1.):
        sizes = np.asarray(cardinalities, dtype=float)
        if (sizes.ndim != 1 or not sizes.size or not np.isfinite(sizes).all()
                or (sizes < 1).any() or (sizes != np.floor(sizes)).any()):
            raise ValueError("cardinalities must contain positive integer sizes.")
        if not np.isfinite(inverse_temperature) or inverse_temperature < 0:
            raise ValueError("inverse_temperature must be finite and nonnegative.")
        self.cardinalities = tuple(sizes.astype(int))
        self.offsets = np.concatenate(([0], np.cumsum(self.cardinalities)))
        self.n_bits = int(self.offsets[-1])
        self.inverse_temperature = float(inverse_temperature)
        self.weights = None
        self.n_memories = 0

    def _states(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.cardinalities):
            raise ValueError("states must have one column per feature.")
        if (not np.isfinite(states).all() or (states != np.floor(states)).any()
                or (states < 0).any() or (states >= self.cardinalities).any()):
            raise ValueError("Feature values must be valid integer codebook indices.")
        return states.astype(int)

    def encode(self, states):
        """Encode complete feature configurations as grouped bipolar vectors."""
        states = self._states(states)
        encoded = -np.ones((len(states), self.n_bits))
        for feature, offset in enumerate(self.offsets[:-1]):
            encoded[np.arange(len(states)), offset + states[:, feature]] = 1.
        return encoded

    def fit(self, states):
        """Replace the memory using unlabeled training configurations only."""
        patterns = self.encode(states)
        if not len(patterns):
            raise ValueError("At least one training memory is required.")
        weights = (patterns.T @ patterns) / (len(patterns) * self.n_bits)
        np.fill_diagonal(weights, 0.)
        self.weights = weights
        self.n_memories = len(patterns)
        return self

    def energy(self, states):
        """Return -1/2 x.T W x for each complete feature configuration."""
        if self.weights is None:
            raise ValueError("Fit the exemplar memory before reading its energy.")
        encoded = self.encode(states)
        return -.5 * np.einsum('bi,ij,bj->b', encoded, self.weights, encoded)

    def log_prior(self, states):
        """Normalize exp(-beta * energy) over the supplied candidate support."""
        scores = -self.inverse_temperature * self.energy(states)
        if not len(scores):
            raise ValueError("At least one candidate is required.")
        return scores - logsumexp(scores)

    def retrieve(self, cue, known_mask=None, initial_state=None, max_sweeps=50,
                 random_order=False, seed=0, tolerance=1e-12):
        """Complete a partial FEATURE cue by asynchronous energy descent.

        With known_mask omitted, -1 marks unknown features. With an explicit
        mask, unknown cue entries are ignored. Random initial values fill unknown
        groups unless initial_state is supplied. Strictly improving whole-group
        updates preserve clamped values and retain the current value on ties.
        The result is a local attractor, which may differ from a stored memory.
        """
        if self.weights is None:
            raise ValueError("Fit the exemplar memory before retrieval.")
        if not np.isfinite(max_sweeps) or max_sweeps < 1 or int(max_sweeps) != max_sweeps:
            raise ValueError("max_sweeps must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")
        cue = np.asarray(cue, dtype=float)
        if cue.shape != (len(self.cardinalities),):
            raise ValueError("cue must have one entry per feature.")
        known = cue != -1 if known_mask is None else np.asarray(known_mask, dtype=bool)
        if known.shape != cue.shape:
            raise ValueError("known_mask must have the same shape as cue.")
        observed = self._states(np.where(known, cue, 0)[None, :])[0]
        rng = np.random.default_rng(seed)
        if initial_state is None:
            state = rng.integers(self.cardinalities)
        else:
            state = self._states(np.asarray(initial_state)[None, :])[0]
        state[known] = observed[known]
        state_trace = [state.copy()]
        current_energy = float(self.energy(state[None, :])[0])
        energy_trace = [current_energy]
        free_features = np.flatnonzero(~known)
        converged = False
        for sweep in range(int(max_sweeps)):
            changed = False
            order = rng.permutation(free_features) if random_order else free_features
            for feature in order:
                candidates = np.tile(state, (self.cardinalities[feature], 1))
                candidates[:, feature] = np.arange(self.cardinalities[feature])
                energies = self.energy(candidates)
                winner = int(np.argmin(energies))
                if energies[winner] < current_energy - tolerance:
                    state[feature] = winner
                    current_energy = float(energies[winner])
                    changed = True
            state_trace.append(state.copy())
            energy_trace.append(current_energy)
            if not changed:
                converged = True
                break
        return dict(state=state.copy(), initial_state=state_trace[0].copy(),
                    state_trace=np.asarray(state_trace), energy_trace=np.asarray(energy_trace),
                    known_mask=known.copy(), converged=converged, sweeps=sweep+1)
