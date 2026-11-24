"""Sweep assumed per-hint reliability and count-prior weight on paired trials."""
from pathlib import Path

import numpy as np

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contour_models.face_recipes_and_observations import block_mask, fit_priors, make_library, observe, sample_training
from computationalExperiments.face_completion.compare_priors_on_six_hint_trials import (
    DIMENSION, ENCODER_SEED, HINTS, IMAGE_SIZE, NOISE, REPEATS, SEED,
    VISIBILITIES, WEIGHT_CHECKS, make_cues, write_csv,
)
from contour_models.vsa_inference import PixelVSA


RELIABILITIES = (1 / 3, .5, .7, .85, .95, .99, .999, 1.)
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "hint_reliability"


def main():
    library = make_library(size=IMAGE_SIZE)
    ids, labels = sample_training(library)
    prior = fit_priors(library, ids, labels, category_learning="counts").logpriors["category"]
    encoder = PixelVSA(IMAGE_SIZE, DIMENSION, seed=ENCODER_SEED)
    rng = np.random.default_rng(SEED)
    trial_ids = rng.choice(library.test_ids, REPEATS, replace=False)
    weights = np.array(WEIGHT_CHECKS)[:, None]
    feature_count = library.states.shape[1]
    shape = (len(RELIABILITIES), len(weights), len(VISIBILITIES), len(HINTS))
    recovered = np.zeros(shape, dtype=int)
    exact = np.zeros(shape, dtype=int)
    for state_id in trial_ids:
        stimulus_seed, cue_seed = map(int, rng.integers(2**32, size=2))
        truth = library.states[state_id]
        for v, visibility in enumerate(VISIBILITIES):
            stimulus_rng = np.random.default_rng(stimulus_seed)
            mask = block_mask(IMAGE_SIZE, visibility / 100, stimulus_rng)
            observed = observe(library.images[state_id], mask, NOISE, stimulus_rng)
            evidence = encoder.evidence_scores(library.images, observed, mask, NOISE, "vsa")
            for correct in HINTS:
                hints = make_cues(truth, cue_seed, correct)
                matches = (library.states == hints).sum(axis=1)
                for r, reliability in enumerate(RELIABILITIES):
                    if reliability == 1:
                        hint_score = np.where(matches == feature_count, 0., -np.inf)
                        assert np.isfinite(hint_score).sum() == 1
                    elif reliability == 1 / 3:
                        hint_score = 0.
                    else:
                        mismatch_probability = (1 - reliability) / 2
                        hint_score = matches * np.log(reliability / mismatch_probability)
                    winners = np.argmax((evidence + hint_score) + weights * prior, axis=1)
                    hits = (library.states[winners] == truth).sum(axis=1)
                    recovered[r, :, v, correct] += hits
                    exact[r, :, v, correct] += hits == feature_count

    cells, summary = [], []
    for r, reliability in enumerate(RELIABILITIES):
        for w, weight in enumerate(WEIGHT_CHECKS):
            settings = dict(hint_reliability=reliability, prior_alpha=weight)
            for v, visibility in enumerate(VISIBILITIES):
                for h in HINTS:
                    cells.append(dict(**settings, visibility_pct=visibility, correct_hints=h,
                                      n=REPEATS, mean_features=recovered[r,w,v,h]/REPEATS,
                                      exact_accuracy=exact[r,w,v,h]/REPEATS))
            summary.append(dict(**settings, n=REPEATS*len(VISIBILITIES)*len(HINTS),
                                mean_features=recovered[r,w].mean()/REPEATS,
                                exact_accuracy=exact[r,w].mean()/REPEATS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "by_condition.csv", cells)
    write_csv(OUTPUT / "summary.csv", summary)


if __name__ == "__main__":
    main()
