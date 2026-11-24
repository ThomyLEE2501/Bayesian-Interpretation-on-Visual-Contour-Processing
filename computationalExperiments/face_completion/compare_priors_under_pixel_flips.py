"""Plot matched counting and label-free Hebbian recovery across capped pixel flips."""
import argparse
import json
from pathlib import Path

import numpy as np

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contour_models.face_recipes_and_observations import block_mask, fit_priors, make_library, sample_training
from computationalExperiments.face_completion.compare_priors_on_six_hint_trials import (
    ALPHA, COMPETITIVE_METHODS, COMPETITIVE_SEEDS, CUE_SCORE, DIMENSION, ENCODER_SEED,
    HINTS, IMAGE_SIZE, INK, SEED, STOPS, VISIBILITIES, draw_recovery_contour,
    fit_competitive_priors, make_cues, write_csv,
)
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from contour_models.vsa_inference import PixelVSA

NOISE_PCT = (2, 5, 10, 15, 20)
FACES = 64
LABELS = ("Feature counts", "Label-free Hebbian (3-seed mean)")


def noisy_images(truth, mask, flip_order):
    """Take nested, uniformly selected flips, with an exact cap on visible pixels."""
    visible = flip_order[mask.flat[flip_order]]
    base = np.where(mask, truth, 0).astype(np.float32)
    for rate in NOISE_PCT:
        count = len(visible) * rate // 100
        observed = base.copy()
        flipped = visible[:count]
        observed.flat[flipped] = 1 - observed.flat[flipped]
        yield observed, count


def run(library, repeats, seed, *, visibilities=VISIBILITIES, include_counts=True):
    training_ids, labels = sample_training(library)
    competitive_priors, _ = fit_competitive_priors(library, training_ids)
    priors = list(competitive_priors.values())
    if include_counts:
        fitted = fit_priors(library, training_ids, labels, category_learning="counts")
        priors.insert(0, fitted.logpriors["category"])
    priors = ALPHA * np.stack(priors)
    encoder = PixelVSA(IMAGE_SIZE, DIMENSION, seed=ENCODER_SEED)
    rng = np.random.default_rng(seed)
    heldout = rng.choice(library.test_ids, repeats, replace=False)
    assert set(training_ids).isdisjoint(heldout)
    totals = np.zeros((len(priors), len(NOISE_PCT), len(HINTS), len(visibilities)), dtype=np.int64)
    visible_totals = np.zeros(len(visibilities), dtype=np.int64)
    flip_totals = np.zeros((len(NOISE_PCT), len(visibilities)), dtype=np.int64)
    max_realized = 0.
    for i, state_id in enumerate(heldout, 1):
        stimulus_seed, cue_seed = map(int, rng.integers(2**32, size=2))
        truth = library.images[state_id]
        state = library.states[state_id]
        feature_hits = (library.states == state).sum(axis=1)
        cues = np.stack([make_cues(state, cue_seed, correct) for correct in HINTS])
        cue_bonus = CUE_SCORE * (library.states[None, :, :] == cues[:, None, :]).sum(axis=2)
        prior_and_cue = priors[:, None, :] + cue_bonus[None, :, :]
        flip_order = np.random.default_rng(stimulus_seed + 2**32).permutation(truth.size)
        for v, visibility in enumerate(visibilities):
            mask = block_mask(IMAGE_SIZE, visibility / 100, np.random.default_rng(stimulus_seed))
            visible = int(mask.sum())
            visible_totals[v] += visible
            for n, (observed, flips) in enumerate(noisy_images(truth, mask, flip_order)):
                flip_totals[n, v] += flips
                max_realized = max(max_realized, flips / visible if visible else 0)
                evidence = encoder.evidence_scores(library.images, observed, mask, NOISE_PCT[n] / 100)
                winners = np.argmax(prior_and_cue + evidence, axis=2)
                totals[:, n, :, v] += feature_hits[winners]
        if i % 8 == 0 or i == repeats:
            print(f"Scored {i}/{repeats} held-out faces.", flush=True)
    assert max_realized <= max(NOISE_PCT) / 100
    return totals / repeats, visible_totals, flip_totals, max_realized


def condition_rows(by_method, visibilities, faces, visible, flips):
    return [
        dict(method=key, target_flip_pct=noise, visibility_pct=visibility, correct_hints=correct,
             held_out_faces=faces, mean_correct_features=float(method_scores[n, correct, v]),
             realized_flip_pct=100 * float(flips[n, v] / visible[v]) if visible[v] else 0.)
        for key, method_scores in by_method.items()
        for n, noise in enumerate(NOISE_PCT)
        for v, visibility in enumerate(visibilities)
        for correct in HINTS
    ]


def plot_contours(path, panels, title):
    norm = Normalize(2, 6)
    palette = LinearSegmentedColormap.from_list("recovery", [(norm(v), c) for v, c in STOPS], N=512)
    levels = np.linspace(min(2., panels.min()) - 1e-7, 6 + 1e-7, 81)
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.8, 2.85 * n_rows + 1.15),
                             squeeze=False, layout="constrained")
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            band = draw_recovery_contour(ax, panels[r, c], levels, palette, norm)
            heading = LABELS[c] if n_rows == 1 else f"{LABELS[c]} · {NOISE_PCT[r]}% flips"
            ax.set_title(heading, loc="left", color=INK, fontweight="bold", fontsize=11)
            if c == 0:
                ax.set_ylabel("Correct hints (of 6)")
            if r < n_rows - 1:
                ax.set_xlabel("")
    fig.colorbar(band, ax=axes, ticks=[2, 3, 4, 5, 6], label="Mean correct features (of 6)")
    fig.suptitle(title, fontsize=14, color=INK)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "capped_noise_contours")
    parser.add_argument("--faces", type=int, default=FACES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    library = make_library(size=IMAGE_SIZE)
    if not 1 <= args.faces <= len(library.test_ids) or args.seed < 0:
        parser.error(f"Choose 1–{len(library.test_ids)} faces and a nonnegative seed.")
    scores, visible, flips, max_noise = run(library, args.faces, args.seed)
    mean_hebbian = scores[1:].mean(axis=0)
    combined = np.stack([scores[0], mean_hebbian], axis=1)
    pooled = combined.mean(axis=0, keepdims=True)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_contours(args.out / "visibility_by_hints.png", pooled,
                  f"Face recovery across visibility and correct hints\n"
                  f"Equal average over 2%, 5%, 10%, 15%, and 20% flips · {args.faces} held-out faces per rate")
    plot_contours(args.out / "visibility_by_hints_per_flip_rate.png", combined,
                  f"Face recovery at each pixel-flip rate\n{args.faces} paired held-out faces per condition")
    by_method = dict(zip(("counts", *COMPETITIVE_METHODS), scores))
    by_method["competitive_mean"] = mean_hebbian
    rows = condition_rows(by_method, VISIBILITIES, args.faces, visible, flips)
    write_csv(args.out / "condition_means.csv", rows)
    metadata = dict(seed=args.seed, held_out_faces_per_condition=args.faces,
                    candidate_recipes=len(library.states), renderer_version=library.renderer_version,
                    target_flip_pct=NOISE_PCT, max_realized_flip_pct=100 * max_noise,
                    flip_scheme="Uniformly select floor(visible pixels × rate) pixels without replacement",
                    rate_weighting="equal", prior_weight=ALPHA, cue_log_bonus=float(CUE_SCORE),
                    target_visible_pct=VISIBILITIES, correct_hints=list(HINTS), hints_per_trial=6,
                    competitive_seeds=COMPETITIVE_SEEDS,
                    pooled_mean_correct_features={key: float(values.mean()) for key, values in by_method.items()})
    (args.out / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["pooled_mean_correct_features"], indent=2))


if __name__ == "__main__":
    main()
