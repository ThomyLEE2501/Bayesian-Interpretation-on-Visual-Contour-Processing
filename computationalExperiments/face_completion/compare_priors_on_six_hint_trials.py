"""Compare learned priors on the matched six-hint computer experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np
from scipy.optimize import brentq

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contour_models.face_recipes_and_observations import (CARDINALITIES, FACTOR_NAMES, FACTOR_VALUES, block_mask,
                           completion_metrics, fit_priors, make_library,
                           normalize_logp, observe, sample_training)
from contour_models.hebbian_category_priors import CompetitiveHebbianPrior
from contour_models.vsa_inference import PixelVSA, entropy

VISIBILITIES = (10, 12, 15, 18, 21, 25, 30, 40, 55, 70)
HINTS = range(7)
NOISE = .05
CUE_SCORE = np.log(.95 / .025)
SEED = 172
REPEATS = 12
ALPHA = 12.
WEIGHT_CHECKS = (0., 1., 3., 6., 12., 24.)
IMAGE_SIZE = 40
DIMENSION = 512
ENCODER_SEED = 17
COMPETITIVE_SEEDS = (0, 1, 2)
COMPETITIVE_EPOCHS = 20
COMPETITIVE_OPTIONS = dict(n_units=3, learning_rate=.03, temperature=.7, homeostasis=2., usage_rate=.02)
COMPETITIVE_METHODS = {f"competitive_{seed}": f"Competitive Hebbian, seed {seed}"
                       for seed in COMPETITIVE_SEEDS}
MAIN_METHODS = ("counts", "hebbian", "kernel", "hopfield")
METHODS = {
    "counts": "Feature counts",
    "hebbian": "Supervised Hebbian",
    "kernel": "Hamming-kernel exemplars",
    "hopfield": "Hopfield energy",
    "hopfield_matched": "Hopfield, matched prior entropy",
    **COMPETITIVE_METHODS,
}
INK = "#293D48"
STOPS = [(2, "#AF3910"), (2.5, "#BC4814"), (3.2, "#CF6019"), (4, "#F39B43"),
         (4.3, "#F2C88F"), (4.55, "#E9E7CD"), (4.78, "#BDDCD3"), (5, "#88C4BC"),
         (5.5, "#459B95"), (6, "#006B67")]


def effective_entropy(log_prior, alpha):
    return float(entropy(np.exp(normalize_logp(alpha * log_prior))))


def fit_conditions(library, alpha, hopfield_beta):
    ids, labels = sample_training(library)
    assert set(ids).isdisjoint(library.test_ids)
    batch = fit_priors(library, ids, labels, category_learning="counts")
    online = fit_priors(library, ids, labels, category_learning="hebbian")
    hopfield = fit_priors(library, ids, labels, category_learning="counts",
                          exemplar_learning="hopfield", hopfield_inverse_temperature=hopfield_beta)
    memory = hopfield.exemplar_model
    log_prior = {
        "counts": batch.logpriors["category"],
        "hebbian": online.logpriors["category"],
        "kernel": batch.logpriors["exemplar"],
        "hopfield": hopfield.logpriors["exemplar"],
    }
    np.testing.assert_allclose(log_prior["hebbian"], log_prior["counts"], rtol=0, atol=1e-12)
    energy_score = -memory.energy(library.states)
    target = effective_entropy(log_prior["kernel"], alpha)
    if alpha == 0:
        matched_beta = hopfield_beta
    else:
        def error(beta):
            return effective_entropy(beta * energy_score, alpha) - target

        upper = 1.
        while error(upper) > 0 and upper < 1024:
            upper *= 2
        if error(upper) > 0:
            raise ValueError("The kernel entropy is below what the Hopfield prior can reach.")
        matched_beta = float(brentq(error, 0, upper))
    log_prior["hopfield_matched"] = normalize_logp(matched_beta * energy_score)
    np.testing.assert_allclose(effective_entropy(log_prior["hopfield_matched"], alpha), target, atol=1e-7)
    competitive_priors, competitive_diagnostics = fit_competitive_priors(library, ids)
    log_prior.update(competitive_priors)
    return log_prior, ids, matched_beta, competitive_diagnostics


def fit_competitive_priors(library, training_ids):
    training_features = library.states[training_ids]
    log_prior = {}
    competitive_diagnostics = {}
    for seed, key in zip(COMPETITIVE_SEEDS, COMPETITIVE_METHODS):
        learner = CompetitiveHebbianPrior(CARDINALITIES, seed=seed, **COMPETITIVE_OPTIONS)
        learner.fit(training_features, epochs=COMPETITIVE_EPOCHS, shuffle_seed=seed)
        log_prior[key] = learner.log_prior(library.states)
        unit_assignments = learner.predict_proba(training_features).argmax(axis=1)
        competitive_diagnostics[key] = {
            "training_observations_including_repeats": learner.n_samples_seen,
            "unit_mixture_weights": learner.category_probabilities.tolist(),
            "hard_training_assignments": np.bincount(unit_assignments, minlength=learner.n_units).tolist(),
        }
    return log_prior, competitive_diagnostics


def make_cues(state, seed, correct):
    """Use the same nested correct/wrong alternatives as the poster experiment."""
    rng = np.random.default_rng(seed)
    cues = (state + rng.integers(1, 3, size=len(CARDINALITIES))) % 3
    right = rng.permutation(len(CARDINALITIES))[:correct]
    cues[right] = state[right]
    return cues


def evaluate(library, priors, seed, repeats, alpha):
    encoder = PixelVSA(IMAGE_SIZE, DIMENSION, seed=ENCODER_SEED)
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(library.test_ids, repeats, replace=False)
    rows = []
    weight_checks = tuple(sorted({*WEIGHT_CHECKS, alpha}))
    sensitivity_priors = {key: np.array(weight_checks)[:, None] * priors[key]
                          for key in ("counts", "kernel", "hopfield", *COMPETITIVE_METHODS)}
    totals = {key: np.zeros((2, len(weight_checks)), dtype=int) for key in sensitivity_priors}
    for trial, state_id in enumerate(test_ids, 1):
        stimulus_seed, cue_seed = map(int, rng.integers(2**32, size=2))
        truth = library.images[state_id]
        state = library.states[state_id]
        for visibility in VISIBILITIES:
            # Resetting the RNG at each visibility nests the block masks
            # and keeps the underlying pixel flips paired across conditions.
            stimulus_rng = np.random.default_rng(stimulus_seed)
            mask = block_mask(IMAGE_SIZE, visibility / 100, stimulus_rng)
            observed = observe(truth, mask, NOISE, stimulus_rng)
            flipped_pixels = int(np.count_nonzero(observed[mask] != truth[mask]))
            evidence = encoder.evidence_scores(library.images, observed, mask, NOISE, "vsa")
            for correct in HINTS:
                cues = make_cues(state, cue_seed, correct)
                visual_and_cue = evidence + CUE_SCORE * (library.states == cues).sum(axis=1)
                for key, log_weights in sensitivity_priors.items():
                    winners = np.argmax(visual_and_cue + log_weights, axis=1)
                    recovered = (library.states[winners] == state).sum(axis=1)
                    totals[key][0] += recovered
                    totals[key][1] += recovered == len(FACTOR_NAMES)
                for method, log_prior in priors.items():
                    picked = int(np.argmax(visual_and_cue + alpha * log_prior))
                    hit = library.states[picked] == state
                    metrics = completion_metrics(library.images[picked], truth, observed, mask)
                    row = dict(method=method, trial=trial, state_id=int(state_id),
                               stimulus_seed=stimulus_seed, cue_seed=cue_seed,
                               visibility_pct=visibility, correct_cues=correct,
                               visible_pixels=int(mask.sum()), flipped_pixels=flipped_pixels,
                               actual_visibility_pct=100 * float(mask.mean()), map_state=picked,
                               score=int(hit.sum()), **metrics)
                    row.update({f"correct_{feature.replace(' ', '_')}": int(ok)
                                for feature, ok in zip(FACTOR_NAMES, hit)})
                    rows.append(row)
    n = repeats * len(VISIBILITIES) * len(HINTS)
    sensitivity = [dict(method=key, prior_weight=weight, n=n,
                        mean_correct_features=float(features / n), exact_accuracy=float(exact / n))
                   for key, total in totals.items()
                   for weight, features, exact in zip(weight_checks, *total)]
    return rows, sensitivity


def condition_key(row):
    return int(row["trial"]), float(row["visibility_pct"]), int(row["correct_cues"])


def check_reference(rows, reference):
    if reference is None:
        return 0
    old = {condition_key(row): row for row in json.loads(reference.read_text(encoding="utf-8"))}
    baseline = [row for row in rows if row["method"] == "counts"]
    assert len(baseline) == len(old), "The comparison and reference should use the same trial plan."
    fields = ("state_id", "stimulus_seed", "cue_seed", "visible_pixels", "flipped_pixels",
              "map_state", "score", "actual_visibility_pct")
    for row in baseline:
        previous = old[condition_key(row)]
        for field in fields:
            if row[field] != previous[field]:
                raise ValueError(f"Reference mismatch in {condition_key(row)}, {field}: "
                                 f"{row[field]} vs {previous[field]}")
    return len(baseline)


def mean(rows, metric):
    return float(np.nanmean([row[metric] for row in rows]))


def summarize(rows):
    by_method = {key: [row for row in rows if row["method"] == key] for key in METHODS}
    scores = {}
    cells = []
    results = {}
    for key, group in by_method.items():
        values = np.array([r["score"] for r in group])
        baseline = by_method["kernel" if key.startswith("hopfield") else "counts"]
        assert [condition_key(r) for r in group] == [condition_key(r) for r in baseline]
        delta = values - np.array([r["score"] for r in baseline])
        results[key] = {
            "mean_correct_features": float(values.mean()),
            "exact_accuracy": float(np.mean(values == len(FACTOR_NAMES))),
            "mean_hidden_recall": mean(group, "hidden_recall"),
            "mean_hidden_false_positive": mean(group, "hidden_false_positive"),
            "mean_hidden_error": mean(group, "hidden_brier"),
            "paired_reference": baseline[0]["method"],
            "changed_recipes": sum(r["map_state"] != b["map_state"] for r, b in zip(group, baseline)),
            "better_trials": int(np.sum(delta > 0)),
            "worse_trials": int(np.sum(delta < 0)),
            "feature_accuracy": {feature: mean(group, f"correct_{feature.replace(' ', '_')}")
                                 for feature in FACTOR_NAMES},
        }
        scores[key] = np.empty((len(HINTS), len(VISIBILITIES)))
        for correct in HINTS:
            for j, visibility in enumerate(VISIBILITIES):
                cell = [r for r in group if r["visibility_pct"] == visibility and r["correct_cues"] == correct]
                average = mean(cell, "score")
                scores[key][correct, j] = average
                cells.append(dict(method=key, visibility_pct=visibility, correct_cues=correct,
                                  n=len(cell), mean_correct_features=average,
                                  exact_accuracy=float(np.mean([r["score"] == len(FACTOR_NAMES) for r in cell])),
                                  mean_hidden_recall=mean(cell, "hidden_recall"),
                                  mean_hidden_false_positive=mean(cell, "hidden_false_positive")))
    return scores, cells, results


def visibility_axis(ax):
    ax.set(xticks=np.log10([10, 15, 21, 30, 40, 55, 70]), yticks=list(HINTS),
           xlabel="Target visibility (log scale)", xlim=(np.log10(10), np.log10(70)), ylim=(0, 6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{10**x:.0f}%"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(colors=INK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#A3B1B7")


def draw_recovery_contour(ax, values, levels, palette, norm):
    x = np.log10(VISIBILITIES)
    x_grid, y_grid = np.meshgrid(x, list(HINTS))
    panel = ax.contourf(x, list(HINTS), values, levels=levels, cmap=palette, norm=norm)
    target = [v for v in (4, 5) if values.min() < v < values.max()]
    if target:
        boundaries = ax.contour(x, list(HINTS), values, levels=target, colors=INK, linewidths=.65)
        ax.clabel(boundaries, fmt="%g", fontsize=8)
    ax.scatter(x_grid, y_grid, s=8, color=INK, alpha=.48, linewidths=0, clip_on=False)
    visibility_axis(ax)
    return panel


def contour_grid(filename, panels, subtitle, alpha, repeats):
    norm = Normalize(2, 6)
    palette = LinearSegmentedColormap.from_list("recovery", [(norm(v), c) for v, c in STOPS], N=512)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), layout="constrained")
    low = min(2., min(values.min() for _, values in panels))
    levels = np.linspace(low - 1e-7, 6 + 1e-7, 81)
    for ax, (title, values) in zip(axes.flat, panels):
        panel = draw_recovery_contour(ax, values, levels, palette, norm)
        ax.set_title(title, color=INK, loc="left", fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Correct hints (of 6)")
    fig.colorbar(panel, ax=axes, ticks=[2, 3, 4, 5, 6], label="Mean correct features (of 6)")
    fig.suptitle(f"{subtitle}\n"
                 f"{repeats} held-out faces per condition  |  5% pixel flips  |  prior weight {alpha:g}",
                 fontsize=14, color=INK)
    fig.savefig(filename, dpi=210, facecolor="white")
    plt.close(fig)


def plot_contours(output, scores, alpha, repeats):
    main_panels = [(METHODS[key], scores[key]) for key in MAIN_METHODS]
    contour_grid(output / "prior_contours.png", main_panels,
                 "Recovery across visibility and hint accuracy", alpha, repeats)
    seeds = [(f"Initialization seed {seed}", scores[key]) for seed, key in zip(COMPETITIVE_SEEDS, COMPETITIVE_METHODS)]
    average = np.mean([s for _, s in seeds], axis=0)
    contour_grid(output / "competitive_hebbian_contours.png", [("Mean across 3 seeds", average), *seeds],
                 "Label-free competitive Hebbian recovery", alpha, repeats)

    x = np.log10(VISIBILITIES)
    x_grid, y_grid = np.meshgrid(x, list(HINTS))
    comparisons = [
        ("hopfield", "counts", "Hopfield − feature counts"),
        ("hopfield", "kernel", "Hopfield − Hamming kernel"),
        ("hopfield_matched", "kernel", "Entropy-matched Hopfield − Hamming kernel"),
    ]
    diff = [scores[method] - scores[reference] for method, reference, _ in comparisons]
    limit = max(.25, np.ceil(4 * max(np.abs(d).max() for d in diff)) / 4)
    change_colors = LinearSegmentedColormap.from_list("change", ["#B64D15", "#F8F7F1", "#006B67"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    for ax, (_, _, title), delta in zip(axes, comparisons, diff):
        panel = ax.contourf(x, list(HINTS), delta, levels=np.linspace(-limit, limit, 61), cmap=change_colors)
        ax.scatter(x_grid, y_grid, s=8, color=INK, alpha=.45, linewidths=0, clip_on=False)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", fontweight="bold")
        visibility_axis(ax)
    axes[0].set_ylabel("Correct hints (of 6)")
    fig.colorbar(panel, ax=axes, label="Difference in mean correct features")
    fig.suptitle("Change in recovery on the same trials", fontsize=14, color=INK)
    fig.savefig(output / "hopfield_differences.png", dpi=210, facecolor="white")
    plt.close(fig)


def plot_feature_accuracy(output, results):
    values = np.array([[results[key]["feature_accuracy"][feature] for feature in FACTOR_NAMES] for key in METHODS])
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    heat = ax.imshow(values, cmap="PuBuGn", vmin=.3, vmax=1., aspect="auto")
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.1%}", ha="center", va="center",
                color="white" if values[i, j] > .76 else INK)
    ax.set(xticks=range(len(FACTOR_NAMES)), xticklabels=FACTOR_NAMES, yticks=range(len(METHODS)),
           yticklabels=list(METHODS.values()), title="Feature recovery across all visibility and hint conditions")
    fig.colorbar(heat, ax=ax, format=FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.savefig(output / "feature_accuracy.png", dpi=210, facecolor="white")
    plt.close(fig)


def plot_prior_weight(output, rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
    for key, color, marker in (("counts", "#A95525", "o"), ("kernel", "#667D9B", "s"),
                               ("hopfield", "#006B67", "D"), ("competitive_mean", "#88588B", "^")):
        group = [row for row in rows if row["method"] == key]
        x = [row["prior_weight"] for row in group]
        for ax, metric in zip(axes, ("mean_correct_features", "exact_accuracy")):
            ax.plot(x, [row[metric] for row in group], color=color, marker=marker,
                    label=METHODS.get(key, "Competitive Hebbian (3-seed mean)"), linewidth=1.15, markersize=4)
            ax.set_xlabel("Prior weight")
            ax.grid(axis="y", alpha=.22)
    axes[0].set_ylabel("Mean correct features (of 6)")
    axes[1].set_ylabel("Trials with all 6 features correct")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recovery on the same trials at different prior weights", color=INK)
    fig.savefig(output / "prior_weight_sensitivity.png", dpi=210, facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "prior_comparison")
    parser.add_argument("--reference", type=Path, help="Optionally verify against the saved poster trial grid")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--hopfield-beta", type=float, default=1.)
    args = parser.parse_args()
    library = make_library(size=IMAGE_SIZE)
    if not 1 <= args.repeats <= len(library.test_ids) or not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error("Choose a positive number of available held-out faces and a finite nonnegative alpha.")
    priors, training_ids, matched_beta, competitive_diagnostics = fit_conditions(
        library, args.alpha, args.hopfield_beta)
    rows, sensitivity = evaluate(library, priors, args.seed, args.repeats, args.alpha)
    matched_count = check_reference(rows, args.reference)
    scores, cells, results = summarize(rows)
    assert results["hebbian"]["changed_recipes"] == 0
    np.testing.assert_array_equal(scores["counts"], scores["hebbian"])
    for key in ("counts", "kernel", "hopfield", *COMPETITIVE_METHODS):
        at_alpha, = [row for row in sensitivity if row["method"] == key and row["prior_weight"] == args.alpha]
        for metric in ("mean_correct_features", "exact_accuracy"):
            assert at_alpha[metric] == results[key][metric]
    for key in METHODS:
        results[key]["effective_prior_entropy_nats"] = effective_entropy(priors[key], args.alpha)
    for key, diagnostics in competitive_diagnostics.items():
        results[key].update(diagnostics)
    for weight in sorted({r["prior_weight"] for r in sensitivity}):
        group = [r for r in sensitivity if r["method"] in COMPETITIVE_METHODS and r["prior_weight"] == weight]
        sensitivity.append(dict(method="competitive_mean", prior_weight=weight, n=len(group) * group[0]["n"],
                                mean_correct_features=mean(group, "mean_correct_features"),
                                exact_accuracy=mean(group, "exact_accuracy")))
    config = dict(renderer_version=library.renderer_version, candidates=len(library.states),
                  feature_names=FACTOR_NAMES, feature_values=FACTOR_VALUES, image_size=IMAGE_SIZE,
                  library_seed=7, training_seed=11, training_draws=len(training_ids),
                  trial_seed=args.seed, held_out_faces_per_condition=args.repeats,
                  target_visibility_pct=VISIBILITIES, correct_hints=list(HINTS), hints_per_trial=6,
                  hint_tally_disclosed=False, cue_log_bonus_per_match=float(CUE_SCORE),
                  pixel_flip_probability=NOISE, evidence="VSA", encoder_dim=DIMENSION,
                  encoder_seed=ENCODER_SEED, prior_weight=args.alpha,
                  hopfield_beta=args.hopfield_beta, entropy_matched_hopfield_beta=matched_beta,
                  competitive_seeds=COMPETITIVE_SEEDS, competitive_epochs=COMPETITIVE_EPOCHS,
                  competitive_options=COMPETITIVE_OPTIONS, competitive_training_uses_labels=False,
                  verified_poster_baseline_trials=matched_count)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "trial_scores.csv", rows)
    write_csv(args.out / "condition_means.csv", cells)
    write_csv(args.out / "prior_weight_sensitivity.csv", sensitivity)
    (args.out / "summary.json").write_text(json.dumps(dict(config=config, methods=results), indent=2), encoding="utf-8")
    plot_contours(args.out, scores, args.alpha, args.repeats)
    plot_feature_accuracy(args.out, results)
    plot_prior_weight(args.out, sensitivity)
    print(f"{matched_count} poster trials verified. {len(rows)} decisions scored.")
    for key, row in results.items():
        print(f"{METHODS[key]:35s} {row['mean_correct_features']:.3f}/6, "
              f"{row['exact_accuracy']:.1%} exact, "
              f"prior entropy {row['effective_prior_entropy_nats']:.3f} nats")


if __name__ == "__main__":
    main()
