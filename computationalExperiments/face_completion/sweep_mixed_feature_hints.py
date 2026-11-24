"""Sweep the weight of three feature hints. The saved mixed-cue notebook calls run()."""

from itertools import product
from pathlib import Path
import csv
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contour_models.face_recipes_and_observations import (
    CARDINALITIES, CLASS_NAMES, FACTOR_NAMES, FACTOR_VALUES, block_mask,
    completion_metrics, describe_state, fit_priors, make_library, observe,
    prediction, sample_training, show_completion, show_observation,
)
from contour_models.vsa_inference import PixelVSA, entropy, factor_marginals, posterior


FEATURES = (5, 1, 3)  # eye shape, ear shape, nose shape
PATTERNS = ("None",) + tuple("".join(flags) for flags in product("CW", repeat=3))
DEFAULT_WEIGHTS = tuple(np.arange(0., 12.5, .5)) + (16., 20., 24.)
DEFAULT_FACE_WEIGHTS = (0., 1., 3., 8., 24.)


def combined_cue_scores(states, statements, reliability):
    """Sum log likelihoods of conditionally independent feature statements.

    Each (feature, value) pair occurs once. Reliability is the probability of
    naming the true value, with errors spread over the remaining alternatives.
    """
    if not 0 < reliability < 1:
        raise ValueError("Cue reliability must lie strictly between zero and one.")
    if len({feature for feature, _ in statements}) != len(statements):
        raise ValueError("Use at most one statement per feature.")
    scores = np.zeros(len(states))
    for feature, value in statements:
        scores += np.log(np.where(
            states[:, feature] == value, reliability,
            (1 - reliability) / (CARDINALITIES[feature] - 1),
        ))
    return scores


def save_face(path, ink_probability):
    """Save an unlabeled 400px PNG, preserving the original pixel grid."""
    grayscale = np.rint(255 * (1 - np.clip(ink_probability, 0, 1))).astype(np.uint8)
    image = Image.fromarray(grayscale)
    image.resize((400, 400), Image.Resampling.NEAREST).save(path)


def save_observation(path, observed, mask):
    rgb = np.repeat((1 - observed)[..., None], 3, axis=-1)
    rgb[~mask] = (.72, .80, .88)
    image = Image.fromarray(np.rint(255 * rgb).astype(np.uint8))
    image.resize((400, 400), Image.Resampling.NEAREST).save(path)


def weight_slug(weight):
    return f"weight_{weight:g}".replace(".", "p")


def plot_sweep(path, title, records):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = (
        ("p_original", "P(original complete recipe)"),
        ("hidden_brier", "Hidden Brier error"),
        ("map_visible_mismatch", "MAP visible mismatch fraction"),
        ("map_target_accuracy", "Fraction of 3 target features correct in MAP"),
    )
    colors = ["black", *plt.get_cmap("tab10").colors[:8]]
    for pattern, color in zip(PATTERNS, colors):
        selected = [row for row in records if row["pattern"] == pattern]
        weights = [row["cue_weight"] for row in selected]
        for ax, (metric, ylabel) in zip(axes.flat, metrics):
            values = [row[metric] for row in selected]
            # MAP panels show sampled points because switches are discrete.
            style = "none" if metric.startswith("map_") else "-"
            ax.plot(weights, values, color=color, linestyle=style, marker="o",
                    markersize=3, markerfacecolor="none", label=pattern)
            ax.set(xlabel="Cue weight beta", ylabel=ylabel)
            ax.grid(alpha=.2)
    axes[0, 0].set_ylim(-.03, 1.03)
    axes[1, 1].set_ylim(-.03, 1.03)
    axes[0, 1].set_ylim(bottom=0)
    axes[1, 0].set_ylim(bottom=0)
    fig.suptitle(title + "\nOrder: eyes / ears / nose. C correct, W wrong. None = no statements.")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(output_dir, visibilities=(.15, .35, .65), seeds=(17, 37, 67),
        cue_weights=DEFAULT_WEIGHTS, face_weights=DEFAULT_FACE_WEIGHTS,
        reliability=.95, prior_strength=1., noise=.05):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cue_weights = tuple(sorted(set(float(w) for w in cue_weights)))
    face_weights = tuple(float(w) for w in face_weights)
    if not cue_weights or any(not np.isfinite(w) or w < 0 for w in cue_weights):
        raise ValueError("Cue weights must be finite and nonnegative.")
    if not set(face_weights).issubset(cue_weights):
        raise ValueError("Every saved face weight must also occur in the sweep.")
    if len(seeds) != len(CLASS_NAMES):
        raise ValueError("Supply one observation seed per category.")
    if any(not 0 <= v <= 1 for v in visibilities):
        raise ValueError("Visibility must lie between zero and one.")

    library = make_library(size=40, seed=7)
    training_ids, training_labels = sample_training(library, 250, seed=11)
    bank = fit_priors(library, training_ids, training_labels, category_learning="counts")
    encoder = PixelVSA(40, 512, seed=17)
    assert set(training_ids).isdisjoint(library.test_ids)
    all_records, cases, face_manifest = [], [], []

    for category, seed in enumerate(seeds):
        state_id = int(library.test_ids[np.argmax(library.teacher_logp[category, library.test_ids])])
        state = library.states[state_id]
        truth = library.images[state_id]
        wrong_values = {feature: (int(state[feature]) + 1) % CARDINALITIES[feature]
                        for feature in FEATURES}
        statements_by_pattern = {"None": []}
        for pattern in PATTERNS[1:]:
            statements_by_pattern[pattern] = [
                (feature, int(state[feature]) if flag == "C" else wrong_values[feature])
                for feature, flag in zip(FEATURES, pattern)
            ]
        cue_scores = {pattern: combined_cue_scores(library.states, statements, reliability)
                      for pattern, statements in statements_by_pattern.items()}
        for pattern, statements in statements_by_pattern.items():
            np.testing.assert_allclose(cue_scores[pattern], combined_cue_scores(
                library.states, list(reversed(statements)), reliability))

        for visibility in visibilities:
            slug = f"{CLASS_NAMES[category]}_seed{seed}_visible{round(100*visibility):02d}"
            folder = output_dir / slug
            folder.mkdir(exist_ok=True)
            rng = np.random.default_rng(seed)
            mask = block_mask(40, visibility, rng)
            observed = observe(truth, mask, noise, rng)
            likelihood = encoder.evidence_scores(library.images, observed, mask, noise, "vsa")
            baseline = posterior(likelihood, bank.logpriors["category"], prior_strength)
            save_face(folder / "truth.png", truth)
            save_observation(folder / "observation.png", observed, mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "visibility_mask.png")
            np.savez_compressed(folder / "input_arrays.npz", truth=truth, observed=observed, mask=mask)
            case = dict(case=slug, category=CLASS_NAMES[category], state_id=state_id,
                        seed=seed, requested_visibility=visibility, actual_visibility=float(mask.mean()),
                        recipe=describe_state(state), statements={pattern: {
                            FACTOR_NAMES[f]: FACTOR_VALUES[f][v] for f, v in statements
                        } for pattern, statements in statements_by_pattern.items()})
            cases.append(case)
            records, saved_predictions = [], {}

            for pattern in PATTERNS:
                statements = statements_by_pattern[pattern]
                distributions = np.stack([
                    posterior(likelihood + weight * cue_scores[pattern],
                              bank.logpriors["category"], prior_strength)
                    for weight in cue_weights
                ])
                np.testing.assert_allclose(distributions.sum(axis=1), 1, atol=1e-12)
                predictions = (distributions @ library.images.reshape(len(library.images), -1)).reshape(-1, 40, 40)
                for weight, q, pred in zip(cue_weights, distributions, predictions):
                    if weight == 0 or pattern == "None":
                        np.testing.assert_allclose(q, baseline, atol=1e-12)
                    winner = int(np.argmax(q))
                    marginals = factor_marginals(q, library.states, CARDINALITIES)
                    metrics = completion_metrics(pred, truth, observed, mask)
                    row = dict(case=slug, category=CLASS_NAMES[category],
                               requested_visibility=visibility, actual_visibility=float(mask.mean()),
                               pattern=pattern, cue_weight=weight, state_id=state_id,
                               map_state=winner, p_original=float(q[state_id]),
                               map_probability=float(q[winner]), posterior_entropy=float(entropy(q)),
                               map_visible_mismatch=float(np.mean(library.images[winner][mask] != observed[mask])) if mask.any() else float("nan"),
                               map_target_accuracy=float(np.mean(library.states[winner, list(FEATURES)] == state[list(FEATURES)])),
                               **metrics)
                    for feature in FEATURES:
                        key = FACTOR_NAMES[feature].replace(" ", "_")
                        row[f"p_true_{key}"] = float(marginals[feature][state[feature]])
                        row[f"map_{key}"] = FACTOR_VALUES[feature][library.states[winner, feature]]
                    records.append(row)
                    if weight in face_weights:
                        destination = folder / "faces" / pattern / weight_slug(weight)
                        destination.mkdir(parents=True, exist_ok=True)
                        save_face(destination / "posterior_mean.png", pred)
                        save_face(destination / "map.png", library.images[winner])
                        saved_predictions[pattern, weight] = (pred, library.images[winner])
                        face_manifest.append(dict(case=slug, pattern=pattern, cue_weight=weight,
                            mean_png=str((destination / "posterior_mean.png").relative_to(output_dir)).replace("\\", "/"),
                            map_png=str((destination / "map.png").relative_to(output_dir)).replace("\\", "/"),
                            map_recipe=describe_state(library.states[winner])))

            all_records.extend(records)
            title = f"{CLASS_NAMES[category]} | seed {seed} | {mask.mean():.1%} visible"
            plot_sweep(folder / "cue_weight_sweep.png", title, records)
            for kind, prediction_index in (("posterior_mean", 0), ("map", 1)):
                fig, axes = plt.subplots(len(PATTERNS), 2 + len(face_weights),
                                         figsize=(2 * (2 + len(face_weights)), 16), squeeze=False)
                for row_index, pattern in enumerate(PATTERNS):
                    show_completion(axes[row_index, 0], truth, "Truth" if row_index == 0 else "")
                    show_observation(axes[row_index, 1], observed, mask, "Observed" if row_index == 0 else "")
                    for col, weight in enumerate(face_weights, start=2):
                        show_completion(axes[row_index, col], saved_predictions[pattern, weight][prediction_index],
                                        f"beta={weight:g}" if row_index == 0 else "")
                    axes[row_index, 0].text(-.12, .5, pattern, transform=axes[row_index, 0].transAxes,
                                             ha="right", va="center", fontsize=11)
                fig.suptitle(title + f" | {kind.replace('_', ' ')}\n"
                             "Cue order: eyes / ears / nose. C correct, W wrong. None = no statements.")
                fig.tight_layout(rect=(.03, 0, 1, .96))
                fig.savefig(folder / f"faces_{kind}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            print(f"Finished {slug}", flush=True)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    config = dict(candidates=len(library.states), features=[FACTOR_NAMES[f] for f in FEATURES],
                  pattern_order=list(PATTERNS), reliability=reliability, prior_strength=prior_strength,
                  cue_weights=cue_weights, face_weights=face_weights, noise=noise,
                  factor_names=FACTOR_NAMES, factor_values=FACTOR_VALUES,
                  model_sha256=hashlib.sha256((Path(__file__).resolve().parents[2] / "contour_models" / "face_recipes_and_observations.py").read_bytes()).hexdigest(),
                  cases=cases, faces=face_manifest)
    (output_dir / "manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} cases, {len(all_records)} metric rows, "
          f"and {2*len(face_manifest)} individual reconstruction PNGs.")
    return config
