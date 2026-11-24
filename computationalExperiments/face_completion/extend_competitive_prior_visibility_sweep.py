"""Add high-visibility conditions to the paired label-free Hebbian run."""
import json
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contour_models.face_recipes_and_observations import make_library
from computationalExperiments.face_completion.compare_priors_under_pixel_flips import (
    ALPHA, COMPETITIVE_METHODS, COMPETITIVE_SEEDS, HINTS, IMAGE_SIZE, NOISE_PCT,
    condition_rows, run, write_csv,
)

VISIBILITIES = (80, 90, 95)


def main():
    root = Path(__file__).resolve().parents[2] / "outputs"
    original = root / "capped_noise_contours"
    output = root / "capped_noise_high_visibility"
    metadata = json.loads((original / "run_summary.json").read_text(encoding="utf-8"))
    faces = metadata["held_out_faces_per_condition"]
    seed = metadata["seed"]
    library = make_library(size=IMAGE_SIZE)
    assert metadata["renderer_version"] == library.renderer_version
    assert metadata["candidate_recipes"] == len(library.states)
    assert metadata["target_flip_pct"] == list(NOISE_PCT)
    assert metadata["correct_hints"] == list(HINTS)
    assert metadata["competitive_seeds"] == list(COMPETITIVE_SEEDS)
    assert metadata["prior_weight"] == ALPHA

    scores, visible, flips, max_noise = run(
        library, faces, seed, visibilities=VISIBILITIES, include_counts=False,
    )
    pooled = scores.mean(axis=0)
    by_method = dict(zip(COMPETITIVE_METHODS, scores))
    by_method["competitive_mean"] = pooled
    summary = {
        "target_visible_pct": VISIBILITIES,
        "held_out_faces_per_condition": faces,
        "seed": seed,
        "renderer_version": library.renderer_version,
        "competitive_seeds": list(COMPETITIVE_SEEDS),
        "target_flip_pct": list(NOISE_PCT),
        "max_realized_flip_pct": 100 * max_noise,
        "correct_hints": list(HINTS),
        "prior_weight": ALPHA,
        "mean_by_visibility": {v: float(m) for v, m in zip(VISIBILITIES, pooled.mean(axis=(0,1)))},
        "mean_by_flip_pct": {n: float(m) for n, m in zip(NOISE_PCT, pooled.mean(axis=(1,2)))},
        "mean_correct_features": float(pooled.mean()),
    }
    output.mkdir(parents=True, exist_ok=True)
    rows = condition_rows(by_method, VISIBILITIES, faces, visible, flips)
    write_csv(output / "condition_means.csv", rows)
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
