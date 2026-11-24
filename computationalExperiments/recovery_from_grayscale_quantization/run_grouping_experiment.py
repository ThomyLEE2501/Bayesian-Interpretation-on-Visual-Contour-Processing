"""Run the image-only grouping experiment and save per-observation results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

if __package__:
    from .grayscale_regions import sample_scene, render, signal_image, quantize, boundary_survival, encode_regions
else:
    from grayscale_regions import sample_scene, render, signal_image, quantize, boundary_survival, encode_regions


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "recovery_from_grayscale_quantization"


def collect(split, seeds, sizes, contrast, samples, gradient_span, spot_amplitude):
    rows, features = [], []
    for seed in seeds:
        scene = sample_scene(seed, signal_offset=contrast)
        for size in sizes:
            separated, joined, lights = render(scene, size, samples, gradient_span, spot_amplitude)
            for light, field in lights.items():
                for width in (8, 16, 32):
                    for phase in range(0, width, width//4):
                        bg = quantize(field, width, phase)
                        pair = [quantize(signal_image(field, a, scene.signal_offset), width, phase)
                                for a in (separated, joined)]
                        differing = np.count_nonzero(pair[0] != pair[1])
                        total, surviving, _ = boundary_survival(joined, pair[1], bg, joined > separated,
                                                               scene.signal_offset)
                        for label, image in enumerate(pair):
                            if label and not differing:
                                desc = features[-1].copy()
                            else:
                                desc = encode_regions(image, dark_signal=scene.signal_offset < 0)
                            features.append(desc)
                            rows.append(dict(split=split, seed=seed, size=size, light=light,
                                             bin_width=width, phase=phase, joined=label,
                                             diagnostic_pixels=differing, outer_edges=total[0],
                                             outer_survive=surviving[0], bridge_edges=total[1],
                                             bridge_survive=surviving[1]))
    return rows, np.array(features)


def fit_logistic(features, labels, penalty=.01):
    """Average log loss plus ridge on slopes. Intercept is unpenalized."""
    def objective(params):
        z = features @ params[:-1] + params[-1]
        residual = expit(z) - labels
        loss = np.mean(np.logaddexp(0, z) - labels*z) + penalty*(params[:-1] @ params[:-1])/2
        grad = np.r_[features.T @ residual / len(labels) + penalty*params[:-1], residual.mean()]
        return loss, grad
    result = minimize(objective, np.zeros(features.shape[1]+1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def probabilities(x, labels, train, calibration, seed=117):
    mean, sd = x[train].mean(axis=0), np.maximum(x[train].std(axis=0), .01)
    scaled = (x-mean)/sd
    params = fit_logistic(scaled[train], labels[train])
    predictions = {"regions": expit(scaled @ params[:-1]+params[-1])}
    # Each property is a continuous random Fourier value code bound to its bipolar role.
    rng = np.random.default_rng(seed)
    shape = (x.shape[1], 256)
    roles = rng.choice([-1., 1.], size=shape)
    frequencies, phases = rng.normal(size=shape), rng.uniform(0, 2*np.pi, size=shape)
    bundled = np.zeros((len(x), 256))
    for j in range(x.shape[1]):
        bundled += roles[j]*np.cos(scaled[:, j, None]*frequencies[j]+phases[j])
    for dim in (64, 256):
        vectors = bundled[:, :dim] / np.sqrt(dim)
        prototypes = [vectors[train & (labels == c)].mean(axis=0) for c in (0, 1)]
        # Difference of squared distances to the two class means.
        a, b = prototypes
        distance_score = (2*vectors @ (b-a) + a@a - b@b)[:, None]
        slope, intercept = fit_logistic(distance_score[calibration], labels[calibration], penalty=.0001)
        predictions[f"vsa{dim}"] = expit(slope*distance_score[:, 0]+intercept)
    return predictions


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = []
    for size in sorted({r["size"] for r in rows}):
        for group in ("all", "pixel-identical", "distinguishable"):
            subset = [r for r in rows if r["size"] == size and
                      (group == "all" or (r["diagnostic_pixels"] == 0) == (group == "pixel-identical"))]
            if not subset:
                continue
            labels = np.array([r["joined"] for r in subset])
            for model in ("pair_oracle", "regions", "vsa64", "vsa256"):
                p = np.array([r[model] for r in subset])
                correct = (p > .5) == labels
                result.append(dict(size=size, subset=group, model=model, observations=len(p),
                                   accuracy=np.where(p == .5, .5, correct).mean(),
                                   brier=np.mean((p-labels)**2),
                                   mean_confidence=np.maximum(p, 1-p).mean()))
    return result


def run(output=DEFAULT_OUTPUT, fit_count=32, calibration_count=16, test_count=24,
        samples=4, gradient_span=10., spot_amplitude=11., signal_offset=7.):
    if min(fit_count, calibration_count, test_count, samples) < 1:
        raise ValueError("Counts and samples must be positive.")
    if not all(np.isfinite(k) and k >= 0 for k in (gradient_span, spot_amplitude)):
        raise ValueError("Lighting strengths must be finite and nonnegative.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Disjoint seed ranges even if split sizes change.
    n = fit_count + calibration_count
    specs = [("fit", range(fit_count), [128]), ("calibration", range(fit_count,n), [128]),
             ("test", range(n,n+test_count), [64,128,256])]
    rows, arrays = [], []
    for name, seeds, sizes in specs:
        part, desc = collect(name, seeds, sizes, signal_offset, samples, gradient_span, spot_amplitude)
        rows.extend(part)
        arrays.append(desc)
    labels = np.array([r["joined"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    predictions = probabilities(np.concatenate(arrays), labels, splits=="fit", splits=="calibration")
    for i, row in enumerate(rows):
        row["pair_oracle"] = float(row["joined"]) if row["diagnostic_pixels"] else .5
        for key, values in predictions.items():
            row[key] = values[i]
    test = [r for r in rows if r["split"] == "test"]
    summary = summarize(test)
    write_csv(output/"observations.csv", rows)
    write_csv(output/"held_out_summary.csv", summary)
    settings = dict(fit_count=fit_count, calibration_count=calibration_count, test_count=test_count,
                    samples=samples, contrast=signal_offset, gradient_span=gradient_span,
                    spot_amplitude=spot_amplitude, codebook_seed=117)
    (output/"settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return test, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gradient-span", type=float, default=10.)
    parser.add_argument("--spot-amplitude", type=float, default=11.)
    parser.add_argument("--signal-offset", type=float, default=7.,
                        help="Gray levels added to local illumination inside the signal (can be negative).")
    args = parser.parse_args()
    _, summary = run(args.output, gradient_span=args.gradient_span, spot_amplitude=args.spot_amplitude,
                     signal_offset=args.signal_offset)
    for row in summary:
        if row["subset"] == "all":
            print(row)
