"""Plot six-feature scores, generate MAP trials, or replay a participant's session."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closure_model import FACTOR_VALUES, make_library, sample_training, fit_priors
from psychophysicalExperiment.face_study import (NOISE, VISIBILITY, POSTER, observation,
                                                trial_cues, score_row, write_rows)
from vsa_inference import PixelVSA

AXES = dict(visibility_pct="Requested visibility (%)", actual_visibility_pct="Visible pixels (%)",
            cues_given="Number of hints", correct_cues="Number of correct hints",
            incorrect_cues="Number of incorrect hints", hint_counts_shown="Hint counts disclosed (0 / 1)",
            noise_pct="Pixel flip probability (%)",
            flipped_pct="Flipped among visible pixels (%)", prior_weight="Prior weight (alpha)",
            cue_weight="Cue weight (beta)", trial="Trial number", response_seconds="Response time (s)")
SCORE_COLORS = ("#873B18", POSTER["orange"], "#DA8443", "#ECA15D", "#F7C797", "#50BFC8", POSTER["teal"])
COLORS = LinearSegmentedColormap.from_list("mean_score", SCORE_COLORS)
DOT_COLORS = ListedColormap(SCORE_COLORS)
VISIBILITY_AXES = {"visibility_pct", "actual_visibility_pct"}
EXAMPLE_PRIOR_WEIGHT = 12.
FIGURES = Path(__file__).resolve().parents[1] / "outputs" / "psychophysics" / "accuracy_contours"


def contour(rows, x, y, output, title, filters=None, log_visibility=False):
    if x == y or x not in AXES or y not in AXES:
        raise ValueError("Choose two different numeric axes from " + ", ".join(AXES))
    log_x, log_y = (log_visibility and key in VISIBILITY_AXES for key in (x, y))
    if log_visibility and not (log_x or log_y):
        raise ValueError("Select a visibility axis to use logarithmic visibility.")
    groups = defaultdict(list)
    omitted = 0
    for r in rows:
        if any(str(r.get(k)) != str(v) and not numeric_equal(r.get(k), v)
               for k, v in (filters or {}).items()):
            continue
        if r.get(x) in ("", None) or r.get(y) in ("", None):
            continue
        a, b, value = float(r[x]), float(r[y]), int(r["score"])
        if not 0 <= value <= 6:
            raise ValueError("Scores must be integer counts from 0 through 6.")
        if np.isfinite(a) and np.isfinite(b):
            if (log_x and a <= 0) or (log_y and b <= 0):
                omitted += 1
                continue
            # Interpolate and offset dots in the same coordinates the reader sees.
            groups[np.log10(a) if log_x else a, np.log10(b) if log_y else b].append(value)
    if not groups:
        if omitted:
            raise ValueError("Log visibility requires a positive percentage. Use the linear scale for zero visibility.")
        raise ValueError("No measured rows have both selected axes and filters.")
    coords = np.array(sorted(groups))
    means = np.array([np.mean(groups[tuple(p)]) for p in coords])
    xs, ys = coords.T
    fig, ax = plt.subplots(figsize=(9.2, 6.3), layout="constrained")
    surface = False
    if len(coords) >= 3 and np.linalg.matrix_rank(coords - coords.mean(axis=0)) == 2:
        mesh = tri.Triangulation(xs, ys)
        ax.tricontourf(mesh, means, levels=np.linspace(0, 6, 49), cmap=COLORS,
                       vmin=0, vmax=6, alpha=.53)
        levels = [s for s in range(1, 6) if min(means) < s < max(means)]
        if levels:
            lines = ax.tricontour(mesh, means, levels=levels, colors=POSTER["ink"], linewidths=.8)
            ax.clabel(lines, fmt="%d", fontsize=9)
        surface = True
    # Limit jitter by the nearest neighboring measured coordinate on each axis.
    def spread(vals):
        values = np.unique(vals)
        if len(values) == 1:
            return {values[0]: 0.}
        gaps = np.diff(values)
        widths = np.minimum(np.r_[gaps[0], gaps], np.r_[gaps, gaps[-1]])
        return dict(zip(values, .18*widths))
    rng = np.random.default_rng(812)
    dx, dy = spread(xs), spread(ys)
    px, py, colors = [], [], []
    for (a, b), vals in groups.items():
        delta = rng.uniform(-1, 1, size=(len(vals), 2)) if len(vals)>1 else np.zeros((1, 2))
        px.extend(a+dx[a]*delta[:, 0])
        py.extend(b+dy[b]*delta[:, 1])
        colors.extend(vals)
    norm = BoundaryNorm(np.arange(-.5, 7), DOT_COLORS.N)
    dots = ax.scatter(px, py, c=colors, cmap=DOT_COLORS, norm=norm, s=24 if len(colors)<500 else 12,
                      edgecolor=POSTER["ink"], linewidth=.32, alpha=.94)
    bar = fig.colorbar(dots, ax=ax, ticks=range(7), shrink=.85)
    bar.set_label("Correctly recovered features (of 6)")
    ax.set(xlabel=AXES[x] + (" · log scale" if log_x else ""),
           ylabel=AXES[y] + (" · log scale" if log_y else ""))
    details = " · ".join(f"{k}={v}" for k,v in (filters or {}).items())
    fig.suptitle(title, fontweight="bold", color=POSTER["ink"])
    ax.set_title((details+"\n" if details else "") +
                 f"{len(colors):,} scored responses · dots offset at repeated coordinates"+
                 ("\nContours interpolate the mean at measured coordinates" if surface else "")+
                 (f"\n{omitted} responses at zero visibility omitted from the log scale" if omitted else ""),
                 fontsize=9)
    for axis, vals, logged in ((ax.xaxis, xs, log_x), (ax.yaxis, ys, log_y)):
        distinct = np.unique(vals)
        if logged:
            ticks = distinct if len(distinct) <= 10 else np.linspace(distinct[0], distinct[-1], 6)
            axis.set_ticks(ticks, labels=[f"{10**v:.3g}" for v in ticks])
        elif len(distinct)<=10:
            axis.set_ticks(distinct)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    fig.savefig(tmp, format="png", dpi=180)
    tmp.replace(output)
    plt.close(fig)


def numeric_equal(a, b):
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False


def sweep_trials(lib, seed, repeats, show_counts=False):
    rng = np.random.default_rng(seed)
    # One recipe, flip field and set of wrong alternatives per repetition. All
    # visibility and noise thresholds share those random draws across the grid.
    ids = rng.choice(lib.test_ids, repeats, replace=False)
    for rep, idx in enumerate(ids):
        stim, cue = map(int, rng.integers(2**32, size=2))
        for v,n in product(VISIBILITY, NOISE):
            yield dict(trial=rep+1, state_id=int(idx), visibility=v, noise=n,
                       stimulus_seed=stim, cue_seed=cue, hint_counts_shown=show_counts)


def simulate(trials, lib, weights, beta, output, replay=False):
    bank = fit_priors(lib, *sample_training(lib), category_learning="counts")
    encoder = PixelVSA(40, 512, seed=17)
    records = []
    for trial in trials:
        observed, mask = observation(lib, trial)
        ll = encoder.evidence_scores(lib.images, observed, mask, trial["noise"], "vsa")
        state = lib.states[trial["state_id"]]
        for k in ([trial["correct_cues"]] if replay else range(7)):
            cue_trial = dict(trial, correct_cues=k)
            cues = trial_cues(state, cue_trial)
            matches = 0 if cues is None else (lib.states == cues).sum(axis=1)
            if trial.get("hint_counts_shown", False):
                # Given a truthful tally and exchangeable hint positions, each
                # recipe satisfying the tally has the same hint likelihood.
                evidence = np.where(matches == k, ll, -np.inf)
            else:
                evidence = ll + beta*np.log(.95/.025)*matches
            baseline = int(np.argmax(evidence))
            for alpha in weights:
                winner = int(np.argmax(evidence + alpha*bank.logpriors["category"]))
                row = score_row(lib, trial, lib.states[winner], cues, observed, mask)
                row.update(source="model", prior_weight=alpha,
                           cue_weight="" if trial.get("hint_counts_shown", False) else beta,
                           map_state=winner, no_prior_map=baseline, map_changed_from_no_prior=int(winner != baseline))
                records.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_rows(Path(output), records)
    return records


def prior_effects(rows, output):
    """Compare each decision to alpha zero on the identical face, pixels and hints."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["visibility_pct"], row["prior_weight"]].append(row)
    summary = [dict(visibility_pct=v, prior_weight=a, responses=len(group),
                    changed_recipes=sum(r["map_changed_from_no_prior"] for r in group),
                    changed_pct=100*np.mean([r["map_changed_from_no_prior"] for r in group]),
                    mean_score=np.mean([r["score"] for r in group]))
               for (v, a), group in sorted(groups.items())]
    write_rows(output/"prior_effect.csv", summary)
    fig, ax = plt.subplots(figsize=(8, 4.9), layout="constrained")
    for v in sorted({r["visibility_pct"] for r in summary}):
        series = [r for r in summary if r["visibility_pct"] == v]
        ax.plot([r["prior_weight"] for r in series], [r["changed_pct"] for r in series],
                marker="o", markersize=4, label=f"{v:g}%")
    ax.set(xlabel="Prior weight (alpha)", ylabel="MAP recipes changed from alpha = 0 (%)", ylim=(-3, 103),
           title="How often does the prior change the chosen recipe?")
    ax.legend(title="Requested visibility", ncols=2)
    ax.grid(axis="y", alpha=.2)
    tmp = output/"prior_effect.tmp"
    fig.savefig(tmp, format="png", dpi=180)
    tmp.replace(output/"prior_effect.png")
    plt.close(fig)


def import_mixed(folder, output):
    """Recover all-six MAP accuracy from the saved three-statement experiment."""
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf8"))
    lib = make_library()
    if manifest["candidates"] != len(lib.states) or manifest["factor_values"] != [list(v) for v in FACTOR_VALUES]:
        raise ValueError("The saved experiment uses a different recipe vocabulary.")
    records = []
    with (folder/"metrics.csv").open(encoding="utf8") as handle:
        for r in csv.DictReader(handle):
            target, choice = lib.states[[int(r["state_id"]), int(r["map_state"])]]
            pattern = r["pattern"]
            records.append(dict(case=r["case"], state_id=r["state_id"], map_state=r["map_state"],
                source="model", score=int(np.sum(choice == target)), visibility_pct=100*float(r["requested_visibility"]),
                actual_visibility_pct=100*float(r["actual_visibility"]), noise_pct=100*manifest["noise"],
                cues_given=0 if pattern=="None" else len(pattern),
                correct_cues=pattern.count("C"), incorrect_cues=pattern.count("W"), hint_counts_shown=0, pattern=pattern,
                prior_weight=manifest["prior_strength"], cue_weight=float(r["cue_weight"])))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"mixed_scores.csv", records)
    contour(records, "visibility_pct", "correct_cues", output/"previous_mixed_cues.png",
            "MAP recipe accuracy with three feature statements", {"cue_weight":3, "cues_given":3})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mixed = commands.add_parser("import-mixed", help="Replot a saved mixed feature cue experiment")
    mixed.add_argument("folder", type=Path)
    mixed.add_argument("--out", type=Path, default=FIGURES)
    sim = commands.add_parser("simulate", help="Generate matched six-hint MAP trials and example plots")
    sim.add_argument("--out", type=Path, default=FIGURES)
    sim.add_argument("--seed", type=int, default=172)
    sim.add_argument("--repeats", type=int, default=12)
    sim.add_argument("--beta", type=float, default=1.)
    sim.add_argument("--prior-weights", type=float, nargs="+", default=[0., 1., 3., 8., 12., 16., 24.])
    sim.add_argument("--show-hint-counts", action="store_true", help="Give the model the exact correct-hint tally")
    replay = commands.add_parser("replay", help="Score the exact inputs and cues from a human session")
    replay.add_argument("session", type=Path)
    replay.add_argument("--alpha", type=float, default=EXAMPLE_PRIOR_WEIGHT)
    replay.add_argument("--beta", type=float, default=1.)
    plot = commands.add_parser("plot", help="Plot participant or model CSV files together by condition")
    plot.add_argument("csv", type=Path, nargs="+")
    plot.add_argument("--x", required=True, choices=AXES)
    plot.add_argument("--y", required=True, choices=AXES)
    plot.add_argument("--where", action="append", default=[], metavar="COLUMN=VALUE")
    plot.add_argument("--title", default="Face recipe accuracy")
    plot.add_argument("--log-visibility", action="store_true", help="Use logarithmic spacing on visibility axes")
    plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command=="import-mixed":
        import_mixed(args.folder, args.out)
        return
    if args.command=="plot":
        rows = []
        for path in args.csv:
            with path.open(encoding="utf-8") as file:
                rows.extend(csv.DictReader(file))
        contour(rows, args.x, args.y, args.out, args.title, dict(s.split("=",1) for s in args.where),
                log_visibility=args.log_visibility)
        return
    if args.command=="replay":
        session = json.loads((args.session/"session.json").read_text(encoding="utf8"))
        lib = make_library(renderer_version=session.get("renderer_version", 1))
        with (args.session/"responses.csv").open(encoding="utf8") as f:
            completed = {int(row["trial"]) for row in csv.DictReader(f)}
        trials = [t for t in session["trials"] if t["trial"] in completed]
        simulate(trials, lib, [args.alpha], args.beta, args.session/"model_replay.csv", replay=True)
        return
    lib = make_library()
    rows = simulate(sweep_trials(lib, args.seed, args.repeats, args.show_hint_counts), lib, args.prior_weights, args.beta,
                    args.out/"model_scores.csv")
    alpha = min(args.prior_weights, key=lambda a: abs(a-EXAMPLE_PRIOR_WEIGHT))
    for x,y,filters,name in [
        ("visibility_pct", "correct_cues", dict(prior_weight=alpha, noise_pct=5), "visibility_by_cues"),
        ("visibility_pct", "noise_pct", dict(prior_weight=alpha, correct_cues=3), "visibility_by_noise"),
        ("visibility_pct", "prior_weight", dict(noise_pct=5, correct_cues=3), "visibility_by_prior")]:
        contour(rows, x, y, args.out/(name+".png"), "MAP recipe accuracy", filters)
        if name == "visibility_by_cues":
            contour(rows, x, y, args.out/(name+"_log.png"), "MAP recipe accuracy", filters, log_visibility=True)
    prior_effects(rows, args.out)
    print(f"Saved {len(rows):,} MAP responses, the paired prior comparison and five figures to {args.out}")


if __name__ == "__main__":
    main()
