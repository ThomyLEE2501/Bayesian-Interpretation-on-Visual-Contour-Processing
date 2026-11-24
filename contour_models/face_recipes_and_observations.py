"""Synthetic, auditable VSA closure experiment; see the companion notebook.

Pixels are the only test-time observations. Generator labels and latent factors
are used to create training/evaluation data, never to encode a test observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from PIL import Image, ImageDraw
from scipy.special import logsumexp
from contour_models.hebbian_category_priors import SupervisedHebbianPrior, CompetitiveHebbianPrior
from contour_models.hopfield_exemplar_prior import HopfieldExemplarPrior

FACTOR_NAMES = ("head shape", "ear shape", "eye spacing", "nose shape", "layout", "eye shape")
FACTOR_VALUES = (
    ("round", "wide", "tall"),
    ("round", "pointed", "long"),
    ("narrow", "medium", "wide"),
    ("triangle", "round", "muzzle"),
    ("aligned", "lowered", "scrambled"),
    ("dots", "arches", "oval pupils"),
)
CARDINALITIES = tuple(map(len, FACTOR_VALUES))
CLASS_NAMES = ("cow-like", "cat-like", "human-like")
PRIOR_NAMES = ("uniform", "generic", "category", "exemplar")
PRIOR_LABELS = {
    "uniform": "Uniform vocabulary",
    "generic": "Layout prior",
    "category": "Latent categories",
    "exemplar": "Training exemplars",
}
GROUPS = {"outline": (0,), "ears": (1,), "eyes": (2, 4, 5), "nose": (3, 4)}
RENDERER_VERSION = 2


def normalize_logp(values):
    values = np.asarray(values, dtype=float)
    return values - logsumexp(values)


def describe_state(state):
    return {name: FACTOR_VALUES[j][int(v)] for j, (name, v) in enumerate(zip(FACTOR_NAMES, state))}


def render_state(state, size=40, *, parts=None, renderer_version=RENDERER_VERSION):
    """Binary line drawing. All geometry and primitive restrictions are explicit.

    These are cartoons, not thresholded photographs or a learned visual encoder.
    All head outlines are closed. Eye shape and layout vary independently.
    """
    if renderer_version not in (1, 2):
        raise ValueError(f"Unknown face renderer version {renderer_version}")
    head, ears, gap, nose, layout, eye_shape = map(int, state)
    im = Image.new("L", (size, size), 0)
    ink = ImageDraw.Draw(im)
    # Feature-guide icons use the same geometry as the trial stimuli.
    excluded = ImageDraw.Draw(Image.new("L", im.size)) if parts is not None else None
    def layer(name):
        return ink if parts is None or name in parts else excluded
    scale = size - 1
    width = max(1, round(size / 32))

    def box(v):
        return tuple(round(x * scale) for x in v)

    def points(v):
        return [(round(x * scale), round(y * scale)) for x, y in v]

    # Ears are part of the same candidate vocabulary in every prior condition.
    d = layer("ears")
    if ears == 0:
        for b in ((.13, .38, .28, .57), (.72, .38, .87, .57)):
            d.ellipse(box(b), outline=1, width=width)
    elif ears == 1:
        for p in (((.21, .35), (.17, .04), (.40, .22)),
                  ((.79, .35), (.83, .04), (.60, .22))):
            pp = points(p)
            d.line(pp + pp[:1], fill=1, width=width)
    else:
        for p in (((.27, .34), (.02, .20), (.06, .37), (.25, .43)),
                  ((.73, .34), (.98, .20), (.94, .37), (.75, .43))):
            pp = points(p)
            d.line(pp + pp[:1], fill=1, width=width)

    bounds = ((.20, .18, .80, .94), (.10, .24, .90, .90), (.27, .12, .73, .96))[head]
    d = layer("outline")
    d.ellipse(box(bounds), outline=1, width=width)

    eye_dx = (.11, .17, .23)[gap]
    d = layer("eyes")
    eye_y = (.43, .51, .34)[layout]
    r = .025
    for side in (-1, 1):
        x = .5 + side * eye_dx
        y = .63 if layout == 2 and side == 1 else eye_y
        if eye_shape == 0 and renderer_version == 2:
            # Mirror the complete dot footprint about the midline. These three
            # spacings leave white space inside even the narrowest head.
            diameter = max(1, round(.05 * size))
            left = round((.43, .38, .33)[gap] * scale)
            if side == 1:
                left = size - left - diameter
            dot_y = (.44, .50, .55 if side == -1 else .62)[layout]
            top = round(dot_y * scale)
            dot_box = (left, top, left+diameter-1, top+diameter-1)
            if diameter <= 2:
                d.rectangle(dot_box, fill=1)
            else:
                d.ellipse(dot_box, fill=1)
        elif eye_shape == 0:
            d.ellipse(box((x-r, y-r, x+r, y+r)), fill=1)
        elif eye_shape == 1:
            d.arc(box((x-.06, y-.05, x+.06, y+.05)),
                  start=180, end=360, fill=1, width=width)
        else:
            d.ellipse(box((x-.06, y-.08, x+.06, y+.08)), outline=1, width=width)
            d.line(points(((x, y-.08), (x, y+.08))), fill=1, width=width)
    ny = (.64, .69, .37)[layout]
    d = layer("nose")
    if nose == 0:
        d.polygon(points(((.44, ny-.04), (.56, ny-.04), (.50, ny+.045))), fill=1)
    elif nose == 1:
        d.ellipse(box((.455, ny-.06, .545, ny+.06)), outline=1, width=width)
    else:
        d.ellipse(box((.32, ny-.09, .68, ny+.09)), outline=1, width=width)
        for x in (.42, .58):
            d.ellipse(box((x-.022, ny-.025, x+.022, ny+.025)), fill=1)
    my = (.78, .84, .70)[layout]
    d = layer("mouth")
    d.arc(box((.40, my-.04, .60, my+.04)), start=0, end=180, fill=1, width=width)
    return np.asarray(im, dtype=np.float32)


@dataclass
class Library:
    states: np.ndarray
    images: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    teacher_logp: np.ndarray
    renderer_version: int = RENDERER_VERSION


def make_library(size=40, seed=7, test_fraction=.25, *, renderer_version=RENDERER_VERSION):
    states = np.array(list(product(*(range(c) for c in CARDINALITIES))), dtype=int)
    images = np.stack([render_state(s, size, renderer_version=renderer_version) for s in states])
    # Split combinations before sampling training examples. The renderer's
    # vocabulary is known; test combinations are never training exemplars.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(states))
    n_test = round(test_fraction * len(states))
    test_ids, train_ids = np.sort(order[:n_test]), np.sort(order[n_test:])
    # Preferred eyes: cow dots, cat oval pupils, human arches.
    modes = ((1, 2, 2, 2, 0, 0), (0, 1, 1, 0, 0, 2), (2, 0, 0, 1, 0, 1))
    teacher = []
    for mode in modes:
        lp = np.zeros(len(states))
        for j, cardinality in enumerate(CARDINALITIES):
            probs = np.full(cardinality, .28 / (cardinality-1))
            probs[mode[j]] = .72
            if j == 4:
                probs = np.array([.76, .20, .04])
            lp += np.log(probs[states[:, j]])
        teacher.append(normalize_logp(lp))
    return Library(states, images, train_ids, test_ids, np.stack(teacher), renderer_version)


def sample_training(library, n_per_class=250, seed=11):
    rng = np.random.default_rng(seed)
    ids, labels = [], []
    for c in range(len(CLASS_NAMES)):
        weights = np.exp(normalize_logp(library.teacher_logp[c, library.train_ids]))
        chosen = rng.choice(library.train_ids, n_per_class, p=weights)
        ids.extend(chosen)
        labels.extend([c] * n_per_class)
    return np.asarray(ids), np.asarray(labels)


@dataclass
class PriorBank:
    logpriors: dict
    class_logp: np.ndarray
    class_weights: np.ndarray
    training_ids: np.ndarray
    exemplar_model: HopfieldExemplarPrior | None = None


def fit_priors(library, training_ids, training_labels=None, smoothing=1., exemplar_precision=2.5,
               category_learning="hebbian", hebbian_learning_rate=1.,
               competitive_units=3, competitive_epochs=20, competitive_seed=0,
               competitive_learning_rate=.03, competitive_temperature=.7,
               competitive_homeostasis=2., exemplar_learning="kernel",
               hopfield_inverse_temperature=1.):
    """Learn only from supplied training factors/categories, not teacher tables.

    Counts and Hebbian fitting use supplied category labels. Competitive
    fitting learns anonymous units from features, with category identity latent.
    Exemplar memory uses a Hamming-kernel density over TRAINING configurations.
    The four families are alternative inductive biases, not a strict total order.
    The shared fitter defaults to supervised online Hebbian category learning.
    Use category_learning="counts" for the original batch-count reference.
    category_learning="competitive" ignores training_labels and learns anonymous
    units. Their indices have no supplied category meaning. The unit count is
    an explicit hyperparameter. The other prior families receive no labels.
    exemplar_learning="hopfield" replaces the exemplar score with normalized
    Hebbian energy. The original Hamming kernel remains the default reference.
    """
    states = library.states
    training_ids, training_labels = np.asarray(training_ids), np.asarray(training_labels)
    if not set(training_ids).issubset(set(library.train_ids)):
        raise ValueError("Training ids must belong to the training split.")
    samples = states[training_ids]
    if category_learning in ("hebbian", "counts") and training_labels.ndim == 0:
        raise ValueError("Supervised fitting requires one category label per training draw.")
    layout_counts = np.bincount(samples[:, 4], minlength=CARDINALITIES[4]) + smoothing
    generic = np.log(layout_counts / layout_counts.sum())[states[:, 4]]
    if category_learning == "competitive":
        learner = CompetitiveHebbianPrior(
            CARDINALITIES, n_units=competitive_units,
            learning_rate=competitive_learning_rate, temperature=competitive_temperature,
            homeostasis=competitive_homeostasis, seed=competitive_seed)
        learner.fit(samples, epochs=competitive_epochs, shuffle_seed=competitive_seed)
        class_lp = learner.class_log_probabilities(states)
        weights = learner.category_probabilities
    elif category_learning == "hebbian":
        learner = SupervisedHebbianPrior(CARDINALITIES, len(CLASS_NAMES),
                                          smoothing, hebbian_learning_rate)
        learner.partial_fit(samples, training_labels)
        class_lp = learner.class_log_probabilities(states)
        weights = learner.category_probabilities
    elif category_learning == "counts":
        class_lp = []
        for c in range(len(CLASS_NAMES)):
            subset = samples[training_labels == c]
            lp = np.zeros(len(states))
            for j, cardinality in enumerate(CARDINALITIES):
                counts = np.bincount(subset[:, j], minlength=cardinality) + smoothing
                lp += np.log(counts / counts.sum())[states[:, j]]
            class_lp.append(normalize_logp(lp))
        class_lp = np.stack(class_lp)
        weights = np.bincount(training_labels, minlength=len(CLASS_NAMES)) + smoothing
        weights = weights / weights.sum()
    else:
        raise ValueError("category_learning must be 'hebbian', 'counts' or 'competitive'.")
    category = logsumexp(class_lp + np.log(weights[:, None]), axis=0)
    exemplar_model = None
    if exemplar_learning == "kernel":
        distances = np.sum(states[:, None, :] != samples[None, :, :], axis=-1)
        exemplar = logsumexp(-exemplar_precision * distances, axis=1) - np.log(len(samples))
    elif exemplar_learning == "hopfield":
        exemplar_model = HopfieldExemplarPrior(CARDINALITIES, hopfield_inverse_temperature)
        exemplar_model.fit(samples)
        exemplar = exemplar_model.log_prior(states)
    else:
        raise ValueError("exemplar_learning must be 'kernel' or 'hopfield'.")
    priors = {
        "uniform": normalize_logp(np.zeros(len(states))),
        "generic": normalize_logp(generic),
        "category": normalize_logp(category),
        "exemplar": normalize_logp(exemplar),
    }
    return PriorBank(priors, class_lp, weights, training_ids.copy(), exemplar_model)


def block_mask(size, visible_fraction, rng, block=5):
    if not 0 <= visible_fraction <= 1:
        raise ValueError("visible_fraction must be in [0, 1]")
    n = int(np.ceil(size / block))
    blocks = rng.random((n, n)) < visible_fraction
    return np.repeat(np.repeat(blocks, block, axis=0), block, axis=1)[:size, :size]


def observe(clean, mask, flip_probability=.05, rng=None):
    """Only visible pixels can be flipped. Unknown pixels are explicitly zeroed."""
    if rng is None:
        rng = np.random.default_rng(0)
    noisy = np.logical_xor(clean > .5, rng.random(clean.shape) < flip_probability)
    return np.where(mask, noisy, 0).astype(np.float32)


def sample_trials(library, n_per_visibility=9, visibilities=(.15, .35, .65), seed=101, flip_probability=.05):
    rng = np.random.default_rng(seed)
    trials = []
    for visibility in visibilities:
        for i in range(n_per_visibility):
            category = i % len(CLASS_NAMES)
            p = np.exp(normalize_logp(library.teacher_logp[category, library.test_ids]))
            index = int(rng.choice(library.test_ids, p=p))
            truth = library.images[index]
            mask = block_mask(truth.shape[0], visibility, rng)
            y = observe(truth, mask, flip_probability, rng)
            trials.append(dict(index=index, category=category, visibility=visibility,
                               truth=truth, mask=mask, observed=y))
    return trials


def prediction(q, images):
    return (np.asarray(q) @ images.reshape(len(images), -1)).reshape(images.shape[1:])


def completion_metrics(probability, truth, observed, mask, threshold=.5):
    """Truth is evaluation-only. Missing contours and invented ones are separate."""
    probability = np.asarray(probability)
    truth = np.asarray(truth) > .5
    mask = np.asarray(mask, dtype=bool)
    hidden = ~mask
    drawn = probability >= threshold

    def mean_or_nan(values):
        return float(np.mean(values)) if len(values) else float("nan")

    return {
        "hidden_brier": mean_or_nan((probability[hidden]-truth[hidden])**2),
        "hidden_recall": mean_or_nan(drawn[hidden & truth]),
        "hidden_false_positive": mean_or_nan(drawn[hidden & ~truth]),
        "visible_error": mean_or_nan((probability[mask]-observed[mask])**2),
    }


def paired_summary(rows, metric, **filters):
    values = [r[metric] for r in rows if all(r[k] == v for k, v in filters.items())]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    sem = np.std(values, ddof=1)/np.sqrt(len(values)) if len(values) > 1 else 0.
    return float(np.mean(values)), float(sem)


def show_observation(ax, observed, mask, title="Observed fragments"):
    rgb = np.repeat((1-observed)[..., None], 3, axis=-1)
    rgb[~mask] = (.72, .80, .88)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_completion(ax, probability, title):
    ax.imshow(probability, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def error_overlay(probability, truth, mask):
    """Black=correct strokes, red=invented hidden strokes, blue=missed strokes."""
    drawn, real = probability >= .5, truth > .5
    rgb = np.ones((*truth.shape, 3))
    rgb[drawn] = (.15, .15, .15)
    rgb[(~mask) & drawn & ~real] = (.88, .16, .13)
    rgb[(~mask) & ~drawn & real] = (.10, .35, .85)
    return rgb
