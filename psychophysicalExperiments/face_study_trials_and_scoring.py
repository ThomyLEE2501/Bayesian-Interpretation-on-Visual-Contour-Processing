"""Stimuli, slides, scoring and session files shared by people and the MAP model."""
import csv
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour_models.face_recipes_and_observations import (CLASS_NAMES, FACTOR_NAMES, FACTOR_VALUES, block_mask,
                           make_library, observe, render_state)

VISIBILITY = (.10, .25, .40, .55, .70)
NOISE = (.02, .05, .10, .15, .20)
VISIBILITY_BOUNDARIES = (20, 35, 60)
VISIBILITY_LABELS = ("Under 20%", "20% to under 35%", "35% to under 60%", "60% and above")
DEFAULT_VISIBILITY_COUNTS = (5, 10, 20, 15)
SESSION_TRIALS = sum(DEFAULT_VISIBILITY_COUNTS)
BLOCKS_PER_SIDE = 8
BLOCK_COUNT = BLOCKS_PER_SIDE ** 2
KEYS = [s.replace(" ", "_") for s in FACTOR_NAMES]
# Display order is independent of the stored six-factor recipe and its numeric ID.
DISPLAY_ORDER = tuple(FACTOR_NAMES.index(name) for name in
                      ("head shape", "ear shape", "eye shape", "eye spacing", "nose shape", "layout"))
DISPLAY = {"oval pupils": "Oval + vertical pupil"}
POSTER = {"ink": "#293D48", "teal": "#006B67", "orange": "#B86632"}
RESIZE = Image.Resampling.NEAREST
DEFAULT_SESSIONS = Path(__file__).resolve().parents[1] / "outputs" / "psychophysics" / "participant_sessions"


def font(size, bold=False):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def display_name(value):
    return DISPLAY.get(value, value.capitalize())


def pixels(ink, mask=None):
    rgb = np.repeat((255 * (1 - ink)).astype(np.uint8)[..., None], 3, axis=-1)
    if mask is not None:
        rgb[~mask] = (184, 204, 224)
    return Image.fromarray(rgb)


def option_icon(j, k, max_size=(128, 60)):
    state = [0, 0, 1, 1, 0, 0]
    state[j] = k
    layers = (("outline",), ("ears",), ("eyes",), ("nose",),
              ("eyes", "nose", "mouth"), ("eyes",))[j]
    # Shared crop per factor preserves spacing, aspect ratio and layout differences.
    bounds = ((2, 3, 38, 39), (0, 0, 40, 24), (8, 12, 32, 23),
              (12, 20, 28, 31), (5, 6, 35, 36), (9, 11, 31, 24))[j]
    ink = render_state(state, parts=layers)
    if j == 4:
        # A faint head locates the features so aligned and lowered are comparable.
        ink = np.maximum(ink, .55 * render_state(state, parts=("outline",)))
        bounds = (6, 5, 34, 39)
    icon = pixels(ink).crop(bounds)
    ratio = min(max_size[0] / icon.width, max_size[1] / icon.height)
    return icon.resize((round(icon.width * ratio), round(icon.height * ratio)), RESIZE)


def panel(images, labels, title):
    width = 1080
    sheet = Image.new("RGB", (width, 480), "white")
    d = ImageDraw.Draw(sheet)
    d.text((width//2, 40), title, anchor="mm", font=font(30, True), fill=POSTER["ink"])
    for i, (img, label) in enumerate(zip(images, labels)):
        x = 180 + 360*i
        sheet.paste(img.resize((280, 280), RESIZE), (x-140, 92))
        d.text((x, 410), label, anchor="mm", font=font(23), fill=POSTER["ink"])
    return sheet


def guides(folder, lib):
    sheet = Image.new("RGB", (1440, 980), "white")
    d = ImageDraw.Draw(sheet)
    d.text((44, 16), "Six features of a face", fill=POSTER["ink"], font=font(40, True))
    for row, j in enumerate(DISPLAY_ORDER):
        y = 82 + row*142
        d.text((44, y+40), FACTOR_NAMES[j].capitalize(), font=font(29, True), fill=POSTER["teal"])
        for k, value in enumerate(FACTOR_VALUES[j]):
            height = 130 if FACTOR_NAMES[j] == "layout" else 94
            icon = option_icon(j, k, (275, height))
            x = 460 + 365*k
            sheet.paste(icon, (x-icon.width//2, y+(height-icon.height)//2))
            d.text((x, y+height+18), display_name(value), anchor="mm", fill=POSTER["ink"], font=font(23))
    sheet.save(folder / "feature_guide.png")
    prototypes = [pixels(lib.images[index]) for index in lib.teacher_logp.argmax(axis=1)]
    examples = panel(prototypes, CLASS_NAMES, "Typical faces")
    ImageDraw.Draw(examples).text((540, 460), "Any combination of the six features is possible.",
                                 anchor="mm", font=font(20), fill=POSTER["teal"])
    examples.save(folder / "typical_faces.png")


HINT_PHRASES = (
    ("round face", "wide face", "tall face"),
    ("round ears", "pointed ears", "long ears"),
    ("narrow eye spacing", "medium eye spacing", "wide eye spacing"),
    ("triangular nose", "round nose", "muzzle nose"),
    ("aligned features", "lowered features", "scrambled features"),
    ("dotted eyes", "arched eyes", "oval eyes with vertical pupils"),
)


def visibility_band(percent):
    return sum(percent >= boundary for boundary in VISIBILITY_BOUNDARIES)


def validate_visibility_counts(counts):
    counts = tuple(counts)
    if (len(counts) != len(VISIBILITY_LABELS)
            or any(not isinstance(n, (int, np.integer)) or n < 0 for n in counts)
            or sum(counts) != SESSION_TRIALS):
        raise ValueError(f"Enter four nonnegative visibility counts that total {SESSION_TRIALS}.")
    return counts


def balanced_draw(values, count, rng):
    cycles, extra = divmod(count, len(values))
    draws = np.concatenate((np.tile(values, cycles), rng.choice(values, extra, replace=False)))
    return rng.permutation(draws)


def make_plan(lib, seed, cued=True, hint_range=(0, 6), hint_accuracy=None, show_counts=True,
              visibility_counts=DEFAULT_VISIBILITY_COUNTS, noise_cap=max(NOISE)):
    low, high = hint_range
    if not 0 <= low <= high <= 6:
        raise ValueError("Hint counts must satisfy 0 <= minimum <= maximum <= 6.")
    if hint_accuracy is not None and not 0 <= hint_accuracy <= 1:
        raise ValueError("Hint accuracy must be between 0 and 1.")
    visibility_counts = validate_visibility_counts(visibility_counts)
    if noise_cap not in NOISE:
        raise ValueError("Maximum noise must be 2%, 5%, 10%, 15% or 20%.")
    noises = [noise for noise in NOISE if noise <= noise_cap]
    # A change to the hint treatment keeps the faces, masks and trial order paired.
    stimulus_seed, total_seed, accuracy_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(stimulus_seed)
    total_rng = np.random.default_rng(total_seed)
    accuracy_rng = np.random.default_rng(accuracy_seed)
    ids = rng.choice(lib.test_ids, SESSION_TRIALS, replace=False)
    conditions = []
    totals = np.arange(low, high+1) if cued else np.array([0])
    for band, count in enumerate(visibility_counts):
        # Sample a block count in the band, then choose the block locations at replay.
        blocks = [n for n in range(1, BLOCK_COUNT+1)
                  if visibility_band(100*n/BLOCK_COUNT) == band]
        for visible, noise, total in zip(balanced_draw(blocks, count, rng),
                                         balanced_draw(noises, count, rng),
                                         balanced_draw(totals, count, total_rng)):
            correct = (accuracy_rng.integers(total+1) if hint_accuracy is None
                       else accuracy_rng.binomial(total, hint_accuracy))
            conditions.append((int(visible), float(noise), int(total), int(correct)))
    rng.shuffle(conditions)
    return [dict(trial=i+1, state_id=int(idx), visibility=blocks/BLOCK_COUNT, noise=noise,
                 visible_block_count=blocks,
                 cues_given=total, correct_cues=k, hint_counts_shown=bool(cued and show_counts),
                 stimulus_seed=int(rng.integers(2**32)), cue_seed=int(rng.integers(2**32)))
            for i, (idx, (blocks, noise, total, k)) in enumerate(zip(ids, conditions))]


def observation(lib, trial):
    if "visible_blocks" in trial:
        # Browser sessions store the presented pixels because JavaScript uses a
        # different random generator. Bit order is row-major, MSB first.
        def unpack(key, side):
            packed = np.frombuffer(base64.b64decode(trial[key], validate=True), dtype=np.uint8)
            return np.unpackbits(packed).reshape(side, side)
        mask = unpack("visible_blocks", 8).astype(bool).repeat(5, axis=0).repeat(5, axis=1)
        return unpack("observed_bits", 40).astype(np.float32), mask
    truth = lib.images[trial["state_id"]]
    rng = np.random.default_rng(trial["stimulus_seed"])
    if "visible_block_count" in trial:
        block_mask_flat = rng.permutation(BLOCK_COUNT) < trial["visible_block_count"]
        mask = block_mask_flat.reshape(BLOCKS_PER_SIDE, BLOCKS_PER_SIDE).repeat(5, axis=0).repeat(5, axis=1)
    else:
        mask = block_mask(40, trial["visibility"], rng)
    observed = observe(truth, mask, trial["noise"], rng)
    return observed, mask


def cue_values(state, seed, correct_count, total=6):
    """Return feature values, with -1 for omissions. Old uncued schedules used None."""
    if correct_count is None:
        return None
    if not 0 <= correct_count <= total <= 6:
        raise ValueError("The number of correct hints must be between zero and the total supplied.")
    rng = np.random.default_rng(seed)
    wrong = (state + rng.integers(1, 3, size=6)) % 3
    order = rng.permutation(6)
    right = order[:correct_count]
    wrong[right] = state[right]
    wrong[order[total:]] = -1
    return wrong


def trial_cues(state, trial):
    if "cue_values" in trial:
        return np.array(trial["cue_values"], dtype=int)
    return cue_values(state, trial["cue_seed"], trial["correct_cues"], trial.get("cues_given", 6))


def hint_text(cues, correct, show_counts):
    phrases = [] if cues is None else [HINT_PHRASES[j][cues[j]] for j in DISPLAY_ORDER if cues[j] >= 0]
    if not phrases:
        return "No hints this trial." + ("\nHint accuracy: 0 correct, 0 incorrect" if show_counts else "")
    hints = "Hints: " + ", ".join(phrases) + "."
    if show_counts:
        return hints + f"\nHint accuracy: {correct} correct, {len(phrases)-correct} incorrect"
    return hints


def score_row(lib, trial, choice, cues, observed, mask):
    state = lib.states[trial["state_id"]]
    hit = np.asarray(choice) == state
    given = 0 if cues is None else int(np.count_nonzero(cues >= 0))
    correct = 0 if cues is None else int(np.sum(cues == state))
    flips = int(np.count_nonzero(observed[mask] != lib.images[trial["state_id"]][mask]))
    row = dict(trial=trial["trial"], state_id=trial["state_id"], renderer_version=lib.renderer_version,
               stimulus_seed=trial["stimulus_seed"],
               cue_seed=trial["cue_seed"], visibility_pct=round(100*trial["visibility"], 8),
               actual_visibility_pct=100*mask.mean(), noise_pct=round(100*trial["noise"], 8),
               visible_pixels=int(mask.sum()), flipped_pixels=flips,
               flipped_pct=100*flips/mask.sum() if mask.any() else "",
               cues_given=given, correct_cues=correct, incorrect_cues=given-correct,
               hint_counts_shown=int(trial.get("hint_counts_shown", False)),
               score=int(hit.sum()))
    for j, key in enumerate(KEYS):
        row["true_"+key] = FACTOR_VALUES[j][state[j]]
        row["chosen_"+key] = FACTOR_VALUES[j][choice[j]]
        row["correct_"+key] = int(hit[j])
        row["cue_"+key] = "" if cues is None or cues[j] < 0 else FACTOR_VALUES[j][cues[j]]
    return row


def write_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


class Session:
    def __init__(self, root, name, number, seed, cued=True, hint_range=(0, 6),
                 hint_accuracy=None, show_counts=True, visibility_counts=DEFAULT_VISIBILITY_COUNTS,
                 noise_cap=max(NOISE)):
        name = name.strip()
        if not re.fullmatch(r"[\w -]{1,60}", name) or not name.strip("-_ "):
            raise ValueError("Use 1–60 letters, numbers, spaces, underscores or hyphens for the name.")
        if not number.isascii() or not number.isdecimal() or not 1 <= int(number) <= 999999:
            raise ValueError("Use a session number between 1 and 999999.")
        self.lib = make_library()
        self.plan = make_plan(self.lib, seed, cued, hint_range, hint_accuracy, show_counts,
                              visibility_counts, noise_cap)
        self.folder = Path(root) / f"{name}_session_{int(number):03d}"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.name, self.number, self.rows = name, int(number), []
        config = dict(subject=name, session=int(number), seed=seed, cued=cued, renderer_version=self.lib.renderer_version,
                      hint_range=hint_range if cued else (0, 0), hint_accuracy=hint_accuracy,
                      show_hint_counts=bool(cued and show_counts),
                      visibility_counts=tuple(visibility_counts), noise_cap=noise_cap,
                      protocol="desktop-v2", trials=self.plan)
        (self.folder/"session.json").write_text(json.dumps(config, indent=2), encoding="utf8")
        guides(self.folder, self.lib)

    def record(self, choice, seconds):
        if len(choice) != 6 or any(v not in (0, 1, 2) for v in choice):
            raise ValueError("Choose a value for every feature.")
        trial = self.plan[len(self.rows)]
        state = self.lib.states[trial["state_id"]]
        observed, mask = observation(self.lib, trial)
        cues = trial_cues(state, trial)
        row = score_row(self.lib, trial, choice, cues, observed, mask)
        filename = f"trial_{trial['trial']:02d}.png"
        row.update(source="participant", subject=self.name, session=self.number,
                   response_seconds=round(seconds, 3), prior_weight="", cue_weight="", feedback_png=filename)
        img = panel([pixels(observed, mask), pixels(render_state(choice)), pixels(self.lib.images[trial["state_id"]])],
                    ("What you saw", "Your choice", "Original face"),
                    f"Trial {trial['trial']}     |     {row['score']} of 6 features correct")
        img.save(self.folder/filename)
        write_rows(self.folder/"responses.csv", [*self.rows, row])
        self.rows.append(row)
        return filename
