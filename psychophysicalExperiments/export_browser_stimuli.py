"""Refresh the static browser stimuli after changing the Python renderer."""
import base64
import io
import json
import tempfile
from pathlib import Path

import numpy as np

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contour_models.face_recipes_and_observations import CLASS_NAMES, FACTOR_NAMES, FACTOR_VALUES, RENDERER_VERSION, make_library
from psychophysicalExperiments.face_study_trials_and_scoring import DISPLAY_ORDER, HINT_PHRASES, NOISE, VISIBILITY, guides, option_icon


def png_uri(data):
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def main():
    lib = make_library()
    icons = []
    for j, values in enumerate(FACTOR_VALUES):
        row = []
        for k in range(len(values)):
            stream = io.BytesIO()
            option_icon(j, k, (264, 150)).save(stream, format="PNG")
            row.append(png_uri(stream.getvalue()))
        icons.append(row)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        guides(folder, lib)
        slides = {name: png_uri((folder/name).read_bytes()) for name in ("feature_guide.png", "typical_faces.png")}
    faces = np.packbits(lib.images.astype(bool).reshape(len(lib.states), -1), axis=1)
    assets = dict(version=RENDERER_VERSION, names=FACTOR_NAMES, values=FACTOR_VALUES,
                  order=DISPLAY_ORDER, hints=HINT_PHRASES, heldOut=lib.test_ids.tolist(),
                  visibility=VISIBILITY, noise=NOISE, icons=icons, slides=slides,
                  prototypes=lib.teacher_logp.argmax(axis=1).tolist(), categories=CLASS_NAMES,
                  faces=base64.b64encode(faces.tobytes()).decode("ascii"))
    target = Path(__file__).with_name("web") / "assets.js"
    target.write_text("const STIMULI = " + json.dumps(assets, separators=(",", ":")) + ";\n", encoding="utf8")


if __name__ == "__main__":
    main()
