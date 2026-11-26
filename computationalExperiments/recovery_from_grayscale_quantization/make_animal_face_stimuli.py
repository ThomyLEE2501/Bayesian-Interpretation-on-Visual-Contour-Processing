"""Composite and quantize a transparent portrait on the experiment's grayscale backgrounds.

From the project root run python -m computationalExperiments.recovery_from_grayscale_quantization.make_animal_face_stimuli.
By default, reproduce the pelican demo from its bundled source. To generate all
384 quantized variants of another portrait, pass SOURCE.png --sweep --output DIR.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .grayscale_fields import Gradient, composite
    from .grayscale_regions import quantize
else:
    from grayscale_fields import Gradient, composite
    from grayscale_regions import quantize


DEMO = Path(__file__).resolve().parents[2] / "outputs" / "grayscale_animal_faces" / "pelican_demo"
PATTERNS = (
    Gradient(), Gradient("linear", angle=np.pi/4, span=64),
    Gradient("periodic", angle=np.pi/3, span=16, width=.11, phase=.25, amplitude=30),
    Gradient("local", spots=((.28,.34,.18,40), (.72,.62,.22,-36))),
)


def prepare_face(source):
    with Image.open(source) as original:
        portrait = original.convert("RGBA")
    if portrait.getchannel("A").getextrema()[0] == 255:
        raise ValueError("The portrait must have a transparent background.")
    portrait.thumbnail((112, 112), Image.Resampling.LANCZOS)
    face = Image.new("LA", (128, 128))
    face.paste(portrait.convert("LA"), ((128-portrait.width)//2, (128-portrait.height)//2))
    return face


def display_bins(image, width, phase):
    bins = quantize(image, width, phase).astype(int)
    lower = (bins - int(phase > 0))*width + phase
    upper = lower + width - 1
    return ((lower.clip(0,255) + upper.clip(0,255) + 1)//2).astype("uint8")


def demo_sheet(output, panels):
    sheet = Image.new("RGB", (4*196, 222), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (title, panel) in enumerate(panels):
        panel = panel.resize((188,188), Image.Resampling.NEAREST)
        corner = (196*i+4, 4)
        alpha = panel.getchannel("A") if "A" in panel.getbands() else None
        sheet.paste(panel.convert("RGB"), corner, alpha)
        draw.text((corner[0], 199), title, font=ImageFont.load_default(), fill="black")
    sheet.save(output/"how_it_was_made.png")


def generate(source, output, sweep=False):
    output.mkdir(parents=True, exist_ok=True)
    face = prepare_face(source)
    face.save(output/"face_128.png")
    pixels = np.asarray(face)
    intensity, alpha = pixels[:,:,0], pixels[:,:,1]/255.
    axis = (np.arange(128)+.5)/128
    x, y = np.meshgrid(axis, axis)
    if not sweep:
        field = PATTERNS[2].values(x, y, 160)
        background = Image.fromarray(np.rint(field).astype("uint8"))
        pre = np.rint(composite(field, intensity, alpha)).astype("uint8")
        original = Image.fromarray(pre)
        final = Image.fromarray(display_bins(pre, 128, 0))
        background.save(output/"background.png")
        original.save(output/"composite.png")
        final.save(output/"quantized.png")
        demo_sheet(output, [("Face cutout", face), ("Periodic background", background),
                            ("Before quantization", original), ("Two gray levels", final)])
        return

    pre_dir = output/"preQuantized"
    pre_dir.mkdir(exist_ok=True)
    for pattern in PATTERNS:
        for baseline in (48,96,160):
            field = pattern.values(x, y, baseline)
            scenes = {"overlay": field + alpha*intensity,
                      "replace": composite(field, intensity, alpha)}
            for mode, scene in scenes.items():
                pre = np.rint(scene).clip(0,255).astype("uint8")
                name = f"{source.stem}_{pattern.kind}_b{baseline}_{mode}"
                Image.fromarray(pre).save(pre_dir/(name+".png"))
                for width in (16,32,64,128):
                    dest = output/"Quantized"/f"w{width}"
                    dest.mkdir(parents=True, exist_ok=True)
                    for phase in range(0,width,width//4):
                        final = Image.fromarray(display_bins(pre, width, phase))
                        final.save(dest/f"{name}_w{width}_p{phase}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEMO/"source_pelican.png")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    output = args.output or (Path.cwd()/(args.source.stem+"_stimuli")
                             if args.sweep or args.source != DEMO/"source_pelican.png" else DEMO)
    generate(args.source, output, args.sweep)
