"""Paired continuous regions under illumination and grayscale quantization."""
from cmath import phase
from dataclasses import dataclass
from collections.abc import Callable
from numbers import Integral

import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy import ndimage


def ellipse(rx, ry, angle=0.):
    """Filled ellipse with local semi-axes rx, ry and rotation in radians."""
    if not np.all(np.isfinite((rx, ry, angle))) or min(rx, ry) <= 0:
        raise ValueError("Ellipse radii must be positive and all parameters finite.")
    c, s = np.cos(angle), np.sin(angle)

    def inside(x, y):
        u, v = c*x + s*y, -s*x + c*y
        return (u/rx)**2 + (v/ry)**2 <= 1
    return inside


def circle(radius=.12):
    """A lobe centered at its own origin, with radius in field-of-view units."""
    return ellipse(radius, radius)


def closed_curve(curve, vertices=256):
    """Fill a simple closed curve supplied as theta -> (local x, local y).

    Sample one turn in radians. The final vertex connects back to the first.
    The result can be used wherever circle() or ellipse() is accepted.
    """
    if not isinstance(vertices, Integral) or vertices < 3:
        raise ValueError("A closed curve needs at least three vertices.")
    theta = np.linspace(0., 2*np.pi, vertices, endpoint=False)
    points = np.column_stack(curve(theta))
    if points.shape != (vertices, 2) or not np.isfinite(points).all():
        raise ValueError("Return one finite (x, y) point for each angle.")
    polygon = PolygonPath(points)
    lo, hi = points.min(axis=0), points.max(axis=0)

    def inside(x, y):
        box = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
        mask = np.zeros_like(box)
        mask[box] = polygon.contains_points(np.column_stack((x[box], y[box])))
        return mask
    return inside


def parabolic_bend(t):
    """Unit-height arc between t=0 and t=1. A zero bend gives a straight bridge."""
    return 4*t*(1-t)


@dataclass(frozen=True)
class Bridge:
    start: int
    end: int
    width: float = .035
    bend: float = 0.
    profile: Callable = parabolic_bend

    def coverage(self, x, y, centers):
        # The bridge's frame starts at its first lobe and points at the second.
        x0, y0 = centers[self.start]
        dx, dy = np.subtract(centers[self.end], (x0, y0))
        length = np.hypot(dx, dy)
        u = ((x-x0)*dx + (y-y0)*dy)/length
        v = (-(x-x0)*dy + (y-y0)*dx)/length
        t = np.clip(u/length, 0., 1.)
        offset = self.bend*self.profile(t)
        return (u >= 0) & (u <= length) & (abs(v-offset) <= self.width/2)


@dataclass(frozen=True)
class Scene:
    centers: tuple = ((.30, .50), (.70, .50))
    shapes: tuple | None = None
    bridges: tuple = (Bridge(0, 1),)
    signal_offset: float = 7.
    light_angle: float = 0.
    spots: tuple = ((.68, .57, .16), (.32, .43, .12))

    def __post_init__(self):
        coords = np.asarray(self.centers, dtype=float)
        if (coords.ndim != 2 or coords.shape[1] != 2 or not len(coords)
                or not np.isfinite(coords).all() or np.any((coords < 0) | (coords > 1))):
            raise ValueError("centers must contain (x, y) pairs in [0, 1].")
        centers = tuple(map(tuple, coords))
        shapes = (circle(),)*len(centers) if self.shapes is None else tuple(self.shapes)
        bridges = tuple(self.bridges)
        if len(shapes) != len(centers) or not all(callable(shape) for shape in shapes):
            raise ValueError("Supply one lobe shape per center, or omit shapes for circles.")
        if len(bridges) >= len(centers):
            raise ValueError("Use fewer bridges than lobes.")
        for bridge in bridges:
            if not isinstance(bridge, Bridge):
                raise ValueError("Each connection must be a Bridge.")
            if not all(isinstance(i, Integral) and 0 <= i < len(centers)
                       for i in (bridge.start, bridge.end)):
                raise ValueError("Bridge endpoints must be valid zero-based lobe indices.")
            if centers[bridge.start] == centers[bridge.end]:
                raise ValueError("A bridge must join two distinct centers.")
            if not np.isfinite((bridge.width, bridge.bend)).all() or bridge.width <= 0:
                raise ValueError("Bridge width must be positive and bend must be finite.")
            if not np.allclose(bridge.profile(np.array([0., 1.])), 0.):
                raise ValueError("A bend profile must return zero at both endpoints.")
        if not np.isfinite(self.signal_offset):
            raise ValueError("signal_offset must be finite.")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "bridges", bridges)


def sample_scene(seed, signal_offset=7.):
    rng = np.random.default_rng(seed)
    midpoint = rng.uniform(.43, .57, 2)
    angle = rng.uniform(0, np.pi)
    shapes = tuple(circle(r) for r in rng.uniform(.103, .144, 2))
    shift = rng.uniform(.18, .205)*np.array([np.cos(angle), np.sin(angle)])
    bridge = Bridge(0, 1, rng.uniform(.027, .043), rng.uniform(-.045, .045))
    light_angle = rng.uniform(0, 2*np.pi)
    spots = tuple(tuple(rng.uniform((.12, .12, .12), (.88, .88, .33))) for _ in range(2))
    return Scene(centers=(midpoint-shift, midpoint+shift), shapes=shapes, bridges=(bridge,),
                 signal_offset=signal_offset, light_angle=light_angle, spots=spots)


def _sample_grid(size, samples):
    axis = (np.arange(size*samples, dtype=np.float32) + .5) / (size*samples)
    return np.meshgrid(axis, axis)


def _pool(a, size, samples):
    return a.reshape(size, samples, size, samples).mean(axis=(1, 3), dtype=np.float32)


def signal_coverage(scene, size, samples=4):
    x, y = _sample_grid(size, samples)
    lobes = np.zeros(x.shape, dtype=bool)
    for (cx, cy), shape in zip(scene.centers, scene.shapes):
        lobes |= shape(x-cx, y-cy)
    signal = lobes.copy()
    for bridge in scene.bridges:
        signal |= bridge.coverage(x, y, scene.centers)
    return _pool(lobes, size, samples), _pool(signal, size, samples)


def render(scene, size, samples=4, gradient_span=10., spot_amplitude=11.):
    """Render the shared illumination fields used by the original benchmark."""
    lobes, signal = signal_coverage(scene, size, samples)
    x, y = _sample_grid(size, samples)
    def pool(a):
        return _pool(a, size, samples)
    angle = scene.light_angle
    lx, ly = np.cos(angle), np.sin(angle)
    rotated = ((x-.5)*lx + (y-.5)*ly) / (abs(lx)+abs(ly))
    local = np.full_like(x, 100.)
    for sign, (cx, cy, sigma) in zip((1, -1), scene.spots):
        local += sign*spot_amplitude*np.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma**2))
    fields = dict(uniform=np.full((size, size), 100., dtype=np.float32),
                  diagonal=pool(100 + gradient_span*(1-x-y)/2),
                  rotated=pool(100+gradient_span*rotated),
                  local=pool(local))
    return lobes, signal, fields


def signal_image(background, coverage, signal_offset):
    """Apply the signed gray-level offset to the local illumination at each pixel."""
    return background + signal_offset*coverage


def quantize(image, width, phase=0):
    """Return bin indices. Interior bins have width gray levels and offset phase."""
    width = np.floor(width)
    if not 0 <= phase < width:
        raise ValueError("Width must be positive integer; phase must be in [0, width)")
    if width < 1 or 256 % width != 0:
        if phase != 0 and phase != 256 % width:
            raise ValueError("phase must be 0 or 256 % width when width does not divide 256")
    values = np.clip(np.asarray(image, dtype=np.float64), 0, 255)
    indices = np.floor((values - phase) / width)
    return (indices + int(phase > 0)).astype(np.uint8)


def boundary_survival(coverage, image, background, diagnostic, signal_offset=7.):
    """Count adjacent pixel pairs crossing the true half-coverage outline.

    An edge survives when its bin jump in the signal's contrast direction
    exceeds the jump already present in the illumination alone.
    signal_offset can be a scalar or the full signal-minus-background field.
    """
    total = np.zeros(2, dtype=int)
    visible = np.zeros(2, dtype=int)
    points = []
    for axis in (0, 1):
        inside = coverage >= .5
        step = np.diff(inside.astype(np.int8), axis=axis)
        polarity = np.sign(signal_offset)
        if np.ndim(polarity):
            # A crossing takes the contrast sign on the interior side.
            before, after = ((polarity[:-1], polarity[1:]) if axis == 0 else
                             (polarity[:, :-1], polarity[:, 1:]))
            polarity = np.where(step > 0, after, before)
        step = step * polarity
        jump = np.diff(image.astype(np.int16), axis=axis)*step
        light_jump = np.diff(background.astype(np.int16), axis=axis)*step
        survived = jump > np.maximum(0, light_jump)
        a, b = ((diagnostic[:-1], diagnostic[1:]) if axis == 0 else
                (diagnostic[:, :-1], diagnostic[:, 1:]))
        is_bridge = a | b
        edge = np.diff(inside, axis=axis)
        for group, region in enumerate((~is_bridge, is_bridge)):
            total[group] += np.count_nonzero(edge & region)
            visible[group] += np.count_nonzero(edge & region & survived)
        yy, xx = np.nonzero(edge)
        points.append((xx + .5*(axis == 1), yy + .5*(axis == 0), survived[yy, xx]))
    return total, visible, points


FEATURES = ("largest area", "second area", "area balance", "compactness",
            "horizontal span", "vertical span", "split-core balance", "core separation")


def encode_regions(image, *, dark_signal=False):
    """Describe candidate regions using the experiment's fixed contrast polarity.

    Isolated components and thin-necked components are competing geometric clues.
    Distances and areas use field-of-view units across resolutions.
    """
    if dark_signal:
        image = -image.astype(float)
    n = image.shape[0]
    descriptor = np.zeros(len(FEATURES))
    for level in np.unique(image)[1:]:
        regions, _ = ndimage.label(image >= level)
        areas = []
        for i, box in enumerate(ndimage.find_objects(regions), start=1):
            if box is None or any(sl.start == 0 or sl.stop == n for sl in box):
                continue
            component = regions[box] == i
            area = component.sum()/n**2
            if not .004 <= area <= .45:
                continue
            areas.append(area)
            # A physical erosion radius disconnects lobes joined by a thin neck.
            dist = ndimage.distance_transform_edt(np.pad(component, 1))
            cores, k = ndimage.label(dist > .025*n)
            balance, separation = 0., 0.
            if k >= 2:
                counts = np.bincount(cores.ravel())[1:]
                first, second = np.argsort(counts)[-2:][::-1]
                centers = ndimage.center_of_mass(cores > 0, cores, [first+1, second+1])
                balance = counts[second]/counts[first]
                separation = np.linalg.norm(np.subtract(*centers))/n
            h, w = component.shape
            descriptor[3:] = np.maximum(descriptor[3:], (
                component.mean(), w/n, h/n, balance, separation))
        if areas:
            first, second = sorted(areas + [0.], reverse=True)[:2]
            descriptor[:3] = np.maximum(descriptor[:3], (first, second, second/first))
    return descriptor
