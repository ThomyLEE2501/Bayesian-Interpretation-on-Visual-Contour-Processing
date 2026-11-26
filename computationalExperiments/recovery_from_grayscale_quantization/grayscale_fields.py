"""Replacement and overlay brightness fields on a common image grid."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gradient:
    """Offsets from a baseline. Angle is the direction of equal-brightness lines."""
    kind: str = "uniform"
    angle: float = 0.0       # radians clockwise from horizontal
    span: float = 0.0        # signed corner-to-corner ramp range
    width: float = 0.1       # distance from one trough to the next peak
    phase: float = 0.0       # cycles at the image center
    amplitude: float = 12.0  # signed periodic amplitude in gray levels
    spots: tuple = ((0.5, 0.5, 0.15, 12.0),)  # x, y, sigma, signed peak offset

    def __post_init__(self):
        if self.kind not in ("uniform", "linear", "periodic", "local"):
            raise ValueError("Choose uniform, linear, periodic, or local.")
        if not np.isfinite((self.angle, self.span, self.width, self.phase, self.amplitude)).all():
            raise ValueError("Gradient parameters must be finite.")
        if self.kind == "periodic" and self.width <= 0:
            raise ValueError("Periodic band width must be positive.")
        if self.kind == "local":
            if not 1 <= len(self.spots) <= 5:
                raise ValueError("A local field needs one to five spots.")
            for cx, cy, sigma, peak in self.spots:
                if (not np.isfinite((cx, cy, sigma, peak)).all() or sigma <= 0
                        or not (0 <= cx <= 1 and 0 <= cy <= 1)):
                    raise ValueError("Use a normalized spot center, positive sigma and finite peak.")

    def values(self, x, y, baseline):
        values = np.full(x.shape, baseline, dtype=float)
        if self.kind in ("linear", "periodic"):
            nx, ny = -np.sin(self.angle), np.cos(self.angle)
            distance = nx*(x-.5) + ny*(y-.5)
            values += self.span*distance/(abs(nx)+abs(ny))
            if self.kind == "periodic":
                values -= self.amplitude*np.cos(np.pi*distance/self.width + 2*np.pi*self.phase)
        elif self.kind == "local":
            for cx, cy, sigma, peak in self.spots:
                values += peak*np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))
        return np.clip(values, 0., 256.)


@dataclass(frozen=True)
class GrayLayers:
    L_b: float = 100.
    L_s: float = 7.
    background: Gradient = Gradient()
    signal: Gradient = Gradient()

    def __post_init__(self):
        if not np.isfinite((self.L_b, self.L_s)).all() or not 0 <= self.L_b+self.L_s <= 256:
            raise ValueError("Use finite baselines with 0 <= L_b + L_s <= 256.")

    def images(self, size, *, mode="replace"):
        """Return the background and the brightness at full signal coverage."""
        if mode not in ("replace", "overlay"):
            raise ValueError("mode must be 'replace' or 'overlay'.")
        axis = (np.arange(size) + .5)/size
        x, y = np.meshgrid(axis, axis)
        signal_gradient = self.background if mode == "overlay" else self.signal
        return (self.background.values(x, y, self.L_b),
                signal_gradient.values(x, y, self.L_b+self.L_s))


def sample_layers(seed):
    # Keep brightness draws independent of sample_scene's geometry stream.
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))

    def draw_gradient():
        kind = rng.choice(("uniform", "linear", "periodic", "local"))
        if kind == "local":
            spots = tuple((rng.uniform(.1,.9), rng.uniform(.1,.9), rng.uniform(.08,.25),
                           rng.choice((-1,1))*rng.uniform(6,18)) for _ in range(rng.integers(1,6)))
            return Gradient(kind, spots=spots)
        return Gradient(kind, angle=rng.uniform(0,np.pi), span=rng.uniform(-24,24),
                        width=rng.uniform(.07,.25), phase=rng.random(), amplitude=rng.uniform(6,18))

    return GrayLayers(rng.uniform(80,176), rng.choice((-1,1))*rng.uniform(4,18),
                      draw_gradient(), draw_gradient())


def composite(background, signal, coverage):
    """Blend the background and full-coverage brightness at the signal boundary."""
    return (1.-coverage)*background + coverage*signal
