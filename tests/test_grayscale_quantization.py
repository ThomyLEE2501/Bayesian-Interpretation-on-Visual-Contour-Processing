"""Checks for information loss and illumination in the joined-region experiment."""
import unittest

import numpy as np
from scipy import ndimage

from computationalExperiments.recovery_from_grayscale_quantization.grayscale_regions import (
    Bridge, Scene, boundary_survival, circle, closed_curve, ellipse, encode_regions,
    quantize, render, sample_scene, signal_image,
)


class GrayscaleGroupingTests(unittest.TestCase):
    def test_manual_lobes_and_selected_connections(self):
        centers = [(.18, .22), (.47, .22), (.77, .22), (.70, .75)]
        oval = closed_curve(lambda t: (.06*np.cos(t), .08*np.sin(t)))
        shapes = [circle(.07), ellipse(.08, .045), oval, circle(.07)]
        bridges = [Bridge(0, 1, width=.03),
                   Bridge(1, 2, width=.03, bend=.085, profile=lambda t: np.sin(np.pi*t))]
        size = 160
        bare, linked, _ = render(Scene(centers, shapes, bridges), size, samples=2)
        self.assertEqual(ndimage.label(bare >= .5)[1], 4)
        self.assertEqual(ndimage.label(linked >= .5)[1], 2)
        for x, y in centers:
            self.assertEqual(bare[int(y*size), int(x*size)], 1)
        for x, y in [(.325, .22), (.62, .305)]:
            self.assertEqual(bare[int(y*size), int(x*size)], 0)
            self.assertEqual(linked[int(y*size), int(x*size)], 1)
        self.assertEqual(linked[int(.22*size), int(.62*size)], 0)
        # The third lobe's local outline is taller than it is wide.
        self.assertEqual(bare[int(.29*size), int(.77*size)], 1)
        self.assertEqual(bare[int(.22*size), int(.84*size)], 0)

    def test_default_connections_and_single_isolated_lobe(self):
        lobes, signal, _ = render(Scene(), 96)
        self.assertEqual(ndimage.label(lobes >= .5)[1], 2)
        self.assertEqual(ndimage.label(signal >= .5)[1], 1)
        lobes, signal, _ = render(Scene(centers=[(.25, .75)], bridges=[]), 96)
        np.testing.assert_array_equal(lobes, signal)
        self.assertEqual(lobes[72, 24], 1)

    def test_invalid_manual_connections(self):
        for bridges in ([Bridge(0, 1), Bridge(1, 0)], [Bridge(0, 2)], [Bridge(0, 0)],
                        [Bridge(0, 1, profile=lambda t: t)]):
            with self.subTest(bridges=bridges), self.assertRaises(ValueError):
                Scene(bridges=bridges)
        with self.assertRaises(ValueError):
            Scene(shapes=[circle()])

    def test_signed_brightness_in_rendering_boundary_counts_and_encoding(self):
        lobes, signal, fields = render(Scene(), 96, samples=1)
        field = fields["diagonal"]
        for offset in (16., -16., 0.):
            with self.subTest(offset=offset):
                image = signal_image(field, signal, offset)
                np.testing.assert_allclose(image[signal == 1], field[signal == 1] + offset)
                np.testing.assert_array_equal(image[signal == 0], field[signal == 0])
                bins, background = quantize(image, 8), quantize(field, 8)
                total, visible, _ = boundary_survival(signal, bins, background, signal > lobes, offset)
                self.assertTrue(np.all(total > 0))
                np.testing.assert_array_equal(visible, total if offset else np.zeros_like(total))
        field = fields["uniform"]
        bright = quantize(signal_image(field, signal, 16.), 8)
        dark = quantize(signal_image(field, signal, -16.), 8)
        self.assertTrue(encode_regions(bright).any())
        np.testing.assert_array_equal(encode_regions(bright), encode_regions(dark, dark_signal=True))

    def test_bin_placement_controls_visibility_at_fixed_contrast(self):
        levels = np.array([102., 109.])
        np.testing.assert_array_equal(quantize(levels, 16, 0), [6, 6])
        np.testing.assert_array_equal(quantize(levels, 16, 12), [5, 6])

    def test_partial_and_erased_bridge_on_the_same_scene(self):
        separate, joined, lights = render(sample_scene(48), 128)
        field = lights["diagonal"]
        self.assertGreater(field[0, 0], field[-1, -1])
        def observe(phase):
            return [quantize(field + 7*coverage, 16, phase) for coverage in (separate, joined)]
        a, b = observe(4)
        total, surviving, _ = boundary_survival(joined, b, quantize(field, 16, 4), joined > separate)
        self.assertTrue(np.any(a != b))
        self.assertTrue(0 < surviving[1] < total[1])
        a, b = observe(12)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(encode_regions(a), encode_regions(b))
        self.assertGreater(np.ptp(b), 0)  # The image still contains visible structure.

    def test_illumination_alone_does_not_earn_contour_credit(self):
        separate, joined, fields = render(sample_scene(8), 64)
        background = quantize(fields["local"], 8)
        total, surviving, _ = boundary_survival(joined, background, background, joined > separate)
        self.assertTrue(np.all(total > 0))
        self.assertFalse(surviving.any())
