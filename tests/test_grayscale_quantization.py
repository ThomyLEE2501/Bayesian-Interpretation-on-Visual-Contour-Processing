"""Checks for information loss and illumination in the joined-region experiment."""
import unittest

import numpy as np
from scipy import ndimage
from computationalExperiments.recovery_from_grayscale_quantization.grayscale_fields import (
    Gradient, GrayLayers, composite, sample_layers,
)

from computationalExperiments.recovery_from_grayscale_quantization.grayscale_regions import (
    Bridge, Scene, boundary_survival, circle, closed_curve, ellipse, encode_regions,
    quantize, render, sample_scene, signal_image,
)


class GrayscaleGroupingTests(unittest.TestCase):
    def test_layer_baselines_crop_and_antialiasing(self):
        for lb, ls, expected in ((100, -100, 0), (310, -54, 256), (-3, 11, 8)):
            bg, fg = GrayLayers(lb, ls).images(4)
            np.testing.assert_array_equal(bg, np.clip(lb, 0, 256))
            np.testing.assert_array_equal(fg, expected)
        for ls in (-101, 157):
            with self.assertRaises(ValueError):
                GrayLayers(100, ls)
        bg, fg = GrayLayers(90, 5, Gradient("linear", span=40), Gradient()).images(5)
        alpha = np.tile([0, .25, 1, .5, 0], (5,1))
        img = composite(bg, fg, alpha)
        np.testing.assert_array_equal(img[:,0], bg[:,0])
        np.testing.assert_array_equal(img[:,2], 95)
        np.testing.assert_allclose(img[:,1], .75*bg[:,1]+.25*95)

    def test_overlay_keeps_the_background_gradient_with_a_signed_offset(self):
        coverage = np.tile([0., .25, 1., .5, 0.], (5, 1))
        for offset in (-7., 0., 7.):
            with self.subTest(offset=offset):
                layers = GrayLayers(100, offset, Gradient("linear", span=20),
                                    Gradient("periodic", amplitude=14))
                bg, fg = layers.images(5, mode="overlay")
                np.testing.assert_allclose(fg, bg + offset)
                np.testing.assert_allclose(composite(bg, fg, coverage),
                                           bg + coverage*offset)
                np.testing.assert_array_equal(layers.images(5, mode="replace")[0], bg)
        with self.assertRaises(ValueError):
            layers.images(5, mode="unknown")

    def test_oriented_and_periodic_fields(self):
        for angle, axis in ((0,0), (np.pi/2,1)):
            _, ramp = GrayLayers(signal=Gradient("linear", angle=angle, span=30)).images(32)
            np.testing.assert_allclose(np.diff(ramp,axis=1-axis), 0, atol=1e-10)
            self.assertTrue(np.all(np.abs(np.diff(ramp,axis=axis)) > 0))
        def stripes(phase):
            return GrayLayers(signal=Gradient("periodic", span=30, width=.125, phase=phase)).images(32)[1]
        ramp = GrayLayers(signal=Gradient("linear", span=30)).images(32)[1]
        np.testing.assert_allclose((stripes(0)+stripes(.5))/2, ramp)
        wave = stripes(0)-ramp
        np.testing.assert_allclose(wave[8:], wave[:-8])

    def test_local_spots_signs_limits_and_random_repeatability(self):
        def levels(spots):
            return GrayLayers(signal=Gradient("local", spots=spots)).images(5)[1]
        bright = (.5,.5,.1,12)
        self.assertEqual(GrayLayers(signal=Gradient("local")).images(5)[1][2,2],119)
        self.assertEqual(levels([bright]*5)[2,2],167)
        self.assertEqual(levels([bright,(.5,.5,.1,-17)])[2,2],102)
        for spots in ([], [bright]*6):
            with self.assertRaises(ValueError):
                levels(spots)
        self.assertEqual(sample_layers(48), sample_layers(48))

    def test_outline_evidence_uses_local_contrast_sign(self):
        bg = np.full((6,6),100.)
        fg = bg.copy()
        fg[:,:3] -= 24
        fg[:,3:] += 24
        mask = np.zeros(bg.shape)
        mask[1:5,1:5] = 1
        total, alive, _ = boundary_survival(mask, quantize(composite(bg,fg,mask),8),
            quantize(bg,8), mask > 0, fg-bg)
        np.testing.assert_array_equal(total,alive)
        self.assertEqual(sum(alive),16)

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
        np.testing.assert_array_equal(quantize(levels, 16, 12), [6, 7])

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
