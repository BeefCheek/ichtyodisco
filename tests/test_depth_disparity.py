import numpy as np

from aquarium.depth.hf_estimator import disparity_to_depth
from aquarium.depth.sampling import TankDepthCalibration


def test_disparity_conversion_is_affine_invariant_and_ordered():
    rng = np.random.default_rng(0)
    disp = rng.uniform(0.3, 4.5, size=(60, 80)).astype(np.float32)  # larger = closer
    z = []
    for scale, shift in [(1.0, 0.0), (3.7, -0.25), (0.2, 5.0)]:  # unknown model scale/shift
        cal = TankDepthCalibration()
        z.append(cal.normalise(disparity_to_depth(scale * disp + shift)))
    np.testing.assert_allclose(z[0], z[1], atol=1e-5)
    np.testing.assert_allclose(z[0], z[2], atol=1e-4)
    near, far = np.unravel_index(disp.argmax(), disp.shape), np.unravel_index(disp.argmin(), disp.shape)
    assert z[0][near] == 0.0 and z[0][far] == 1.0  # closest pixel at the glass, farthest at the back


def test_mid_tank_fish_is_not_crushed_to_the_glass():
    # back wall ~ disparity 0.3, front glass ~ 4.3, fish halfway through the water column
    disp = np.linspace(0.3, 4.3, 101, dtype=np.float32)[None, :].repeat(10, 0)
    cal = TankDepthCalibration(near_percentile=0, far_percentile=100)
    z = cal.normalise(disparity_to_depth(disp))
    assert abs(float(z[0, 50]) - 0.5) < 0.02
