from chiscore import liu_sf
from numpy.testing import assert_allclose


def test_liu_sf():

    w = [0] * 64 + [4985811.050871694]
    t = 906024.952053349
    assert_allclose(
        liu_sf(t, w, [1] * len(w), [0] * len(w)),
        [0.6698986705489816, 1.0, 0, 4985811.050871694, 7051001.607572404],
        atol=1e-7,
    )

    assert_allclose(
        liu_sf(3.245, [1.02, 3.0], [1] * 2, [0] * 2),
        [0.4054052698281726, 1.2854059895411523, 0, 4.02, 4.481160563961081],
    )

    assert_allclose(
        liu_sf(2, [0.5, 0.4, 0.1], [1, 2, 1], [1, 0.6, 0.8])[0], 0.4577529852208846
    )
    assert_allclose(
        liu_sf(6, [0.5, 0.4, 0.1], [1, 2, 1], [1, 0.6, 0.8])[0], 0.0310791861868015
    )
    assert_allclose(
        liu_sf(3.5, [0.35, 0.15, 0.35, 0.15], [1, 1, 6, 2], [6, 2, 6, 2])[0],
        0.9563148345837034,
    )
    assert_allclose(liu_sf(2, [0.7, 0.3], [2, 1], [1.160032, 2])[0], 0.6243581256085478)
