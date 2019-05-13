from chiscore import liu_sf
from numpy.testing import assert_allclose


def test_liu_sf():

    w = [0] * 64 + [4985811.050871694]
    t = 906024.952053349
    r = liu_sf(t, w, [1] * len(w), [0] * len(w))
    assert_allclose(r[:-1], [0.6698986705489816, 1.0, 0], atol=1e-7)
    info = r[-1]
    assert_allclose(info["mu_q"], 4985811.050871694)
    assert_allclose(info["sigma_q"], 7051001.607572404)

    r = liu_sf(3.245, [1.02, 3.0], [1] * 2, [0] * 2)
    assert_allclose(r[:-1], [0.4054052698281726, 1.2854059895411523, 0])
    info = r[-1]
    assert_allclose(info["mu_q"], 4.02)
    assert_allclose(info["sigma_q"], 4.481160563961081)

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

    q = liu_sf([0.2, 0.5, 13.3], [0.7, 0.3], [2, 1], [1.160032, 2])[0]
    assert_allclose(q, [0.9863204788738281, 0.9475226498132635, 0.0022189212777912193])
