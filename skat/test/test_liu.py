from skat import skat_mod_liu, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_skat_mod_liu():
    with data_file("skat_mod_liu.npz") as filepath:
        data = load(filepath)

    assert_allclose(skat_mod_liu(data["args"][0][0], data["args"][1]), data["pliumod"])
