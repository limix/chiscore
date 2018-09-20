from chiscore import mod_liu, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_mod_liu():
    with data_file("mod_liu.npz") as filepath:
        data = load(filepath)

    assert_allclose(mod_liu(data["args"][0][0], data["args"][1]), data["pliumod"])
