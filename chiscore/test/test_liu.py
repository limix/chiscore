from chiscore import mod_liu, data_file
from numpy import load
from numpy.testing import assert_allclose


def test_mod_liu():
    with data_file("mod_liu.npz") as filepath:
        data = load(filepath)

    assert_allclose(
        mod_liu(data["args"][0][0], data["args"][1]),
        [0.6698986703936864, 4985811.050871694, 7051001.607572404, 1.],
    )

    assert_allclose(
        mod_liu([3.245], [1.02, 3.0]),
        [0.4017898342526547, 4.02, 4.481160563961081, 1.2281511342584952],
    )


def main():
    print([3.245], [1.02, 3.0])
    print(mod_liu([3.245], [1.02, 3.0]))


if __name__ == "__main__":
    main()
