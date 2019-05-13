import shutil
import tempfile
import warnings
from os.path import dirname, join, realpath

_filenames = [
    "davies_pvalue.npz",
    "optimal_davies_pvalue.npz",
    "danilo_nan.npz",
    "bound.npz",
    "inf.npz",
]


class data_file(object):
    def __init__(self, filenames):
        global _filenames

        self._unlist = False
        if not isinstance(filenames, (tuple, list)):
            filenames = [filenames]
            self._unlist = True

        for fn in filenames:
            if fn not in _filenames:
                raise ValueError(
                    "Unrecognized file name {}. Choose one of these: {}".format(
                        fn, _filenames
                    )
                )

        self._dirpath = tempfile.mkdtemp()
        self._filenames = filenames

    def __enter__(self):
        import pkg_resources

        filepaths = [join(self._dirpath, fn) for fn in self._filenames]

        for fn, fp in zip(self._filenames, filepaths):
            if __name__ == "__main__":
                shutil.copy(join(dirname(realpath(__file__)), fn), fp)
            else:
                resource_path = "_data/{}".format(fn)
                content = pkg_resources.resource_string(
                    __name__.split(".")[0], resource_path
                )

                with open(fp, "wb") as f:
                    f.write(content)

        if self._unlist:
            return filepaths[0]
        return filepaths

    def __exit__(self, *_):
        try:
            shutil.rmtree(self._dirpath)
        except PermissionError as e:
            warnings.warn(str(e) + "\n. I will ignore it and proceed.")
