# chiscore

[![Travis](https://img.shields.io/travis/com/limix/chiscore.svg?style=flat-square&label=linux%20%2F%20macos%20build)](https://travis-ci.com/limix/chiscore) [![AppVeyor](https://img.shields.io/appveyor/ci/Horta/chiscore.svg?style=flat-square&label=windows%20build)](https://ci.appveyor.com/project/Horta/chiscore)

Estimate the joint significance of test statistics derived from linear combination
of chi-squared distributions.

## Install

We recommend installing it via
[conda](http://conda.pydata.org/docs/index.html):

```bash
conda install -c conda-forge chiscore
```

Alternatively, chiscore can also be installed using
[pip](https://pypi.python.org/pypi/pip):

```bash
pip install chiscore
```

## Running the tests

After installation, you can test it

```bash
python -c "import chiscore; chiscore.test()"
```

as long as you have [pytest](https://docs.pytest.org/en/latest/).

## Usage

```python
>>> from chiscore import davies_pvalue
>>> q = 1.5
>>> w = [[0.3, 5.0], [5.0, 1.5]]
>>> davies_pvalue(q, w)
{'p_value': 0.6151796819770086, 'param': {'liu_pval': 0.6151796819770086, 'Is_Converged': 1.0}, 'p_value_resampling': None, 'pval_zero_msg': None}
```

```python
>>> from chiscore import mod_liu
>>> q = 1.5
>>> w = [0.3, 5.0]
>>> mod_liu(q, w)
(0.6230031759923031, 5.3, 7.083784299369935, 1.0071999066892092)
```

```python
>>> from chiscore import optimal_davies_pvalue
>>> q = [1.5, 3.0]
>>> mu = -0.5
>>> var = 1.0
>>> kur = 3.0
>>> w = [10.0, 0.2, 0.1, 0.3]
>>> remain_var = 0.5
>>> df = 3.4
>>> trho = [5.1, 0.2]
>>> grid = [0., 0.01]
>>> optimal_davies_pvalue(q, mu, var, kur, w, remain_var, df, trho, grid)
0.966039962464624
```

## Authors

* [Danilo Horta](https://github.com/horta)

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/limix/chiscore/master/LICENSE.md).
