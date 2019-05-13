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

### Davies

```python
>>> from chiscore import davies_pvalue
>>> q = 1.5
>>> w = [[0.3, 5.0], [5.0, 1.5]]
>>> davies_pvalue(q, w)
{'p_value': 0.6151796819770086, 'param': {'liu_pval': 0.6151796819770086, 'Is_Converged': 1.0}, 'p_value_resampling': None, 'pval_zero_msg': None}
```

### Liu

Let us approximate

    𝑋 = 0.5⋅χ²(1, 1) + 0.4⋅χ²(2, 0.6) + 0.1⋅χ²(1, 0.8),

and evaluate Pr(𝑋 > 2).

```python
>>> from chiscore import liu_sf
>>>
>>> w = [0.5, 0.4, 0.1]
>>> dofs = [1, 2, 1]
>>> deltas = [1, 0.6, 0.8]
>>> (q, dof, delta, _) = liu_sf(2, w, dofs, deltas)
>>> q
0.4577529852208846
>>> dof
3.5556138890755395
>>> delta
0.7491921870025307
```

Therefore, we have

    Pr(𝑋 > 2) ≈ Pr(χ²(3.56, 0.75) > 𝑡⁺𝜎ₓ + 𝜇ₓ) = 0.458.

### P-value

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

## References

* Lee, Seunggeun, Michael C. Wu, and Xihong Lin. "Optimal tests for rare variant
  effects in sequencing association studies." Biostatistics 13.4 (2012): 762-775.
* Liu, H., Tang, Y., & Zhang, H. H. (2009). A new chi-square approximation to the
  distribution of non-negative definite quadratic forms in non-central normal
  variables. Computational Statistics & Data Analysis, 53(4), 853-856.

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/limix/chiscore/master/LICENSE.md).
