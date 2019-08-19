# Extended example: BART ridership 2011-2018

This is an example analysis of high-dimensional multivariate count data using the public [BART ridership dataset](https://www.bart.gov/about/reports/ridership).
We generalize the HMM-VAE model [Part III](https://github.com/pyro-ppl/sandbox/blob/master/2019-08-time-series/part_iii_custom.ipynb) to jointly model hourly ridership between every pair of BART stations (over 2000 pairs in total).

- [preprocess.py](https://github.com/pyro-ppl/sandbox/blob/master/2019-08-time-series/bart/preprocess.py) is a script to download and preprocess data. The function `preprocess.load_hourly_od()` returns a dataset with a single `torch.Tensor` of counts, shaped 70128 x 47 x 47.
- [forecast.py](https://github.com/pyro-ppl/sandbox/blob/master/2019-08-time-series/bart/forecast.py) is the main Pyro code containing a model and guide and helpers for training and forecasting.
- [forecast.ipynb](https://github.com/pyro-ppl/sandbox/blob/master/2019-08-time-series/bart/forecast.py) is a notebook documenting how to train a model and forecast forward. It includes plots of result forecasts.
