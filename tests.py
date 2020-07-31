from pmprophet.model import PMProphet, Sampler, Seasonality
import pandas as pd
import numpy as np
import pymc3 as pm


def test_manning_reduced_six_months():
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    df = df.head(180)
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name="model")
    m.add_seasonality(seasonality=7, fourier_order=3)
    m.add_seasonality(seasonality=30, fourier_order=3)
    m.fit()
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_manning():
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    df = df.head(180)
    m = PMProphet(df, growth=True, intercept=True, name="model")
    m.add_seasonality(seasonality=30, fourier_order=3)
    m.fit(method=Sampler.METROPOLIS, draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_automatic_changepoints_manning():
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name="model")
    m.add_seasonality(seasonality=365, fourier_order=3)
    m.fit(method=Sampler.METROPOLIS)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_multiplicative_seasonality():
    z = np.sin(np.linspace(0, 200, 200)) * np.linspace(0, 200, 200)
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2018-01-01", periods=200)
    df["y"] = z
    m = PMProphet(df, auto_changepoints=False, growth=True, intercept=False, name="model")
    with m.model:
        m.priors['growth'] = pm.Constant('growth_model', 1)
    m.add_seasonality(seasonality=3.14 * 2, fourier_order=3, mode=Seasonality.MULTIPLICATIVE)
    m.fit()
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_automatic_changepoints_3_funneling_predictions():
    deltas = np.random.normal(scale=.1, size=200)
    y = np.cumsum(deltas)
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2018-01-01", periods=200)
    df["y"] = y
    m = PMProphet(df, auto_changepoints=True, growth=True, name="model")
    m.fit(method=Sampler.METROPOLIS, chains=1, draws=2000)
    m.predict(200, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_automatic_changepoints():
    z = np.arange(200) + np.concatenate([np.zeros(100), np.arange(100) * -2])
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2018-01-01", periods=200)
    df["y"] = z
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name="model")
    m.fit(method=Sampler.METROPOLIS, draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_automatic_changepoints_2():
    z = (
            np.arange(200)
            + np.concatenate([np.arange(60) * 2, np.arange(60) * -2, np.arange(80) * 5])
            + np.random.normal(0, 2, size=200)
    )
    df = pd.DataFrame()
    df["ds"] = pd.date_range(start="2018-01-01", periods=200)
    df["y"] = z
    m = PMProphet(
        df, auto_changepoints=True, growth=True, intercept=False, name="model"
    )
    m.fit(method=Sampler.METROPOLIS, draws=1000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(intercept=False)


def test_seasonality_shape():
    """
    Verify that that the size of the fourier timeseries is correct
    """
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name="model")
    m.add_seasonality(seasonality=3, fourier_order=3, mode=Seasonality.ADDITIVE)
    m.add_seasonality(seasonality=30, fourier_order=3, mode=Seasonality.ADDITIVE)
    m.fit(method=Sampler.METROPOLIS, draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=False)
    assert m.trace['seasonality_model'].shape[1] == 12  # (3 + 3) * 2 (sin and cos)


if __name__ == "__main__":
    test_manning()
    test_manning_reduced_six_months()
    test_multiplicative_seasonality()
    test_automatic_changepoints()
    test_automatic_changepoints_2()
    test_automatic_changepoints_3_funneling_predictions()
    test_seasonality_shape()
