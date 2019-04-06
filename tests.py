from pmprophet.model import PMProphet
import pandas as pd
import numpy as np


def test_manning_reduced_six_months():
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    df = df.head(180)
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name='model')
    m.add_seasonality(seasonality=7, fourier_order=3)
    m.add_seasonality(seasonality=30, fourier_order=3)
    m.fit(method='Metropolis', draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(
        intercept=False,
    )


def test_manning():
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    df = df.head(180)
    m = PMProphet(df, growth=True, intercept=True, name='model')
    m.add_seasonality(seasonality=30, fourier_order=3)
    m.add_seasonality(seasonality=365, fourier_order=3)
    m.fit(method='Metropolis', draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(
        intercept=False,
    )


def test_automatic_changepoints_manning():
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name='model')
    m.add_seasonality(seasonality=365, fourier_order=3)
    m.fit(method='Metropolis', draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(
        intercept=False,
    )


def test_automatic_changepoints():
    z = np.arange(200) + np.concatenate([np.zeros(100), np.arange(100) * -2])
    df = pd.DataFrame()
    df['ds'] = pd.date_range(start='2018-01-01', periods=200)
    df['y'] = z
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=True, name='model')
    m.fit(method='Metropolis', draws=2000)
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(
        intercept=False,
    )


def test_automatic_changepoints_2():
    z = np.arange(200) + np.concatenate([
        np.arange(60) * 2,
        np.arange(60) * -2,
        np.arange(80) * 5,
    ])
    df = pd.DataFrame()
    df['ds'] = pd.date_range(start='2018-01-01', periods=200)
    df['y'] = z
    m = PMProphet(df, auto_changepoints=True, growth=True, intercept=False, name='model')
    m.fit(method='NUTS')
    m.predict(60, alpha=0.2, include_history=True, plot=True)
    m.plot_components(
        intercept=False,
    )


if __name__ == "__main__":
    test_manning()
    test_manning_reduced_six_months()
    test_automatic_changepoints_manning()
    test_automatic_changepoints()
    test_automatic_changepoints_2()
