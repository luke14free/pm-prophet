import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T


class PMProphet:
    def __init__(self, data, growth=False, intercept=True, model=None, name=None, change_points=[], n_change_points=0):
        self.data = data.copy()
        self.data['ds'] = pd.to_datetime(arg=self.data['ds'])
        self.seasonality = []
        self.holidays = []
        self.regressors = []
        self.model = pm.Model() if model is None else model
        self.intercept = intercept
        self.growth = growth
        self.priors = {}
        self.params = {}
        self.trace = {}
        self.start = {}
        self.change_points = pd.DatetimeIndex(change_points)
        self.name = name

        if change_points and change_points:
            raise Exception("You can either specify a list of changepoint dates of a number of them")
        if 'y' not in data.columns:
            raise Exception("Target variable should be called `y` in the `data` dataframe")
        if 'ds' not in data.columns:
            raise Exception("Time variable should be called `ds` in the `data` dataframe")
        if name is None:
            raise Exception("Specify a model name through the `name` parameter")

        if n_change_points:
            self.change_points = pd.date_range(
                start=pd.to_datetime(self.data['ds'].min()),
                end=pd.to_datetime(self.data['ds'].max()),
                periods=n_change_points + 2
            )[1:-1]  # Exclude first and last change-point

    @staticmethod
    def fourier_series(dates, period, series_order):
        t = np.array(
            (dates - pd.datetime(1970, 1, 1))
                .dt.total_seconds()
                .astype(np.float)
        ) / (3600 * 24.)
        return np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(series_order)
            for fun in (np.sin, np.cos)
        ])

    def add_seasonality(self, seasonality, order):
        self.seasonality.extend(['f_%s_%s' % (seasonality, order_idx) for order_idx in range(order)])
        fourier_series = PMProphet.fourier_series(
            pd.to_datetime(self.data['ds']), seasonality, order
        )
        for order_idx in range(order):
            self.data['f_%s_%s' % (seasonality, order_idx)] = fourier_series[:, order_idx]

    def add_holiday(self, name, date_start, date_end):
        self.data[name] = ((self.data.ds > date_start) & (self.data.ds < date_end)).astype(int) * self.data['y'].mean()
        self.holidays.append(name)

    def add_regressor(self, name, regressor=None):
        self.regressors.append(name)
        if regressor:
            self.data[name] = regressor

    def generate_priors(self):
        with self.model:
            if 'sigma' not in self.priors:
                self.priors['sigma'] = pm.HalfCauchy('sigma_%s' % self.name, 100, testval=1.)
            if 'seasonality' not in self.priors and self.seasonality:
                self.priors['seasonality'] = pm.Laplace('seasonality_%s' % self.name, 0, 1,
                                                        shape=len(self.seasonality))
            if 'holidays' not in self.priors and self.holidays:
                self.priors['holidays'] = pm.Laplace('holidays_%s' % self.name, 0, 10, shape=len(self.holidays))
            if 'regressors' not in self.priors and self.regressors:
                self.priors['regressors'] = pm.Normal('regressors_%s' % self.name, 0, 10,
                                                      shape=len(self.regressors))
            if self.growth and 'growth' not in self.priors:
                self.priors['growth'] = pm.Laplace('growth_%s' % self.name, 0, 20)
            if self.growth and 'change_points' not in self.priors and len(self.change_points):
                self.priors['change_points'] = pm.Laplace('change_points_%s' % self.name, 0, 20,
                                                          shape=len(self.change_points))
            if self.intercept and 'intercept' not in self.priors:
                self.priors['intercept'] = pm.Gamma('intercept_%s' % self.name, mu=self.data['y'].mean(), sd=200,
                                                    testval=1.0)

    def _fit_growth(self, prior=True, pct=50):
        x = np.arange(len(self.data), dtype='float64') if prior else np.ones(len(self.data), dtype='float64')

        s = [self.data.ix[(self.data['ds'] - i).abs().argsort()[:1]].index[0] for i in self.change_points]
        g = self.priors['growth'] if prior else np.percentile(self.trace['growth_%s' % self.name], pct, axis=0)

        def d(i):
            return self.priors['change_points'][i] if prior else np.percentile(
                self.trace['change_points_%s' % self.name][i], pct, axis=0)

        return sum([
            (x - s[i - 1] if i else 0) * (g + d(i - 1) if i else 0) * (x > s[i - 1] if i else 0)
            for i in range(len(s) + 1)
        ])

    def _prepare_fit(self):
        self.generate_priors()

        y = np.zeros(len(self.data))
        if self.intercept:
            y += self.priors['intercept']

        if self.growth:
            y += self._fit_growth()

        regressors = np.zeros(len(self.data))
        for idx, regressor in enumerate(self.regressors):
            regressors += self.priors['regressors'][idx] * self.data[regressor]
        holidays = np.zeros(len(self.data))
        for idx, holiday in enumerate(self.holidays):
            holidays += self.priors['holidays'][idx] * self.data[holiday]

        seasonality = np.zeros(len(self.data))
        for idx, seasonal_component in enumerate(self.seasonality):
            seasonality += self.data[seasonal_component].values * self.priors['seasonality'][idx]
        seasonality *= self.data['y'].mean()

        with self.model:
            if self.seasonality:
                pm.Deterministic('seasonality_hat_%s' % self.name, seasonality)
            if self.regressors:
                pm.Deterministic('regressors_hat_%s' % self.name, regressors)
            if self.holidays:
                pm.Deterministic('holidays_hat_%s' % self.name, holidays)

        self.y = y + regressors + holidays + seasonality

    def finalize_model(self):
        self._prepare_fit()
        with self.model:
            pm.Normal('y_%s' % self.name, self.y, self.priors['sigma'], observed=self.data['y'])
            pm.Deterministic('y_hat_%s' % self.name, self.y)

    def fit(self, draws=500, method='NUTS', map_initialization=False, finalize=True, step_kwargs={}, sample_kwargs={}):
        if finalize:
            self.finalize_model()

        with self.model:
            if map_initialization:
                self.start = pm.find_MAP()

            if draws:
                if method == 'NUTS' or method == 'Metropolis':
                    self.trace = pm.sample(
                        draws,
                        step=pm.Metropolis(**step_kwargs) if method == 'Metropolis' else pm.NUTS(**step_kwargs),
                        start=self.start if map_initialization else None,
                        **sample_kwargs
                    )
                else:
                    res = pm.fit(draws, start=self.start if map_initialization else None)
                    self.trace = res.sample(10 ** 4)
                return self.trace
            return self.start

    def plot_components(self, model=True, seasonality=True, growth=True, regressors=True, intercept=True,
                        plt_kwargs={}):
        sigma = np.percentile(self.trace['sigma_%s' % self.name], 50, axis=0)
        ddf = pd.DataFrame(
            [
                self.data['ds'].astype(str),
                self.data['y'],
                np.percentile(self.trace['y_hat_%s' % self.name], 50, axis=0),
                np.percentile(self.trace['y_hat_%s' % self.name], 2, axis=0) + sigma,
                np.percentile(self.trace['y_hat_%s' % self.name], 98, axis=0) - sigma,
            ]).T

        ddf.columns = ['ds', 'y', 'y_mid', 'y_low', 'y_high']
        ddf.loc[:, 'ds'] = pd.to_datetime(ddf['ds'])
        if not plt_kwargs:
            plt_kwargs = {'figsize': (20, 10)}

        if model:
            self._plot_model(ddf, plt_kwargs)
        if seasonality and self.seasonality:
            self._plot_seasonality(ddf, plt_kwargs)
        if growth and self.growth:
            self._plot_growth(ddf, plt_kwargs)
        if intercept and self.intercept:
            self._plot_intercept(ddf, plt_kwargs)
        if regressors and self.regressors:
            self._plot_regressors(plt_kwargs)

    def _plot_growth(self, ddf, plot_kwargs):
        ddf['growth_mid'] = self._fit_growth(prior=False, pct=50)
        ddf['growth_low'] = self._fit_growth(prior=False, pct=2)
        ddf['growth_high'] = self._fit_growth(prior=False, pct=98)
        plt.figure(**plot_kwargs)
        ddf.plot(x='ds', y='growth_mid')
        plt.title("Model Growth")
        plt.fill_between(
            ddf['ds'].values,
            ddf['growth_low'].values.astype(float),
            ddf['growth_high'].values.astype(float),
            alpha=.3
        )
        plt.grid()
        plt.show()

    def _plot_intercept(self, ddf, plot_kwargs):
        g_trace = self.trace['intercept_%s' % self.name]
        ddf['intercept_mid'] = np.ones(len(ddf)) * np.percentile(g_trace, 50, axis=0)
        ddf['intercept_low'] = np.ones(len(ddf)) * np.percentile(g_trace, 2, axis=0)
        ddf['intercept_high'] = np.ones(len(ddf)) * np.percentile(g_trace, 98, axis=0)
        plt.figure(**plot_kwargs)
        ddf.plot(x='ds', y='intercept_mid')
        plt.title("Model Intercept")
        plt.fill_between(
            ddf['ds'].values,
            ddf['intercept_low'].values.astype(float),
            ddf['intercept_high'].values.astype(float),
            alpha=.3
        )
        plt.grid()
        plt.show()

    @staticmethod
    def _plot_model(ddf, plot_kwargs):
        plt.figure(**plot_kwargs)
        ddf.plot(ax=plt.gca(), x='ds', y='y_mid')
        ddf.plot('ds', 'y', style='k.', ax=plt.gca())
        plt.fill_between(
            ddf['ds'].values,
            ddf['y_low'].values.astype(float),
            ddf['y_high'].values.astype(float),
            alpha=.3
        )
        plt.grid()
        plt.title("Model")
        plt.axes().xaxis.label.set_visible(False)
        plt.legend(['fitted', 'observed'])
        plt.show()

    def _plot_regressors(self, plot_kwargs):
        plt.figure(**plot_kwargs)
        pm.forestplot(self.trace, varnames=['regressors_%s' % self.name], ylabels=self.regressors)
        plt.show()

    def _plot_seasonality(self, ddf, plot_kwargs):
        periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))
        for period in periods:
            cols = [i for i in self.seasonality if float(i.split("_")[1]) == period]
            mid = 0
            low = 0
            high = 0
            for col in cols:
                s_trace = self.trace['seasonality_%s' % self.name][:, periods.index(period)]

                mid += ddf['y'].mean() * np.percentile(s_trace, 50, axis=-1) * self.data[col]
                low += ddf['y'].mean() * np.percentile(s_trace, 2, axis=-1) * self.data[col]
                high += ddf['y'].mean() * np.percentile(s_trace, 98, axis=-1) * self.data[col]

            ddf.loc[:, "%s_mid" % int(period)] = mid
            ddf.loc[:, "%s_low" % int(period)] = np.min([low, high], axis=0)
            ddf.loc[:, "%s_high" % int(period)] = np.max([low, high], axis=0)
        for period in periods:
            graph = ddf.head(int(period))
            if int(period) == 7:
                graph.loc[:, 'ds'] = [str(i[:3]) for i in graph['ds'].dt.weekday_name]
            plt.figure(**plot_kwargs)
            graph.plot(
                y="%s_mid" % int(period),
                x='ds',
                color='C0',
                legend=False
            )
            plt.grid()

            if int(period) == 7:
                plt.xticks(range(7), graph['ds'].values)

            plt.fill_between(
                graph['ds'].values,
                graph["%s_low" % int(period)].values.astype(float),
                graph["%s_high" % int(period)].values.astype(float),
                alpha=.3
            )

            plt.title("Model Seasonality for period: %s days" % int(period))
            plt.axes().xaxis.label.set_visible(False)
            plt.show()


if __name__ == '__main__':
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    df['regressor'] = np.random.normal(loc=0, scale=1, size=(len(df)))
    df = df.head(400)

    # Fit both growth and intercept
    m = PMProphet(df, growth=True, intercept=True, n_change_points=20, name='model')

    # Add yearly seasonality (order: 3)
    m.add_seasonality(seasonality=365.5, order=3)

    # Add monthly seasonality (order: 3)
    m.add_seasonality(seasonality=30, order=3)

    # Add weekly seasonality (order: 3)
    m.add_seasonality(seasonality=7, order=2)

    # Add a white noise regressor
    m.add_regressor('regressor')

    # Fit the model (using NUTS, 1000 draws and MAP initialization)
    m.fit(draws=10 ** 5, method='AVDI')
