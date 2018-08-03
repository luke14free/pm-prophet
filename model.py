import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pymc3 as pm


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
                periods=n_change_points
            )[:-1]  # Exclude last change-point

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
        total_growth = 0
        for idx, i in enumerate(self.change_points):
            start = self.data.ix[(self.data['ds'] - i).abs().argsort()[:1]].index[0]
            if idx + 1 == len(self.change_points):
                end = len(self.data)  # last change point is valid till the end of the dataset
            else:
                end = self.data.ix[(self.data['ds'] - self.change_points[idx + 1]).abs().argsort()[:1]].index[0]
            growth = np.arange(0, len(self.data), dtype='float64')
            growth[0:start - 1] = 0
            growth[end + 1: len(growth)] = 0
            growth *= self.priors['change_points'][idx] if prior else np.percentile(
                self.trace['change_points_%s' % self.name][idx], pct, axis=0)
            total_growth += growth
        if total_growth is 0:
            total_growth = np.arange(0, len(self.data), dtype='float64')
        total_growth *= self.priors['growth'] if prior else np.percentile(self.trace['growth_%s' % self.name], pct,
                                                                          axis=0)
        return total_growth

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

    def fit(self, draws=500, method='NUTS', map_initialization=True, finalize=True):
        if finalize:
            self.finalize_model()

        with self.model:
            if map_initialization:
                self.start = pm.find_MAP()

            if draws:
                if method == 'NUTS':
                    self.trace = pm.sample(draws, start=self.start if map_initialization else None)
                else:
                    res = pm.fit(draws, start=self.start if map_initialization else None)
                    self.trace = res.sample(10 ** 4)
                return self.trace
            return self.start

    def plot_components(self, model=True, seasonality=True, growth=True, regressors=True, intercept=True):
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
        if model:
            self._plot_model(ddf)
        if seasonality and self.seasonality:
            self._plot_seasonality(ddf)
        if growth and self.growth:
            self._plot_growth(ddf)
        if intercept and self.intercept:
            self._plot_intercept(ddf)
        if regressors and self.regressors:
            self._plot_regressors()

    def _plot_growth(self, ddf):
        g_trace = self._fit_growth(prior=False)
        ddf['growth_mid'] = np.arange(0, len(ddf)) * np.percentile(g_trace, 50, axis=0)
        ddf['growth_low'] = np.arange(0, len(ddf)) * np.percentile(g_trace, 2, axis=0)
        ddf['growth_high'] = np.arange(0, len(ddf)) * np.percentile(g_trace, 98, axis=0)
        ddf.plot(x='ds', y='growth_mid')
        plt.title("Model Growth")
        plt.fill_between(
            ddf['ds'].values,
            ddf['growth_low'].values.astype(float),
            ddf['growth_high'].values.astype(float),
            alpha=.3
        )
        plt.show()

    def _plot_intercept(self, ddf):
        g_trace = self.trace['intercept_%s' % self.name]
        ddf['intercept_mid'] = np.ones(len(ddf)) * np.percentile(g_trace, 50, axis=0)
        ddf['intercept_low'] = np.ones(len(ddf)) * np.percentile(g_trace, 2, axis=0)
        ddf['intercept_high'] = np.ones(len(ddf)) * np.percentile(g_trace, 98, axis=0)
        ddf.plot(x='ds', y='intercept_mid')
        plt.title("Model Intercept")
        plt.fill_between(
            ddf['ds'].values,
            ddf['intercept_low'].values.astype(float),
            ddf['intercept_high'].values.astype(float),
            alpha=.3
        )
        plt.show()

    @staticmethod
    def _plot_model(ddf):
        ddf.plot(ax=plt.gca(), x='ds', y='y_mid')
        ddf.plot('ds', 'y', style='k.', ax=plt.gca())
        plt.fill_between(
            ddf['ds'].values,
            ddf['y_low'].values.astype(float),
            ddf['y_high'].values.astype(float),
            alpha=.3
        )
        plt.title("Model")
        plt.axes().xaxis.label.set_visible(False)
        plt.legend(['fitted', 'observed'])
        plt.show()

    def _plot_regressors(self):
        pm.forestplot(self.trace, varnames=['regressors_%s' % self.name], ylabels=self.regressors)
        plt.show()

    def _plot_seasonality(self, ddf):
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

            graph.plot(
                y="%s_mid" % int(period),
                x='ds',
                color='C0',
                legend=False
            )

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
    m = PMProphet(df, intercept=True, growth=True, name='model')
    m.add_seasonality(365.25, 2)
    m.add_seasonality(7, 2)
    m.fit(draws=100)
    m.plot_components()
