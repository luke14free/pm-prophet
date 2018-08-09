import math

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pymc3 as pm


class PMProphet:
    def __init__(self, data, growth=False, intercept=True, model=None, name=None, change_points=[], n_change_points=0):
        self.data = data.copy()
        self.data['ds'] = pd.to_datetime(arg=self.data['ds'])
        self.data.index = range(len(self.data))
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
                self.priors['sigma'] = pm.HalfCauchy('sigma_%s' % self.name, 10, testval=1.)
            if 'seasonality' not in self.priors and self.seasonality:
                self.priors['seasonality'] = pm.Laplace('seasonality_%s' % self.name, 0, 10,
                                                        shape=len(self.seasonality))
            if 'holidays' not in self.priors and self.holidays:
                self.priors['holidays'] = pm.Laplace('holidays_%s' % self.name, 0, 10, shape=len(self.holidays))
            if 'regressors' not in self.priors and self.regressors:
                self.priors['regressors'] = pm.Normal('regressors_%s' % self.name, 0, 10,
                                                      shape=len(self.regressors))
            if self.growth and 'growth' not in self.priors:
                self.priors['growth'] = pm.Normal('growth_%s' % self.name, 0, 0.5)
            if self.growth and 'change_points' not in self.priors and len(self.change_points):
                self.priors['change_points'] = pm.Laplace('change_points_%s' % self.name, 0, 0.5,
                                                          shape=len(self.change_points))
            if self.intercept and 'intercept' not in self.priors:
                self.priors['intercept'] = pm.Normal('intercept_%s' % self.name, self.data['y'].mean(),
                                                     self.data['y'].std() * 2, testval=1.0)

    def fit_growth(self, prior=True):
        s = [self.data.ix[(self.data['ds'] - i).abs().argsort()[:1]].index[0] for i in self.change_points]
        g = self.priors['growth'] if prior else self.trace['growth_%s' % self.name]

        x = np.arange(len(self.data)) if prior else np.array([np.arange(len(self.data))] * len(g)).T

        def d(i):
            return self.priors['change_points'][i] if prior else self.trace['change_points_%s' % self.name][:, i]

        output = x * g
        if s:
            output = []
            for i in range(len(s) + 1):
                local_growth = (g + d(i - 1) if i else 0)
                local_x = (x - s[i - 1] if i else 0)
                local_cond = (x > s[i - 1] if i else 0)
                local_regression = local_growth * local_x * local_cond
                if local_regression is 0 and not prior:
                    output.append(np.zeros(x.shape[1]))
                else:
                    output.append(local_regression)
            output = np.sum(output, axis=0)

        return output

    def _prepare_fit(self):
        self.generate_priors()

        y = np.zeros(len(self.data))
        if self.intercept:
            y += self.priors['intercept']

        if self.growth:
            y += self.fit_growth()

        regressors = np.zeros(len(self.data))
        for idx, regressor in enumerate(self.regressors):
            regressors += self.priors['regressors'][idx] * self.data[regressor]
        holidays = np.zeros(len(self.data))
        for idx, holiday in enumerate(self.holidays):
            holidays += self.priors['holidays'][idx] * self.data[holiday]

        seasonality = np.zeros(len(self.data))
        for idx, seasonal_component in enumerate(self.seasonality):
            seasonality += self.data[seasonal_component].values * self.priors['seasonality'][idx]
        # seasonality *= self.data['y'].mean()

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
            pm.Normal(
                'y_%s' % self.name,
                mu=(self.y - self.data['y'].mean()) / self.data['y'].std(),
                sd=self.priors['sigma'],
                observed=(self.data['y'] - self.data['y'].mean()) / self.data['y'].std()
            )
            pm.Deterministic('y_hat_%s' % self.name, self.y)

    def fit(self, draws=500, method='NUTS', map_initialization=False, finalize=True, step_kwargs={}, sample_kwargs={}):
        if finalize:
            self.finalize_model()

        with self.model:
            if map_initialization:
                self.start = pm.find_MAP(maxeval=10000)

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

    def predict(self, forecasting_periods=10, freq='D', extra_data=None, include_history=True, alpha=0.05, plot=False):
        last_date = self.data['ds'].max()
        dates = pd.date_range(
            start=last_date,
            periods=forecasting_periods + 1,  # An extra in case we include start
            freq=freq)
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:forecasting_periods]  # Return correct number of periods

        new_df = pd.DataFrame()

        if include_history:
            new_df['y'] = np.concatenate([self.data['y'], np.zeros(forecasting_periods) * np.nan])
            new_df['ds'] = np.concatenate([self.data['ds'], dates])
        else:
            new_df['y'] = np.zeros(forecasting_periods)
            new_df['ds'] = dates

        if extra_data is not None:
            for column in extra_data.columns:
                if column not in ['y', 'ds']:
                    new_df[column] = extra_data[column]

        m = PMProphet(
            data=new_df,
            growth=self.growth,
            intercept=self.intercept,
            model=self.model,
            name=self.name
        )

        m.change_points = self.change_points

        periods = {}
        for column in self.data.columns:
            if column.startswith("f_"):
                period, order = column[2:].split("_")
                periods.setdefault(period, [])
                periods[period].append(int(order))

        for period, orders in periods.items():
            m.add_seasonality(seasonality=float(period), order=max(orders) + 1)

        m.priors = self.priors
        m.trace = self.trace

        # Start with the trend
        y_hat = m.fit_growth(prior=False)

        # Add seasonality
        y_hat += m.fit_seasonality(flatten_components=True)

        # Add intercept
        y_hat += self.trace['intercept_%s' % self.name]

        y_hat_noised = y_hat + np.random.normal(0, self.trace['sigma_%s' % self.name])

        ddf = pd.DataFrame(
            [
                np.percentile(y_hat, 50, axis=-1),
                np.percentile(y_hat_noised, math.ceil(100 - (100 * alpha / 2)), axis=-1),
                np.percentile(y_hat_noised, math.floor(100 * alpha / 2), axis=-1),
            ]
        ).T
        ddf['ds'] = m.data['ds']
        ddf.columns = ['y_hat', 'y_high', 'y_low', 'ds']

        if plot:
            plt.figure(figsize=(20, 10))
            ddf.plot('ds', 'y_hat', ax=plt.gca())
            ddf['orig_y'] = self.data['y']
            plt.fill_between(
                ddf['ds'].values,
                ddf['y_low'].values.astype(float),
                ddf['y_high'].values.astype(float),
                alpha=.3
            )

            ddf.plot('ds', 'orig_y', style='k.', ax=plt.gca(), alpha=.2)
            for change_point in m.change_points:
                plt.axvline(change_point, color='C2', lw=1, ls='dashed')
            plt.axvline(pd.to_datetime(self.data.ds).max(), color='C3', lw=1, ls='dotted')
            plt.show()

        return ddf

    def make_trend(self, alpha):
        fitted_growth = self.fit_growth(prior=False)
        ddf = pd.DataFrame(
            [
                self.data['ds'].astype(str),
                self.data['y'],
                np.mean(fitted_growth, axis=-1),
                np.percentile(fitted_growth, math.floor(alpha * 100 / 2), axis=-1),
                np.percentile(fitted_growth, math.ceil(100 - alpha * 100 / 2), axis=-1),
            ]).T

        ddf.columns = ['ds', 'y', 'y_mid', 'y_low', 'y_high']
        ddf.loc[:, 'ds'] = pd.to_datetime(ddf['ds'])
        return ddf

    def plot_components(self, seasonality=True, growth=True, regressors=True, intercept=True, change_points=True,
                        plt_kwargs={}, alpha=0.05):
        if not plt_kwargs:
            plt_kwargs = {'figsize': (20, 10)}

        if seasonality and self.seasonality:
            self._plot_seasonality(alpha, plt_kwargs)
        if growth and self.growth:
            self._plot_growth(alpha, plt_kwargs)
        if intercept and self.intercept:
            self._plot_intercept(alpha, plt_kwargs)
        if regressors and self.regressors:
            self._plot_regressors(alpha, plt_kwargs)
        if change_points and len(self.change_points):
            self._plot_change_points(alpha, plt_kwargs)

    def _plot_growth(self, alpha, plot_kwargs):
        ddf = self.make_trend(alpha)
        g = self.fit_growth(prior=False)
        ddf['growth_mid'] = np.percentile(g, 50, axis=-1)
        ddf['growth_low'] = np.percentile(g, 98, axis=-1)
        ddf['growth_high'] = np.percentile(g, 2, axis=-1)
        plt.figure(**plot_kwargs)
        ddf.plot(x='ds', y='growth_mid', ax=plt.gca())
        plt.title("Model Growth")
        plt.fill_between(
            ddf['ds'].values,
            ddf['growth_low'].values.astype(float),
            ddf['growth_high'].values.astype(float),
            alpha=.3
        )
        for change_point in self.change_points:
            plt.axvline(change_point, color='C2', lw=1, ls='dashed')
        plt.grid()
        plt.show()

    def _plot_intercept(self, alpha, plot_kwargs):
        plt.figure(**plot_kwargs)
        pm.forestplot(self.trace, varnames=['intercept_%s' % self.name], ylabels="Intercept")
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

    def _plot_regressors(self, alpha, plot_kwargs):
        plt.figure(**plot_kwargs)
        pm.forestplot(self.trace, varnames=['regressors_%s' % self.name], ylabels=self.regressors)
        plt.show()

    def fit_seasonality(self, flatten_components=False):
        periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))
        idx = 0
        ts = np.zeros((len(periods), len(self.data), self.trace['seasonality_%s' % self.name].shape[0]))
        for pdx, period in enumerate(periods):
            cols = [i for i in self.seasonality if float(i.split("_")[1]) == period]
            for col in cols:
                s_trace = self.trace['seasonality_%s' % self.name][:, idx]
                ts[pdx, :] += s_trace * np.repeat([self.data[col]], len(s_trace)).reshape(len(self.data), len(s_trace))
                idx += 1
        return ts.sum(axis=0) if flatten_components else ts

    def _plot_change_points(self, alpha, plot_kwargs):
        plt.figure(**plot_kwargs)
        pm.forestplot(self.trace, varnames=['change_points_%s' % self.name], ylabels=self.change_points.astype(str))
        plt.grid()
        plt.title("Growth Change Points")
        plt.show()

    def _plot_seasonality(self, alpha, plot_kwargs):
        periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))
        ts = self.fit_seasonality()
        ddf = pd.DataFrame(
            np.vstack([
                np.percentile(ts, 50, axis=-1),
                np.percentile(ts, 2, axis=-1),
                np.percentile(ts, 98, axis=-1)
            ]).T,
            columns=["%s_%s" % (p, l) for l in ['mid', 'low', 'high'] for p in periods]
        )
        ddf['ds'] = self.data['ds']
        for period in periods:
            graph = ddf.head(int(period))
            if period == 7:
                graph.loc[:, 'ds'] = [str(i[:3]) for i in graph['ds'].dt.weekday_name]
            plt.figure(**plot_kwargs)
            graph.plot(
                y="%s_mid" % period,
                x='ds',
                color='C0',
                legend=False,
                ax=plt.gca()
            )
            plt.grid()

            if period == 7:
                plt.xticks(range(7), graph['ds'].values)

            plt.fill_between(
                graph['ds'].values,
                graph["%s_low" % period].values.astype(float),
                graph["%s_high" % period].values.astype(float),
                alpha=.3,
            )

            plt.title("Model Seasonality for period: %s days" % period)
            plt.axes().xaxis.label.set_visible(False)
            plt.show()
