import datetime
from enum import Enum
from typing import List, Union, Dict

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as tt
import logging


class Seasonality(Enum):
    ADDITIVE = 1
    MULTIPLICATIVE = 2


class Sampler(Enum):
    METROPOLIS = pm.Metropolis
    NUTS = pm.NUTS
    ADVI = pm.ADVI


DateType = Union[datetime.datetime, pd.datetime, str]


class PMProphet:
    """Prophet forecaster.

    Parameters
    ----------
    data : pd.DataFrame (with 'y' and 'ds' columns)
        Data to be used for fitting the model.
    growth : bool
        Include the growth component.
    intercept : bool
        Include the intercept.
    model : PyMC3 model.
        Initialize with a model.
    name : string
        Name of the model. Needed for to generate the theano/pymc3 variables.
    changepoints : list
        List of dates at which to include potential changepoints.
    n_changepoints : int, default: 25
        Number of potential changepoints to include. Either specify this of `changepoints`.
    changepoints_prior_scale : float, default: 0.05
        Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    holidays_prior_scale : float, default: 10.0
        Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    seasonality_prior_scale : float, default: 10.0
        Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality.
    regressors_prior_scale : float, default: 10.0
        Parameter modulating the strength of the
        regressors model. Larger values allow the model to fit larger regressors
        fluctuations, smaller values dampen the regressors.
    positive_regressors_coefficients : bool, default: False
        Parameter forcing the regressors coefficients be positive (i.e. sampled
        from an Exponential instead than from a Laplacian).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        growth: bool = False,
        intercept: bool = True,
        model: pm.Model = None,
        name: str = None,
        changepoints: List = [],
        n_changepoints: int = 25,
        changepoints_prior_scale: float = 2.5,
        holidays_prior_scale: float = 2.5,
        seasonality_prior_scale: float = 2.5,
        regressors_prior_scale: bool = 2.5,
        positive_regressors_coefficients: bool = False,
        auto_changepoints: bool = False,
    ):
        self.data = data.copy()
        self.data["ds"] = pd.to_datetime(arg=self.data["ds"])
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
        self.priors_names = {}
        self.changepoints = pd.DatetimeIndex(changepoints)
        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.regressors_prior_scale = regressors_prior_scale
        self.positive_regressors_coefficients = positive_regressors_coefficients
        self.name = name
        self.multiplicative_data = set([])
        self.skip_first = None
        self.chains = None
        self.changepoint_weights = None
        self.auto_changepoints = auto_changepoints
        self.w = 1

        if len(changepoints) > 0 and n_changepoints > 0:
            logging.warning(
                "ignoring the `n_changepoints` parameter since a list of changepoints were passed"
            )
            n_changepoints = None
        if auto_changepoints:
            logging.info(
                "Enabling autochangepoints, ignoring other changepoints parameters"
            )
            logging.warning(
                "Note that the automatic changepoint feature is experimental"
            )
        if "y" not in data.columns:
            raise Exception(
                "Target variable should be called `y` in the `data` dataframe"
            )
        if "ds" not in data.columns:
            raise Exception(
                "Time variable should be called `ds` in the `data` dataframe"
            )
        if name is None:
            raise Exception("Specify a model name through the `name` parameter")

        if n_changepoints:
            self.changepoints = pd.date_range(
                start=pd.to_datetime(self.data["ds"].min()),
                end=pd.to_datetime(self.data["ds"].max()),
                periods=n_changepoints + 2,
            )[
                1:-1
            ]  # Exclude first and last change-point

    @staticmethod
    def fourier_series(
        dates: pd.Series, period: Union[int, float], series_order: int
    ) -> np.array:
        """Provides Fourier series components with the specified frequency
        and order.

        Parameters
        ----------
        dates : pd.Series
            Timestamps of dates.
        period : int or float
            Number of days of the period.
        series_order : int
            Number of components.

        Returns
        -------
        Matrix with seasonality features.
        """
        t = np.array(
            (dates - pd.datetime(1970, 1, 1)).dt.total_seconds().astype(np.float)
        ) / (3600 * 24.0)
        return np.column_stack(
            [
                fun((2.0 * (i + 1) * np.pi * t / period))
                for i in range(series_order)
                for fun in (np.sin, np.cos)
            ]
        )

    def add_seasonality(
        self,
        seasonality: Union[int, float],
        fourier_order: int,
        mode: Seasonality = Seasonality.ADDITIVE,
    ):
        """Add a seasonal component.

        Parameters
        ----------
        seasonality : int
            Period lenght in day for the seasonality (e.g. 7 for weekly, 30 for daily..)
        fourier_order : int
            Number of Fourier components to use. Minimum is 2.
        mode : str
            Type of modeling. Either 'multiplicative' or 'additive'.

        """
        self.seasonality.extend(
            ["f_%s_%s" % (seasonality, order_idx) for order_idx in range(fourier_order)]
        )
        fourier_series = PMProphet.fourier_series(
            pd.to_datetime(self.data["ds"]), seasonality, fourier_order
        )
        for order_idx in range(fourier_order):
            self.data["f_%s_%s" % (seasonality, order_idx)] = fourier_series[
                :, order_idx
            ]

        if mode == Seasonality.MULTIPLICATIVE:
            for order_idx in range(fourier_order):
                self.multiplicative_data.add("f_%s_%s" % (seasonality, order_idx))

    def add_holiday(
        self,
        name: str,
        date_start: DateType,
        date_end: DateType,
        mode=Seasonality.ADDITIVE,
    ):
        """Add holiday features

        Parameters
        ----------
        name : string
            Name of the holiday component.
        date_start : datetime
            Datetime from which the holiday begins
        date_end : datetime
            Datetime to which the holiday ends
        mode : str
            Type of modeling. Either 'multiplicative' or 'additive'.
        """
        self.data[name] = (
            (self.data.ds > date_start) & (self.data.ds < date_end)
        ).astype(int) * self.data["y"].mean()
        if mode == Seasonality.MULTIPLICATIVE:
            self.multiplicative_data.add(name)

        self.holidays.append(name)

    def add_regressor(
        self, name: str, regressor: np.array = None, mode=Seasonality.ADDITIVE
    ):
        """Add an additional regressor to be used for fitting and predicting.

        Parameters
        ----------
        name : string
            Name of the regressor.
        regressor : np.array, default: None
            optionally pass an array of values to be copied in the model data
        mode : str
            Type of modeling. Either 'multiplicative' or 'additive'.
        """
        self.regressors.append(name)
        if regressor:
            self.data[name] = regressor

        if mode == Seasonality.MULTIPLICATIVE:
            self.multiplicative_data.add(name)

    def generate_priors(self):
        """Set up the priors for the model."""
        with self.model:
            if "sigma" not in self.priors:
                self.priors["sigma"] = pm.HalfCauchy(
                    "sigma_%s" % self.name, 10, testval=1.0
                )

            if "seasonality" not in self.priors and self.seasonality:
                self.priors["seasonality"] = pm.Laplace(
                    "seasonality_%s" % self.name,
                    0,
                    self.seasonality_prior_scale,
                    shape=len(self.seasonality),
                )
            if "holidays" not in self.priors and self.holidays:
                self.priors["holidays"] = pm.Laplace(
                    "holidays_%s" % self.name,
                    0,
                    self.holidays_prior_scale,
                    shape=len(self.holidays),
                )
            if "regressors" not in self.priors and self.regressors:
                if self.positive_regressors_coefficients:
                    self.priors["regressors"] = pm.Exponential(
                        "regressors_%s" % self.name,
                        self.regressors_prior_scale,
                        shape=len(self.regressors),
                    )
                else:
                    self.priors["regressors"] = pm.Laplace(
                        "regressors_%s" % self.name,
                        0,
                        self.regressors_prior_scale,
                        shape=len(self.regressors),
                    )
            if self.growth and "growth" not in self.priors:
                self.priors["growth"] = pm.Normal("growth_%s" % self.name, 0, 0.1)
            if (
                len(self.changepoints)
                and "changepoints" not in self.priors
                and len(self.changepoints)
            ):
                if self.auto_changepoints:
                    k = self.n_changepoints
                    alpha = pm.Gamma("alpha", 1.0, 1.0)
                    beta = pm.Beta("beta", 1.0, alpha, shape=k)
                    w1 = pm.Deterministic(
                        "w1",
                        tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
                        * beta,
                    )
                    w, _ = theano.map(
                        fn=lambda x: tt.switch(tt.gt(x, 1e-4), x, 0), sequences=[w1]
                    )
                    self.w = pm.Deterministic("w", w)
                else:
                    k = len(self.changepoints)
                    w = 1
                cgpt = pm.Deterministic(
                    "cgpt",
                    pm.Laplace("cgpt_inner", 0, self.changepoints_prior_scale, shape=k)
                    * w,
                )
                self.priors["changepoints"] = pm.Deterministic(
                    "changepoints_%s" % self.name, cgpt
                )
            if self.intercept and "intercept" not in self.priors:
                self.priors["intercept"] = pm.Normal(
                    "intercept_%s" % self.name,
                    self.data["y"].mean(),
                    self.data["y"].std() * 2,
                )

        self.priors_names = {k: v.name for k, v in self.priors.items()}

    def _fit_growth(self, prior: bool = True) -> any:
        """Fit the growth component."""
        s = [
            self.data.ix[(self.data["ds"] - i).abs().argsort()[:1]].index[0]
            for i in self.changepoints
        ]
        g = self.priors["growth"] if prior else self.trace[self.priors_names["growth"]]

        x = (
            np.arange(len(self.data))
            if prior
            else np.array([np.arange(len(self.data))] * len(g)).T
        )

        if len(self.changepoints):
            d = (
                self.priors["changepoints"]
                if prior
                else [
                    self.trace[self.priors_names["changepoints"]][:, i]
                    for i in range(len(s))
                ]
            )
        else:
            d = []

        regression = x * g
        if s and d:
            if prior:
                total = len(self.data)
                if (
                    self.auto_changepoints
                ):  # Overwrite changepoints with a uniform prior
                    with self.model:
                        s = pm.Uniform("s", 0, total, shape=len(self.changepoints))
                    piecewise_regression, _ = theano.scan(
                        fn=lambda x: tt.concatenate(
                            [tt.arange(x) * 0, tt.arange(total - x)]
                        ),
                        sequences=[s],
                    )
                    piecewise_regression = (
                        piecewise_regression.T * d.dimshuffle("x", 0)
                    ).sum(axis=-1)
                else:
                    base_piecewise_regression = []

                    for i in s:
                        if i > 0:
                            local_x = x.copy()[:-i]
                            local_x = np.concatenate(
                                [
                                    np.zeros(i)
                                    if prior
                                    else np.zeros((i, local_x.shape[1])),
                                    local_x,
                                ]
                            )
                            base_piecewise_regression.append(local_x)

                    piecewise_regression = np.array(base_piecewise_regression)
                    piecewise_regression = (
                        piecewise_regression.T * d.dimshuffle("x", 0)
                    ).sum(axis=-1)
            else:
                base_piecewise_regression = []

                for i in s:
                    local_x = x.copy()[:-i]
                    local_x = np.concatenate(
                        [
                            np.zeros(i) if prior else np.zeros((i, local_x.shape[1])),
                            local_x,
                        ]
                    )
                    base_piecewise_regression.append(local_x)

                piecewise_regression = np.array(base_piecewise_regression)
                d = np.array(d)
                regression_pieces = [
                    piecewise_regression[i] * d[i] for i in range(len(s))
                ]
                piecewise_regression = np.sum(
                    [piece for piece in regression_pieces if len(piece)], axis=0
                )
            regression = piecewise_regression

        return regression

    def _prepare_fit(self):
        self.generate_priors()

        multiplicative_regressors = np.zeros(len(self.data))
        additive_regressors = np.zeros(len(self.data))

        for idx, regressor in enumerate(self.regressors):
            if regressor in self.multiplicative_data:
                multiplicative_regressors += (
                    self.priors["regressors"][idx] * self.data[regressor]
                )
            else:
                additive_regressors += (
                    self.priors["regressors"][idx] * self.data[regressor]
                )

        additive_holidays = np.zeros(len(self.data))
        multiplicative_holidays = np.zeros(len(self.data))
        for idx, holiday in enumerate(self.holidays):
            if holiday in self.multiplicative_data:
                multiplicative_holidays += (
                    self.priors["holidays"][idx] * self.data[holiday]
                )
            else:
                additive_holidays += self.priors["holidays"][idx] * self.data[holiday]

        additive_seasonality = np.zeros(len(self.data))
        multiplicative_seasonality = np.zeros(len(self.data))
        for idx, seasonal_component in enumerate(self.seasonality):
            if seasonal_component in self.multiplicative_data:
                multiplicative_seasonality += (
                    self.data[seasonal_component].values
                    * self.priors["seasonality"][idx]
                )
            else:
                additive_seasonality += (
                    self.data[seasonal_component].values
                    * self.priors["seasonality"][idx]
                )

        with self.model:
            if self.seasonality:
                if not isinstance(additive_seasonality, np.ndarray):
                    pm.Deterministic(
                        "additive_seasonality_hat_%s" % self.name, additive_seasonality
                    )
                if not isinstance(multiplicative_seasonality, np.ndarray):
                    pm.Deterministic(
                        "multiplicative_seasonality_hat_%s" % self.name,
                        multiplicative_seasonality,
                    )
            if self.regressors:
                if not isinstance(additive_regressors, np.ndarray):
                    pm.Deterministic(
                        "additive_regressors_hat_%s" % self.name, additive_regressors
                    )
                if not isinstance(multiplicative_regressors, np.ndarray):
                    pm.Deterministic(
                        "multiplicative_regressors_hat_%s" % self.name,
                        multiplicative_regressors,
                    )
            if self.holidays:
                if not isinstance(additive_holidays, np.ndarray):
                    pm.Deterministic(
                        "additive_holidays_hat_%s" % self.name, additive_holidays
                    )
                if not isinstance(multiplicative_holidays, np.ndarray):
                    pm.Deterministic(
                        "multiplicative_holidays_hat_%s" % self.name,
                        multiplicative_holidays,
                    )

        multiplicative_terms = [
            i
            for i in [
                multiplicative_regressors,
                multiplicative_seasonality,
                multiplicative_holidays,
            ]
            if not isinstance(i, np.ndarray)
        ]

        if multiplicative_terms and not self.growth:
            raise Exception(
                "Multiplicative terms require a model with trend; i.e. `growth=True`"
            )

        y = np.zeros(len(self.data))

        if self.growth:
            if multiplicative_terms:
                y = self._fit_growth() * sum(multiplicative_terms)
            else:
                y = self._fit_growth()

        if self.intercept:
            y += self.priors["intercept"]

        self.y = y + additive_regressors + additive_holidays + additive_seasonality

    def finalize_model(self):
        """Finalize the model."""
        self._prepare_fit()
        with self.model:
            pm.Normal(
                "y_%s" % self.name,
                mu=(self.y - self.data["y"].mean()) / self.data["y"].std(),
                sd=self.priors["sigma"],
                observed=(self.data["y"] - self.data["y"].mean())
                / self.data["y"].std(),
            )
            pm.Deterministic("y_hat_%s" % self.name, self.y)

    def _finalize_auto_changepoints(self, plot: bool = False):
        if self.auto_changepoints:
            self.changepoints = []
            self.changepoint_weights = []
            if plot:
                plt.figure(figsize=(20, 10))
            for i in range(self.n_changepoints):
                changepoint_location = self.trace["s"][:, i][self.skip_first :]
                changepoint_weight = self.trace["w"][:, i][self.skip_first :]
                if plot:
                    plt.hist(changepoint_location, alpha=np.median(changepoint_weight))
                changepoint_idx = np.median(changepoint_location)
                self.changepoints.append(self.data.ds[int(changepoint_idx)])
                self.changepoint_weights.append(np.median(changepoint_weight))
            if plot:
                plt.grid()
                plt.title(
                    "Automatic changepoints plausible locations (alpha is weight)"
                )

    def fit(
        self,
        draws: int = 500,
        chains: int = 4,
        trace_size: int = 500,
        method: Sampler = Sampler.NUTS,
        map_initialization: bool = False,
        finalize: bool = True,
        step_kwargs: Dict = None,
        sample_kwargs: Dict = None,
    ):
        """Fit the PMProphet model.

        Parameters
        ----------
        draws : int, > 0
            The number of MCMC samples.
        chains: int, =4
            The number of MCMC draws.
        trace_size: int, =1000
            The last N number of samples to keep in the trace
        method : Sampler
            The sampler of your choice
        map_initialization : bool
            Initialize the model with maximum a posteriori estimates.
        finalize : bool
            Finalize the model.
        step_kwargs : dict
            Additional arguments for the sampling algorithms
            (`NUTS` or `Metropolis`).
        sample_kwargs : dict
            Additional arguments for the PyMC3 `sample` function.
        """

        if sample_kwargs is None:
            sample_kwargs = {}
        if step_kwargs is None:
            step_kwargs = {}
        if chains * draws < trace_size and method != Sampler.ADVI:
            raise Exception(
                "Desired trace size should be smaller than the sampled data points"
            )

        self.skip_first = (chains * draws) - trace_size if method != Sampler.ADVI else 0
        self.chains = chains

        if finalize:
            self.finalize_model()

        with self.model:
            if map_initialization:
                self.start = pm.find_MAP(maxeval=10000)
                if draws == 0:
                    self.trace = {k: np.array([v]) for k, v in self.start.items()}

            if draws:
                if method != Sampler.ADVI:
                    step_method = method.value(**step_kwargs)
                    self.trace = pm.sample(
                        draws,
                        chains=chains,
                        step=step_method,
                        start=self.start if map_initialization else None,
                        **sample_kwargs
                    )
                else:
                    res = pm.fit(
                        draws, start=self.start if map_initialization else None
                    )
                    self.trace = res.sample(trace_size)

    def predict(
        self,
        forecasting_periods: int = 10,
        freq: str = "D",
        extra_data: pd.DataFrame = None,
        include_history: bool = True,
        alpha: float = 0.05,
        plot: bool = False,
    ):
        """Predict using the PMProphet model.

        Parameters
        ----------
        forecasting_periods : int, > 0
            Number of future points to forecast
        freq : string, default: 'D'
        extra_data : pd.DataFrame
        include_history : bool
            If True, predictions are concatenated to the data.
        alpha : float
            Width of the the credible intervals.
        plot : bool
            Plot the predictions.

        Returns
        -------
        A pd.DataFrame with the forecast components.
        """

        if self.auto_changepoints:
            self._finalize_auto_changepoints(plot=False)

        last_date = self.data["ds"].max()
        dates = pd.date_range(
            start=last_date,
            periods=forecasting_periods + 1,  # An extra in case we include start
            freq=freq,
        )
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:forecasting_periods]  # Return correct number of periods

        new_df = pd.DataFrame()

        if include_history:
            new_df["y"] = np.concatenate(
                [self.data["y"], np.zeros(forecasting_periods) * np.nan]
            )
            new_df["ds"] = np.concatenate([self.data["ds"], dates])
        else:
            new_df["y"] = np.zeros(forecasting_periods)
            new_df["ds"] = dates

        for regressor in self.regressors:
            new_df[regressor] = self.data[regressor]

        if extra_data is not None:
            for column in extra_data.columns:
                if column not in ["y", "ds"]:
                    new_df[column] = extra_data[column]

        m = PMProphet(
            data=new_df,
            growth=self.growth,
            intercept=self.intercept,
            model=self.model,
            name=self.name,
        )

        m.changepoints = self.changepoints

        periods = {}
        for column in self.data.columns:
            if column.startswith("f_"):
                period, order = column[2:].split("_")
                periods.setdefault(period, [])
                periods[period].append(int(order))

        for period, orders in periods.items():
            m.add_seasonality(seasonality=float(period), fourier_order=max(orders) + 1)

        m.priors = self.priors
        m.priors_names = self.priors_names
        m.trace = self.trace
        m.multiplicative_data = self.multiplicative_data

        draws = max(
            self.trace[var].shape[-1]
            for var in self.trace.varnames
            if "hat_{}".format(self.name) not in var
        )
        if self.growth:
            # Start with the trend
            y_hat = m._fit_growth(prior=False)
        else:
            y_hat = np.zeros((len(m.data.ds.values), draws))

        multiplicative_seasonality = np.zeros((len(m.data.ds.values), draws))
        additive_seasonality = np.zeros((len(m.data.ds.values), draws))
        if self.seasonality:
            # Add seasonality
            additive_seasonality, multiplicative_seasonality = m._fit_seasonality(
                flatten_components=True
            )

        if self.intercept:
            # Add intercept
            y_hat += self.trace[self.priors_names["intercept"]]

        # Add regressors
        multiplicative_regressors = np.zeros((len(m.data.ds.values), draws))
        additive_regressors = np.zeros((len(m.data.ds.values), draws))
        for idx, regressor in enumerate(self.regressors):
            trace = m.trace[m.priors_names["regressors"]][:, idx]
            if regressor in self.multiplicative_data:
                multiplicative_regressors += trace * np.repeat(
                    [m.data[regressor]], len(trace)
                ).reshape(len(m.data), len(trace))
            else:
                additive_regressors += trace * np.repeat(
                    [m.data[regressor]], len(trace)
                ).reshape(len(m.data), len(trace))

        # Add holidays
        additive_holidays = np.zeros((len(m.data.ds.values), draws))
        multiplicative_holidays = np.zeros((len(m.data.ds.values), draws))
        for idx, holiday in enumerate(self.holidays):
            trace = m.trace[m.priors_names["holidays"]][:, idx]
            if holiday in self.multiplicative_data:
                multiplicative_holidays += trace * np.repeat(
                    [m.data[holiday]], len(trace)
                ).reshape(len(m.data), len(trace))
            else:
                additive_holidays += trace * np.repeat(
                    [m.data[holiday]], len(trace)
                ).reshape(len(m.data), len(trace))

        if (
            np.sum(
                multiplicative_holidays
                + multiplicative_seasonality
                + multiplicative_regressors
            )
            == 0
        ):
            multiplicative_term = 1
        else:
            multiplicative_term = (
                multiplicative_holidays
                + multiplicative_seasonality
                + multiplicative_regressors
            )

        y_hat *= multiplicative_term
        y_hat += additive_seasonality + additive_holidays + additive_regressors

        y_hat_noised = y_hat[:, self.skip_first :] + np.random.normal(
            0, self.trace[self.priors_names["sigma"]][self.skip_first :]
        )

        ddf = pd.DataFrame(
            [
                np.percentile(y_hat_noised, 50, axis=-1),
                np.percentile(y_hat_noised, alpha / 2 * 100, axis=-1),
                np.percentile(y_hat_noised, (1 - alpha / 2) * 100, axis=-1),
            ]
        ).T

        ddf["ds"] = m.data["ds"]
        ddf.columns = ["y_hat", "y_low", "y_high", "ds"]

        if plot:
            plt.figure(figsize=(20, 10))
            ddf.plot("ds", "y_hat", ax=plt.gca())
            ddf["orig_y"] = self.data["y"]
            plt.fill_between(
                ddf["ds"].values,
                ddf["y_low"].values.astype(float),
                ddf["y_high"].values.astype(float),
                alpha=0.3,
            )

            ddf.plot("ds", "orig_y", style="k.", ax=plt.gca(), alpha=0.2)
            for idx, change_point in enumerate(self.changepoints):
                if self.auto_changepoints:
                    plt.axvline(
                        change_point,
                        color="C2",
                        lw=1,
                        ls="dashed",
                        alpha=self.changepoint_weights[idx],
                    )
                else:
                    plt.axvline(change_point, color="C2", lw=1, ls="dashed")
            plt.axvline(
                pd.to_datetime(self.data.ds).max(), color="C3", lw=1, ls="dotted"
            )
            plt.grid(axis="y")
            plt.show()

        return ddf

    def make_trend(self, alpha: float = 0.05):
        """
        Generates the trend component for the model
        :param alpha: float
            Width of the credible intervals.

        Returns
        -------
        A pd.DataFrame with the trend components and confidence intervals.
        """
        fitted_growth = self._fit_growth(prior=False)
        ddf = pd.DataFrame(
            [
                self.data["ds"].astype(str),
                self.data["y"],
                np.mean(fitted_growth, axis=-1),
                np.percentile(fitted_growth, alpha / 2 * 100, axis=-1),
                np.percentile(fitted_growth, (1 - alpha / 2) * 100, axis=-1),
            ]
        ).T

        ddf.columns = ["ds", "y", "y_mid", "y_low", "y_high"]
        ddf.loc[:, "ds"] = pd.to_datetime(ddf["ds"])
        return ddf

    def plot_components(
        self,
        seasonality: bool = True,
        growth: bool = True,
        regressors: bool = True,
        intercept: bool = True,
        changepoints: bool = True,
        holidays: bool = True,
        plt_kwargs: Dict = None,
        alpha: float = 0.05,
    ):
        """Plot the PMProphet forecast components.

        Will plot whichever are available of: trend, holidays, weekly
        seasonality, and yearly seasonality.

        Parameters
        ----------
        seasonality : bool
            Plot seasonality components if feasible.
        growth : bool
            Plot growth component if feasible.
        regressors : bool
            Plot regressors if feasible.
        intercept : bool
            Plot intercept if feasible.
        changepoints : bool
            Plot changepoints if feasible.
        plt_kwargs : dict
            Additional arguments passed to plotting functions.
        alpha : float
            Width of the the credible intervals.
        """
        if plt_kwargs is None:
            plt_kwargs = {"figsize": (20, 10)}

        if self.auto_changepoints:
            self._finalize_auto_changepoints(plot=changepoints)
        if seasonality and self.seasonality:
            self._plot_seasonality(alpha, plt_kwargs)
        if growth and self.growth:
            self._plot_growth(alpha, plt_kwargs)
        if intercept and self.intercept:
            self._plot_intercept(alpha, plt_kwargs)
        if regressors and self.regressors:
            self._plot_regressors(alpha, plt_kwargs)
        if changepoints and len(self.changepoints) and self.growth:
            self._plot_changepoints(alpha, plt_kwargs)
        if holidays and self.holidays:
            self._plot_holidays(alpha, plt_kwargs)

    def _plot_growth(self, alpha: float, plot_kwargs: Dict):
        ddf = self.make_trend(alpha)
        g = self._fit_growth(prior=False)[:, self.skip_first :]
        ddf["growth_mid"] = np.percentile(g, 50, axis=-1)  # np.mean(g, axis=-1)
        ddf["growth_low"] = np.percentile(g, alpha / 2 * 100, axis=-1)
        ddf["growth_high"] = np.percentile(g, (1 - alpha / 2) * 100, axis=-1)
        plt.figure(**plot_kwargs)
        ddf.plot(x="ds", y="growth_mid", ax=plt.gca())
        plt.title("Model Trend")
        plt.fill_between(
            ddf["ds"].values,
            ddf["growth_low"].values.astype(float),
            ddf["growth_high"].values.astype(float),
            alpha=0.3,
        )
        for change_point in self.changepoints:
            plt.axvline(change_point, color="C2", lw=1, ls="dashed")
        plt.grid()
        plt.show()

    def _plot_intercept(self, alpha: float, plot_kwargs: Dict):
        plt.figure(**plot_kwargs)
        pm.forestplot(
            self.trace[self.skip_first // self.chains :],
            varnames=[self.priors_names["intercept"]],
            alpha=alpha,
        )
        plt.show()

    @staticmethod
    def _plot_model(ddf: pd.DataFrame, plot_kwargs: Dict):
        plt.figure(**plot_kwargs)
        ddf.plot(ax=plt.gca(), x="ds", y="y_mid")
        ddf.plot("ds", "y", style="k.", ax=plt.gca())
        plt.fill_between(
            ddf["ds"].values,
            ddf["y_low"].values.astype(float),
            ddf["y_high"].values.astype(float),
            alpha=0.3,
        )
        plt.grid()
        plt.title("Model")
        plt.axes().xaxis.label.set_visible(False)
        plt.legend(["fitted", "observed"])
        plt.show()

    def _plot_regressors(self, alpha: float, plot_kwargs: Dict):
        plt.figure(**plot_kwargs)
        pm.forestplot(
            self.trace[self.skip_first // self.chains :],
            alpha=alpha,
            varnames=[self.priors_names["regressors"]],
            ylabels=self.regressors,
        )
        plt.grid()
        plt.show()

    def _plot_holidays(self, alpha: float, plot_kwargs: dict):
        plt.figure(**plot_kwargs)
        pm.forestplot(
            self.trace[self.skip_first // self.chains :],
            alpha=alpha,
            varnames=[self.priors_names["holidays"]],
            ylabels=self.holidays,
        )
        plt.grid()
        plt.show()

    def _fit_seasonality(self, flatten_components: bool = False):
        periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))
        idx = 0
        additive_ts = np.zeros(
            (
                len(periods),
                len(self.data),
                self.trace[self.priors_names["seasonality"]].shape[0],
            )
        )
        multiplicative_ts = np.zeros(
            (
                len(periods),
                len(self.data),
                self.trace[self.priors_names["seasonality"]].shape[0],
            )
        )

        for pdx, period in enumerate(periods[::-1]):
            cols = [i for i in self.seasonality if float(i.split("_")[1]) == period]
            for col in cols:
                s_trace = self.trace[self.priors_names["seasonality"]][:, idx]
                if col in self.multiplicative_data:
                    multiplicative_ts[pdx] += s_trace * np.repeat(
                        [self.data[col]], len(s_trace)
                    ).reshape(len(self.data), len(s_trace))
                else:
                    additive_ts[pdx] += s_trace * np.repeat(
                        [self.data[col]], len(s_trace)
                    ).reshape(len(self.data), len(s_trace))
                idx += 1
        return (
            additive_ts.sum(axis=0) if flatten_components else additive_ts,
            multiplicative_ts.sum(axis=0) if flatten_components else multiplicative_ts,
        )

    def _plot_changepoints(self, alpha: float, plot_kwargs: Dict):
        plt.figure(**plot_kwargs)
        pm.forestplot(
            self.trace[self.skip_first // self.chains :],
            alpha=alpha,
            varnames=[self.priors_names["changepoints"]],
            ylabels=np.array(self.changepoints).astype(str),
        )
        plt.grid()
        plt.title("Growth Change Points")
        plt.show()

    def _plot_seasonality(self, alpha: float, plot_kwargs: bool):
        # two_tailed_alpha = int(alpha / 2 * 100)
        periods = list(set([float(i.split("_")[1]) for i in self.seasonality]))

        additive_ts, multiplicative_ts = self._fit_seasonality()

        all_seasonalities = [("additive", additive_ts)]
        if len(self.multiplicative_data):
            all_seasonalities.append(("multiplicative", multiplicative_ts))
        for sn, ts in all_seasonalities:
            if (sn == "multiplicative" and np.sum(ts) == 1) or (
                sn == "additive" and np.sum(ts) == 0
            ):
                continue
            ddf = pd.DataFrame(
                np.vstack(
                    [
                        np.percentile(ts[:, :, self.skip_first :], 50, axis=-1),
                        np.percentile(
                            ts[:, :, self.skip_first :], alpha / 2 * 100, axis=-1
                        ),
                        np.percentile(
                            ts[:, :, self.skip_first :], (1 - alpha / 2) * 100, axis=-1
                        ),
                    ]
                ).T,
                columns=[
                    "%s_%s" % (p, l)
                    for l in ["mid", "low", "high"]
                    for p in periods[::-1]
                ],
            )
            ddf.loc[:, "ds"] = self.data["ds"]
            for period in periods:
                if int(period) == 0:
                    step = int(
                        self.data["ds"].diff().mean().total_seconds() // float(period)
                    )
                else:
                    step = int(period)
                graph = ddf.head(step)
                if period == 7:
                    ddf.loc[:, "dow"] = [i for i in ddf["ds"].dt.weekday]
                    graph = (
                        ddf[
                            [
                                "dow",
                                "%s_low" % period,
                                "%s_mid" % period,
                                "%s_high" % period,
                            ]
                        ]
                        .groupby("dow")
                        .mean()
                        .sort_values("dow")
                    )
                    graph.loc[:, "ds"] = [
                        ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]
                        for i in graph.index
                    ]
                    graph = graph.sort_index()
                plt.figure(**plot_kwargs)
                graph.plot(
                    y="%s_mid" % period, x="ds", color="C0", legend=False, ax=plt.gca()
                )
                plt.grid()

                if period == 7:
                    plt.xticks(range(7), graph["ds"].values)
                    plt.fill_between(
                        np.arange(0, 7),
                        graph["%s_low" % period].values.astype(float),
                        graph["%s_high" % period].values.astype(float),
                        alpha=0.3,
                    )
                else:
                    plt.fill_between(
                        graph["ds"].values,
                        graph["%s_low" % period].values.astype(float),
                        graph["%s_high" % period].values.astype(float),
                        alpha=0.3,
                    )

                plt.title("Model Seasonality (%s) for period: %s days" % (sn, period))
                plt.axes().xaxis.label.set_visible(False)
                plt.show()
