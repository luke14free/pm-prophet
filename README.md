# pm-prophet 
Simplified version of the [Facebook Prophet](https://facebook.github.io/prophet/) model re-implemented in PyMC3

What's implemented:
* Nowcasting
* Intercept, growth
* Regressors
* Holidays
* Additive seasonality
* Fitting and plotting
* Custom choice of priors (not included in the original prophet)
* Fitting with NUTS/AVDI/Metropolis
* Changepoints in growth

What's not yet implemented w.r.t. [Facebook Prophet](https://facebook.github.io/prophet/):
* Forecasting
* Multiplicative seasonality
* Saturating growth
* (potentially other things)

Simple example:
    
```python
import pandas as pd
import numpy as np
from model import PMProphet

df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
df['regressor'] = np.random.normal(loc=0, scale=1, size=(len(df)))

# Fit both growth and intercept
m = PMProphet(df, growth=True, intercept=True, n_change_points=20, name='model')

# Add yearly seasonality (order: 3)
m.add_seasonality(seasonality=365.5, order=2)

# Add monthly seasonality (order: 3)
m.add_seasonality(seasonality=30, order=2)

# Add a white noise regressor
m.add_regressor('regressor')

# Fit the model (using NUTS, 1000 draws and MAP initialization)
m.fit(
    draws=10**4, 
    method='Metropolis', # you can also try NUTS or AVDI
    sample_kwargs={'chains':1, 'njobs':1}, # NOTE: you should use more than 1 chain
    map_initialization=False
)

m.plot_components()
```

![Model](https://github.com/luke14free/pm-prophet/blob/master/examples/images/download.png)
![S1](https://github.com/luke14free/pm-prophet/blob/master/examples/images/download-1.png)
![S2](https://github.com/luke14free/pm-prophet/blob/master/examples/images/download-2.png)

## BYOP - Bring Your Own Priors

The default priors are:

Variable | Prior | Parameters
--- | --- | --- 
`regressors` | Normal | loc:0, scale:10 
`holidays` | Laplace | loc:0, scale:10 
`seasonality` | Laplace | loc:0, scale:1 
`growth` | Laplace | loc:0, scale:10 
`changepoints` | Laplace | loc:0, scale:10 
`intercept` | Flat Prior | - 
`sigma` | Half Cauchy | loc:100

But you can change model priors by inspecting and modifying the distribution in

```python
m.priors
```

which is a dictionary of {prior: pymc3-distribution}.
