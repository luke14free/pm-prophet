# pm-prophet 
Simplified version of the [Facebook Prophet](https://facebook.github.io/prophet/) model re-implemented in PyMC3

Note that this implementation uses un-standardized data behind the scenes to generate easily understandable parameters (yet this might not be optimal for best fitting purposes).

What's implemented:
* Nowcasting
* Intercept, growth
* Regressors
* Additive seasonality
* Fitting and plotting
* Custom choice of priors (not included in the original prophet)
* Fitting with NUTS/AVDI
* Changepoints in growth (WIP)

What's not yet implemented w.r.t. [Facebook Prophet](https://facebook.github.io/prophet/):
* Forecasting
* Multiplicative seasonality
* Holidays (but you can add them as regressors)
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
m.add_seasonality(seasonality=365.5, order=3)

# Add monthly seasonality (order: 3)
m.add_seasonality(seasonality=30, order=3)

# Add weekly seasonality (order: 3)
m.add_seasonality(seasonality=7, order=2)

# Add a white noise regressor
m.add_regressor('regressor')

# Fit the model (using NUTS, 1000 draws and MAP initialization)
m.fit(draws=10**5, method='AVDI', map_initialization=True)

m.plot_components()
```
