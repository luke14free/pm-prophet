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

What's not yet implemented w.r.t. [Facebook Prophet](https://facebook.github.io/prophet/):
* Forecasting
* Multiplicative seasonality
* Holidays (but you can add them as regressors)
* Changepoints in mean/growth
* (potentially other things)

Simple example:
    
    import pandas as pd
    import numpy as np
    from model import PMProphet

    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")

    # Fit both growth and intercept
    m = PMProphet(df, growth=True, intercept=True, name='model')

    # Add yearly seasonality (order: 3)
    m.add_seasonality(seasonality=365.5, order=3)

    # Add monthly seasonality (order: 3)
    m.add_seasonality(seasonality=30, order=3)

    # Add weekly seasonality (order: 3)
    m.add_seasonality(seasonality=7, order=2)

    # Add a white noise regressor
    df['regressor'] = np.random.normal(loc=0, scale=1, size=(len(df)))
    m.add_regressor('regressor')

    # Fit the model (using NUTS, 1000 draws and MAP initialization)
    m.fit(draws=1000, map_initialization=True)
    
    # Plot the fitted components
    m.plot_components()
