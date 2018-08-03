# pm-prophet 
Simplified version of the Facebook's Prophet model re-implemented in PyMC3

Note that this implementation uses un-standardized data behind the scenes to generate easily understandable parameters (yet this might not be optimal for best fitting purposes).

What's implemented:
* Nowcasting
* Intercept, growth
* Regressors
* Additive seasonality
* Fitting and plotting
* Custom choice of priors (not included in the original prophet)

What's not implemented w.r.t. Facebook's Prophet:
* Forecasting
* Multiplicative seasonality
* Holidays (but you can add them as regressors)
* (potentially other things)

Simple example:
    
    import pandas as pd
    from model import PMProphet
    
    df = pd.read_csv("examples/example_wp_log_peyton_manning.csv")
    m = PMProphet(df, intercept=True, growth=True, name='model')
    m.add_seasonality(365.25, 2)
    m.add_seasonality(7, 2)
    m.fit(draws=100)
    m.plot_components()