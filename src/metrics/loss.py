import pandas as pd
import numpy as np


def custom_novartis_metric(actuals, forecast, avg_volume):
    """
    This function aims to compute the Custom Accuracy Metric
    for the Novartis Datathon, 3rd edition.

    Given the actuals followed by the forecast and the avg_volume
    of the brand, it will compute the metric score.

    Keyword parameters:
        actuals (float vector): Real value of Y
        forecast (float vector): Volume forecast
        avg_volume (float): Average monthly volume of the 12 months
                            prior to the generic entry.

    Returns:
        custom_metric: Uncertainty Metric score (%)
    """

    # Compute the first part of the equation
    # (custom MAPE with Average volume)
    custom_mape = sum(abs(actuals - forecast)) / (24 * avg_volume)

    # Compute the second part of the equation
    # (custom 6-first-months MAPE with Average volume)
    six_month_mape = \
        abs(sum(actuals[:6]) - sum(forecast[:6])) / (6 * avg_volume)

    # Compute the third part of the equation
    # (custom 6-months MAPE with Average volume)
    twelve_month_mape = \
        abs(sum(actuals[6:12]) - sum(forecast[6:12])) / (6 * avg_volume)

    # Compute the fourth part of the equation
    # (custom 12-months MAPE with Average volume)
    last_month_mape = \
        abs(sum(actuals[12:]) - sum(forecast[12:])) / (12 * avg_volume)

    # Compute the custom metric
    custom_metric = 0.5 * custom_mape + 0.3 * six_month_mape + \
                    0.1 * (twelve_month_mape + last_month_mape)

    return custom_metric * 100


def uncertainty_novartis_metric(actuals, upper_bound, lower_bound, avg_volume):
    """
    This function aims to compute the Uncertainty Metric for the
    Novartis Datathon, 3rd edition.

    Given the actuals followed by the upper_bound and lower_bound intervals and the
    average volume, it will compute the metric score.

    Keyword parameters:
        actuals (float vector): Real value of Y
        upper_bound (float vector): upper_bound forecast interval (percentile 95)
        lower_bound (float vector): lower_bound forecast interval (percentile 5)
        avg_volume (float): Average monthly volume of the 12 months
                            prior to the generic entry.

    Returns:
        error_metric: Uncertainty Metric score (%)
    """
    # Assert that all the sizes are OK
    assert (len(lower_bound) == len(upper_bound)) == (len(actuals) == 24), \
        "We should have 24 sorted actuals, upper_bound and lower_bound intervals"

    uncertainty_first6 = (
                             # Wide intervals are penalized
                                 0.85 * sum(abs(upper_bound[:6] - lower_bound[:6])) +
                                 0.15 * 2 / 0.05 * (
                                     # If actuals are outside of the intervals, it adds error
                                         sum((lower_bound[:6] - actuals[:6]) * (actuals[:6] < lower_bound[:6])) +
                                         sum((actuals[:6] - upper_bound[:6]) * (actuals[:6] > upper_bound[:6]))
                                 )
                         ) / (6 * avg_volume) * 100

    uncertainty_last18 = (
                                 0.85 * sum(abs(upper_bound[6:] - lower_bound[6:])) +
                                 0.15 * 2 / 0.05 * (
                                         sum((lower_bound[6:] - actuals[6:]) * (actuals[6:] < lower_bound[6:])) +
                                         sum((actuals[6:] - upper_bound[6:]) * (actuals[6:] > upper_bound[6:]))
                                 )
                         ) / (18 * avg_volume) * 100

    return (0.6 * uncertainty_first6 + 0.4 * uncertainty_last18)


def custom_metric(actuals, forecast, max_volume, avg_volume):
    # Scale normalized predictions
    actuals = actuals * max_volume / avg_volume
    forecast = forecast * max_volume / avg_volume

    # Compute the first part of the equation
    # (custom MAPE with Average volume)
    custom_mape = (actuals - forecast).abs().sum() / 24

    # Compute the second part of the equation
    # (custom 6-first-months MAPE with Average volume)
    six_month_mape = (actuals[:6].sum() - forecast[:6].sum()).abs() / 6

    # Compute the third part of the equation
    # (custom 6-months MAPE with Average volume)
    twelve_month_mape = (actuals[6:12].sum() - forecast[6:12].sum()).abs() / 6

    # Compute the fourth part of the equation
    # (custom 12-months MAPE with Average volume)
    last_month_mape = (actuals[12:].sum() - forecast[12:].sum()).abs() / 12

    # Compute the custom metric
    custom_metric = (0.5 * custom_mape
                     + 0.3 * six_month_mape
                     + 0.1 * (twelve_month_mape + last_month_mape))

    return custom_metric


def uncertainty_metric(actuals, upper_bound, lower_bound, max_volume, avg_volume):
    # Assert that all the sizes are OK
    actuals = (actuals * max_volume / avg_volume)
    upper_bound = (upper_bound * max_volume / avg_volume)
    lower_bound = (lower_bound * max_volume / avg_volume)

    uncertainty_first6 = (
        # Wide intervals are penalized
        0.85 * ((upper_bound[:6] - lower_bound[:6]).abs()).sum() +
        0.15 * 2 / 0.05 * (
            # If actuals are outside of the intervals, it adds error
            ((lower_bound[:6] - actuals[:6]) * (actuals[:6] < lower_bound[:6])).sum() +
            ((actuals[:6] - upper_bound[:6]) * (actuals[:6] > upper_bound[:6])).sum()
        )
     ) / 6

    uncertainty_last18 = (
        0.85 * ((upper_bound[6:] - lower_bound[6:]).abs()).sum()
             + 0.15 * 2 / 0.05 * (
                ((lower_bound[6:] - actuals[6:]) * (actuals[6:] < lower_bound[6:])).sum()
                + ((actuals[6:] - upper_bound[6:]) * (actuals[6:] > upper_bound[6:])).sum()
                                 )
                         ) / 18

    return 0.6 * uncertainty_first6 + 0.4 * uncertainty_last18
