__all__ = ["BatterySOHForecastDataset", "BatterySOHForecaster"]


def __getattr__(name):
    if name == "BatterySOHForecastDataset":
        from forecasting.data import BatterySOHForecastDataset

        return BatterySOHForecastDataset
    if name == "BatterySOHForecaster":
        from forecasting.model import BatterySOHForecaster

        return BatterySOHForecaster
    raise AttributeError(name)
