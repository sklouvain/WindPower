from powerprediction.experiments.calibration import run_calibration
from powerprediction.experiments.forecast import run_forecast


def compute_region(region_name, subregion_name, run_method_name):
    if run_method_name == "calibration":
        run_calibration(region_name, subregion_name)
    elif run_method_name == "forecast":
        run_forecast(region_name, subregion_name)
    else:
        raise ValueError(""" The run method should be "calibration" or "forecast". """)




