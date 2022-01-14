import numpy as np
import lightgbm as lgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from powerprediction.utils.data_utils import check_value_list, read_json_from_file, train_val_test_split
from powerprediction.utils.data_utils import write_json_to_file, mape, shuffle_train_val
from powerprediction.utils.data_utils import parse_date_from_file_name
from powerprediction.utils.data_reader import read_matlab
from powerprediction.utils.cdb_read import read_cdb_curves

from powerprediction.utils.data_utils import check_value_list
from powerprediction.utils.data_reader import read_matlab


def run_forecast(region_name, subregion_name):
    check_value_list(region_name, "region.json")
    check_value_list(subregion_name, "subregion.json", key_name=region_name)

    root_json = read_json_from_file("datasets_root.json", root_path="..\\store_files\\initial_run_files\\")
    root_name = root_json["root_name"]
    forecast_file_root = root_name + region_name + "\\forecast_data\\"
    forecast_file_json = read_json_from_file("forecast_read.json",
                                             root_path="..\\store_files\\initial_run_files\\")
    forecast_file_name = forecast_file_json[region_name]

    dataset = read_matlab(forecast_file_root + forecast_file_name, subregion_name)
    load_features = ("wind_speed_100m", "temperature")
    x_forecast, _ = dataset.load_data(window_size=0, features=load_features)
    # dates = getattr(dataset, "dates").copy()
    print(x_forecast.shape[0])
    year, month, day, hour = parse_date_from_file_name(forecast_file_name)

    y_cap, y_prod = read_cdb_curves(region=region_name, subregion=subregion_name,
                                    date=(year, month, day, hour),
                                    hours_diff=x_forecast.shape[0])

    x_forecast_flat = x_forecast.reshape(x_forecast.shape[0], -1)

    points_root = root_name + region_name + "\\forecast_files\\"
    points_file_name = "{}_{}_points.json".format(region_name, subregion_name)
    best_genes_json = read_json_from_file(points_file_name, root_path=points_root)
    best_genes = best_genes_json["genetic_selected_points"]

    x_forecast_genes = x_forecast_flat[:, best_genes]

    """ Loading the model """
    print("Loading the model..")
    file_save_name = '{}_{}_lgbm_model.txt'.format(region_name, subregion_name)
    lgbm_model = lgbm.Booster(model_file=points_root + file_save_name)
    print("Number of trees for the saved model:", lgbm_model.num_trees())

    y_pred = lgbm_model.predict(x_forecast_genes)
    print("Length of forcasted samples:", len(y_pred))

    print("Final estimated production values:")
    y_prod_estimated = y_pred * y_cap
    print(y_prod_estimated)
    print("MAPE:", mape(y_prod_estimated, y_prod))
    print(np.shape(y_prod_estimated)[0])
