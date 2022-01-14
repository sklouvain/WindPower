import numpy as np
import lightgbm as lgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from powerprediction.utils.data_utils import check_value_list, read_json_from_file, train_val_test_split
from powerprediction.utils.data_utils import write_json_to_file, mape, shuffle_train_val
from powerprediction.utils.data_reader import read_matlab
from powerprediction.utils.cdb_read import read_cdb_curves
from powerprediction.experiments.genomics import genomics


def run_calibration(region_name, subregion_name):
    check_value_list(region_name, "region.json")
    check_value_list(subregion_name, "subregion.json", key_name=region_name)

    root_json = read_json_from_file("datasets_root.json", root_path="..\\store_files\\initial_run_files\\")
    root_name = root_json["root_name"]
    dataset = read_matlab(root_name + region_name + "\\" + region_name + ".mat", subregion_name)
    load_features = ("wind_speed_100m", "temperature")
    x, y = dataset.load_data(window_size=0, features=load_features)
    dates = getattr(dataset, "dates").copy()
    y_prod = getattr(dataset, "production").copy()
    y_cap = getattr(dataset, "capacity").copy()

    # y_cap, y_prod = read_cdb_curves(region=region_name, subregion=subregion_name, dataset=dataset)

    def_file_name = "calibration_" + region_name + "_" + subregion_name + ".json"
    def_json = read_json_from_file(def_file_name, root_path="..\\store_files\\definition_files\\")
    print(def_json)

    best_genes = None

    if def_json["feature_selection_model"] == "genetic":
        genetic_dict = def_json["genetic_parameters"]
        best_genes = genomics(x=x, y=y, nb_of_genes_list=genetic_dict["nb_of_genes_list"],
                              selection_model=genetic_dict["selection_model"],
                              selection_metric=genetic_dict["selection_metric"],
                              population_len=genetic_dict["population_len"],
                              epochs=genetic_dict["epochs"], crossover_rate=genetic_dict["crossover_rate"],
                              mutation_rate=genetic_dict["mutation_rate"],
                              tournament_size=genetic_dict["tournament_size"],
                              seed_value=genetic_dict["seed_value"],
                              activation_model=genetic_dict["activation_model"],
                              activation_metric=genetic_dict["activation_metric"])

    print(best_genes)
    # best_genes = [i for i in range(100)]

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x,
                                                                          y,
                                                                          val_percentage=0.20,
                                                                          test_percentage=0.15)

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    x_train_flat, y_train, x_val_flat, y_val = shuffle_train_val(x_train_flat, y_train, x_val_flat, y_val)
    x_train_genes = x_train_flat[:, best_genes]
    x_val_genes = x_val_flat[:, best_genes]
    x_test_genes = x_test_flat[:, best_genes]

    y_pred = None
    if def_json["model_parameters"]["ml_model_type"] == "fixed":
        if def_json["model_parameters"]["model_name"] == "lgbm_calib":
            lgbm_json = read_json_from_file(def_json["model_parameters"]["model_name"] + ".json",
                                            root_path="..\\store_files\\definition_files\\")

            lgbm_core = lgbm_json["core_parameters"]
            gbm = lgbm.LGBMRegressor(boosting_type=lgbm_core["boosting_type"],
                                     num_leaves=lgbm_core["num_leaves"],
                                     learning_rate=lgbm_core["learning_rate"],
                                     n_estimators=lgbm_core["n_estimators"],
                                     n_jobs=lgbm_core["n_jobs"],
                                     random_state=lgbm_core["random_state"])

            lgbm_fit = lgbm_json["fit_parameters"]
            # print(lgbm_fit["verbose"])
            gbm.fit(x_train_genes, y_train, eval_set=[(x_val_genes, y_val)],
                    eval_metric=lgbm_fit["eval_metric"],
                    early_stopping_rounds=lgbm_fit["early_stopping_rounds"],
                    verbose=lgbm_fit["verbose"])

            y_lgbm_pred = gbm.predict(x_test_genes, num_iteration=gbm.best_iteration_)

            """ Saving the model """
            # should try the relative path
            file_save_name = '{}_{}_lgbm_model.txt'.format(
                region_name, subregion_name)
            save_path_root = root_name + region_name + "\\forecast_files\\"
            # gbm.booster_.save_model('{}_{}_lgbm_model.txt'.format(
            #     region_name, subregion_name
            # ))

            gbm.booster_.save_model(save_path_root + file_save_name)

            points_dict = {"genetic_selected_points": best_genes}
            # write_json_to_file(json_ob=points_dict,
            #                    json_name="{}_{}_points.json".format(
            #                        region_name, subregion_name
            #                    ),
            #                    root_path="..\\store_files\\forecast_files\\")

            write_json_to_file(json_ob=points_dict,
                               json_name="{}_{}_points.json".format(
                                   region_name, subregion_name
                               ),
                               root_path=save_path_root)

            """ Forecasting the Production """
            y_prod_test = y_prod[-len(y_test):]
            y_cap_test = y_cap[-len(y_test):]
            estimated_production = np.multiply(y_lgbm_pred, y_cap_test)

            print("Best Mape Using Lgbm on genetic selected features:", mape(y_test, y_lgbm_pred), "%")
            print("Mape for production:", mape(y_prod_test, estimated_production), "%")
            print()
            print("Best Mse Using Lgbm on genetic selected features:",
                  mean_squared_error(y_test, y_lgbm_pred))
            print("Mse for production:", mean_squared_error(y_prod_test, estimated_production), "MWh")
            print()
            print("Best Mae Using Lgbm on genetic selected features:",
                  mean_absolute_error(y_test, y_lgbm_pred))
            print("Mse for production:", mean_absolute_error(y_prod_test, estimated_production), "MWh")
            print()
