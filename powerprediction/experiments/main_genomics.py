import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import joblib
from sklearn.model_selection import cross_validate

from powerprediction.utils.data_reader import basic_argparser, read_matlab
from powerprediction.utils.data_utils import train_val_test_split, get_coordinates_from_arrays, get_font_dict
from powerprediction.utils.data_utils import mape, shuffle_train_val, plot_target,  compute_variance_from_data
from powerprediction.utils.data_utils import plot_predictions, plot_map, convert_multidimensional_map_points
from powerprediction.utils.data_utils import read_json_from_file, get_points_list_from_json
from powerprediction.experiments.genomics import genomics


@dataclass
class InitArgs:
    filename: str = ""
    dataset: str = ""


def main(args):
    dataset = read_matlab(args.filename, args.dataset)
    load_features = ("wind_speed_100m", "temperature")
    # load_features = ("wind_speed_100m",)
    x, y = dataset.load_data(window_size=0, features=load_features)
    dates = getattr(dataset, "dates").copy()
    y_prod = getattr(dataset, "production").copy()
    y_cap = getattr(dataset, "capacity").copy()

    lat = getattr(dataset, "lat").copy()
    lon = getattr(dataset, "lon").copy()
    lat, lon = get_coordinates_from_arrays(lat, lon)
    print(np.shape(lat), lat)
    print(np.shape(lon), lon)

    """ Initializing the parameters"""
    nb_of_genes_list = [20]
    selection_model = "Lgbm"
    selection_metric = "mse"
    activation_model = "Lgbm"
    activation_metric = "mse"

    population_len = 300
    epochs = 10
    crossover_rate = 0.8
    mutation_rate = 0.005
    tournament_size = 51
    seed_value = 17

    # Uncomment to run genomics() and comment the next best_genes = [...]
    # best_genes = genomics(x=x, y=y, nb_of_genes_list=nb_of_genes_list,
    #                       selection_model=selection_model, selection_metric=selection_metric,
    #                       population_len=population_len,  epochs=epochs,
    #                       crossover_rate=crossover_rate, mutation_rate=mutation_rate,
    #                       tournament_size=tournament_size, seed_value=seed_value,
    #                       activation_model=activation_model, activation_metric=activation_metric)

    json_name = args.filename.split("\\")[-1].replace(".mat", "") + "_" + args.dataset + ".json"
    json_ob = read_json_from_file(json_name)
    best_genes = json_ob["px50_lgbm_mse"]
    # best_genes = [0]

    print("Nb of pixels:", len(best_genes))
    print("Best pixels:", best_genes)

    """ Comparation with basic Pearson Correlation """

    nb_of_genes = len(best_genes)
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x,
                                                                          y,
                                                                          val_percentage=0.20,
                                                                          test_percentage=0.15)

    print("Start training", dates[0])
    print("End Training", dates[-(len(y_test) + 1)])
    print("Start Testing", dates[-len(y_test)])
    print("End Testing", dates[-1])

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    x_train_flat, y_train, x_val_flat, y_val = shuffle_train_val(x_train_flat, y_train, x_val_flat, y_val)

    fs = SelectKBest(score_func=f_regression, k=nb_of_genes)
    x_train_flat_ps = fs.fit_transform(x_train_flat, y_train)
    x_val_flat_ps = fs.fit_transform(x_val_flat, y_val)
    x_test_flat_ps = fs.transform(x_test_flat)

    gbm = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=50, learning_rate=0.07, n_estimators=150, n_jobs=-1)
    gbm.fit(x_train_flat_ps, y_train, eval_set=[(x_val_flat_ps, y_val)], eval_metric="l2", early_stopping_rounds=7,
            verbose=False)
    y_pred = gbm.predict(x_test_flat_ps, num_iteration=gbm.best_iteration_)

    print("\n\nLgbm MAPE using the {} features extracted with Pearson Correlation:".format(nb_of_genes),
          mape(y_test, y_pred), "%")
    print("\n\nLgbm MSE using the {} features extracted with Pearson Correlation:".format(nb_of_genes),
          mean_squared_error(y_test, y_pred))

    """ Reading the wind farms from json """
    region = args.filename.split("_")[-1].split(".")[0].lower()
    print(region)
    file_name = region + '_farms.json'
    # json_ob = read_json_from_file(file_name=file_name, root_path="..\\store_files\\wind_farms\\")
    # print(json_ob)
    # farms_list = get_points_list_from_json(json_ob)
    # for name, latitude, longitude in farms_list:
    #     print("Name:", name)
    #     print("Latitude:", latitude)
    #     print("Longitude:", longitude)
    #     print()
    farms_list = None

    """ Plot best map pixels """
    x_shape = np.shape(x_train)
    print(x_shape)

    map_points = best_genes
    points_dict = {}

    if len(load_features) == 1:
        points_dict[load_features[0]] = map_points
    else:
        wind_points, temp_points = convert_multidimensional_map_points(map_points)
        points_dict[load_features[0]] = wind_points
        points_dict[load_features[1]] = temp_points

    """ Plot points on train samples """
    for feature_ind in range(x_shape[3]):
        for i in range(5):
            sample_obj = x_train[i, :, :, feature_ind]
            plot_map(sample_x=sample_obj,
                     cmap='hot',
                     interpolation='nearest',
                     title='Train Data X_{}_{}'.format(i, load_features[feature_ind]),
                     map_points=points_dict[load_features[feature_ind]],
                     lat=lat,
                     lon=lon,
                     farms=farms_list)

    """ Plot points on variance maps """
    for feature_ind in range(x_shape[3]):
        var_matrix = compute_variance_from_data(x[:, :, :, feature_ind])
        str_name = load_features[feature_ind]
        plot_map(sample_x=var_matrix,
                 cmap='RdPu',
                 interpolation='none',
                 title='Historical Variance: {}'.format(str_name),
                 map_points=points_dict[load_features[feature_ind]],
                 lat=lat,
                 lon=lon,
                 farms=farms_list)

    """ Plot genetic points and wind farm """
    for feature_ind in range(len(load_features)):
        mat = np.ones((x.shape[1], x.shape[2]))
        str_name = load_features[feature_ind]
        plot_map(sample_x=mat,
                 cmap='summer',
                 interpolation='none',
                 title='Selected points vs Wind Farms: {}'.format(str_name),
                 map_points=points_dict[load_features[feature_ind]],
                 lat=lat,
                 lon=lon,
                 farms=farms_list,
                 min_coeff=1,
                 max_coeff=1,
                 mid_coeff=0.67)

    """ Plot predictions """
    x_train_genes = x_train_flat[:, map_points]
    x_val_genes = x_val_flat[:, map_points]
    x_test_genes = x_test_flat[:, map_points]
    # x_train_genes = x_train_flat
    # x_val_genes = x_val_flat
    # x_test_genes = x_test_flat

    x_cv = np.concatenate([x_train_genes, x_val_genes])
    y_cv = np.concatenate([y_train, y_val])

    boosting_types = ['gbdt', 'goss']
    num_leaves_list = [30, 50, 70, 90, 110]
    learning_rates = [0.05, 0.07, 1.0, 1.3]
    n_estimators_list = [150]

    best_boosting_type = None
    best_num_leaves = None
    best_learning_rate = None
    best_n_estimators = None
    best_score_mean = 1000000

    # Uncomment to do grid search and uncomment the code after """ Grid Search Obtained Model """
    # for boosting_type in boosting_types:
    #     for num_leaves in num_leaves_list:
    #         for learning_rate in learning_rates:
    #             for n_estimators in n_estimators_list:
    #                 model = lgb.LGBMRegressor(boosting_type=boosting_type,
    #                                           num_leaves=num_leaves,
    #                                           learning_rate=learning_rate,
    #                                           n_estimators=n_estimators,
    #                                           n_jobs=-1)
    #                 scores = cross_validate(model, x_cv, y_cv, scoring=('neg_mean_absolute_percentage_error',
    #                                                                     'neg_mean_squared_error',
    #                                                                     'neg_mean_absolute_error')
    #                                         )
    #                 for key in scores:
    #                     scores[key] *= -1
    #
    #                 print("Model parameters:")
    #                 print(model.get_params())
    #                 print("Boosting Type = {}, Num leaves = {}, Learning Rate = {}, N Estimators = {}".format(
    #                     boosting_type,
    #                     num_leaves,
    #                     learning_rate,
    #                     n_estimators
    #                 ))
    #                 print("test_neg_mean_squared_error:", scores['test_neg_mean_squared_error'])
    #                 print("MSE Mean:", scores['test_neg_mean_squared_error'].mean())
    #                 print("test_neg_mean_absolute_percentage_error:", scores['test_neg_mean_absolute_percentage_error'])
    #                 print("MAPE Mean:", scores['test_neg_mean_absolute_percentage_error'].mean())
    #                 print("test_neg_mean_absolute_error:", scores['test_neg_mean_absolute_error'])
    #                 print("MAE Mean:", scores['test_neg_mean_absolute_error'].mean())
    #
    #                 mse_mean = scores['test_neg_mean_squared_error'].mean()
    #                 mae_mean = scores['test_neg_mean_squared_error'].mean()
    #                 score_mean = (mae_mean * mae_mean + mse_mean) / 2
    #                 print("MAE MSE Score =", score_mean)
    #                 print()
    #
    #                 if score_mean < best_score_mean:
    #                     best_score_mean = score_mean
    #                     best_boosting_type = boosting_type
    #                     best_num_leaves = num_leaves
    #                     best_learning_rate = learning_rate
    #                     best_n_estimators = n_estimators

    """ Already tuned lgbm model """
    gbm = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=50, learning_rate=0.07, n_estimators=150, n_jobs=-1,
                            random_state=17)

    """ Grid Search Obtained Model """
    # gbm = lgb.LGBMRegressor(boosting_type=best_boosting_type,
    #                         num_leaves=best_num_leaves,
    #                         learning_rate=best_learning_rate,
    #                         n_estimators=best_n_estimators,
    #                         n_jobs=-1,
    #                         random_state=17)

    print("Best Model:")
    print(gbm.get_params())

    gbm.fit(x_train_genes,
            y_train,
            eval_set=[(x_val_genes, y_val)],
            eval_metric="l2",
            early_stopping_rounds=7,
            verbose=False)
    y_lgbm_pred = gbm.predict(x_test_genes, num_iteration=gbm.best_iteration_)

    """ Saving the model """
    gbm.booster_.save_model('lgbm_model.txt')

    """ Forecasting the Production """
    y_prod_test = y_prod[-len(y_test):]
    y_cap_test = y_cap[-len(y_test):]
    estimated_production = np.multiply(y_lgbm_pred, y_cap_test)

    print("Best Mape Using Lgbm on genetic selected features:", mape(y_test, y_lgbm_pred), "%")
    print("Mape for production:", mape(y_prod_test, estimated_production), "%")
    print()
    print("Best Mse Using Lgbm on genetic selected features:", mean_squared_error(y_test, y_lgbm_pred))
    print("Mse for production:", mean_squared_error(y_prod_test, estimated_production), "MWh")
    print()
    print("Best Mae Using Lgbm on genetic selected features:", mean_absolute_error(y_test, y_lgbm_pred))
    print("Mse for production:", mean_absolute_error(y_prod_test, estimated_production), "MWh")
    print()


    font = get_font_dict()
    plt.title('Actual vs Predicted Target Values', fontdict=font)
    plt.xlabel("Dates", fontdict=font)
    plt.ylabel("Production / Capacity Ratio", fontdict=font)
    plt.plot(dates[-len(y_test):], y_test, 'co', linestyle='-', label='actual')
    plt.plot(dates[-len(y_test):], y_lgbm_pred, 'ro', linestyle='-', label='predicted')
    plt.plot(dates[-len(y_test):], abs(y_test - y_lgbm_pred) / y_test, 'yo', linestyle='-', label='% error')
    plt.legend()
    plt.show()

    """
    FRA - Best genes 200 px Ridge MSE
    [376, 561, 631, 983, 1112, 1231, 1343, 1546, 1728, 1889, 2043, 2061, 2137, 2197, 2214, 2363, 2872, 2897,
     3064, 3101, 3402, 3565, 3751, 3855, 3970, 3978, 4052, 4103, 4204, 4412, 4483, 4537, 4550, 4567, 4629, 
     4901, 4914, 4928, 4940, 5103, 5110, 5265, 5272, 5398, 5411, 5532, 5586, 5649, 5764, 5786, 5805, 5806, 
     5812, 6025, 6067, 6080, 6180, 6219, 6413, 6461, 6536, 6595, 6757, 6915, 6946, 7220, 7383, 7470, 7512, 
     7591, 7595, 7634, 7706, 7768, 7780, 8113, 8215, 8246, 8280, 8357, 8591, 8621, 8663, 8742, 8781, 8816,
     8874, 8936, 8976, 9011, 9024, 9196, 9218, 9225, 9279, 9304, 9318, 9425, 9529, 9531, 9538, 9546, 9552, 
     9559, 9633, 9718, 9720, 9731, 9763, 9769, 9807, 9809, 9864, 9886, 10022, 10037, 10079, 10091, 10103, 
     10167, 10216, 10334, 10407, 10416, 10417, 10457, 10467, 10490, 10599, 10681, 10779, 11091, 11254, 11439, 
     11518, 11524, 11559, 11651, 11785, 11924, 11977, 11986, 12070, 12111, 12183, 12237, 12518, 12562, 12592, 
     12600, 12708, 12934, 12985, 13002, 13060, 13079, 13114, 13411, 13451, 13608, 13675, 13754, 13786, 13795, 
     13806, 13894, 13901, 14070, 14152, 14178, 14186, 14293, 14372, 14457, 14492, 14675, 14828, 14891, 14992, 
     15046, 15050, 15143, 15282, 15376, 15402, 15408, 15432, 15648, 15719, 15759, 15780, 15859, 16008, 16044, 
     16100, 16342, 16431, 16615, 16643, 16644]
    """

    """
    FRA - Best genes 50 px Lgbm MSE
    [1506, 2312, 2636, 2969, 3127, 3280, 3496, 3626, 3863, 4896, 5528, 5976, 6132, 6448, 7411, 8110, 8346,
     8634, 9732, 9885, 9903, 10022, 10085, 10092, 10260, 10354, 11506, 12004, 12316, 13590, 13643, 13644,
     13792, 13923, 14046, 14164, 14197, 14253, 14265, 14441, 14608, 14681, 14807, 15092, 15696, 16019, 16106,
     16519, 16607, 16989]
    """

    """
    FRA - Best genes 100 px Lgbm MSE
    [193, 515, 528, 604, 1727, 1777, 2129, 2256, 2981, 3058, 3487, 3582, 3763, 3886, 3914, 4072, 4531, 4549,
     4727, 4938, 5002, 5176, 5344, 5424, 5457, 5893, 5912, 5930, 5953, 6068, 6160, 6198, 6394, 6577, 6658,
     6716, 6730, 6731, 7110, 7189, 7192, 7516, 7677, 7692, 7720, 7740, 8085, 8207, 8237, 8273, 8278, 8298,
     8392, 8574, 8714, 8816, 8900, 9234, 9344, 9816, 9959, 10356, 10574, 10576, 10692, 10823, 10901, 11012,
     11203, 11419, 11966, 12541, 12623, 12646, 12828, 13125, 13257, 13382, 13445, 13673, 13730, 13780,
     13824, 14321, 14366, 15077, 15094, 15501, 15898, 15961, 15993, 16222, 16232, 16339, 16356, 16603,
     16730, 16762, 16783, 16919]
    """

    """
    FRA - Best genes 300 px Ridge MSE
    [85, 110, 161, 186, 251, 396, 460, 462, 466, 537, 559, 808, 858, 866, 889, 1001, 1117, 1140, 1179, 1183, 1212, 1573, 1666, 1786, 1787, 1821, 1895, 1987, 2021, 2028, 2324, 2330, 2370, 2389, 2445, 2482, 2551, 2581, 2608, 2642, 2671, 2707, 2756, 2770, 2780, 2799, 2824, 2832, 2910, 3015, 3023, 3094, 3103, 3192, 3201, 3274, 3282, 3307, 3387, 3435, 3520, 3597, 3603, 3630, 3651, 3680, 3685, 3738, 3794, 3867, 3964, 3970, 4050, 4200, 4356, 4492, 4507, 4535, 4740, 4926, 5046, 5096, 5159, 5213, 5268, 5285, 5301, 5329, 5614, 5619, 5636, 5726, 5851, 5890, 5962, 6008, 6022, 6107, 6117, 6124, 6210, 6276, 6329, 6337, 6365, 6371, 6373, 6417, 6516, 6559, 6620, 6702, 6706, 6721, 6738, 6777, 6801, 6827, 6947, 6979, 7117, 7140, 7254, 7264, 7325, 7446, 7485, 7524, 7579, 7608, 7619, 7722, 7805, 7812, 7927, 7932, 7965, 7970, 8003, 8007, 8046, 8101, 8117, 8118, 8123, 8195, 8246, 8254, 8396, 8500, 8635, 8808, 8894, 8914, 8921, 9053, 9058, 9143, 9196, 9207, 9208, 9315, 9353, 9394, 9400, 9402, 9403, 9460, 9482, 9625, 9630, 9631, 9633, 9738, 9836, 9837, 9851, 9858, 9901, 10043, 10085, 10102, 10200, 10221, 10321, 10359, 10416, 10437, 10472, 10527, 10550, 10603, 10707, 10784, 10820, 10852, 10987, 10998, 11036, 11042, 11291, 11303, 11327, 11358, 11406, 11463, 11520, 11550, 11570, 11630, 11742, 11749, 11783, 11785, 11925, 11991, 12056, 12058, 12097, 12098, 12119, 12241, 12318, 12360, 12438, 12521, 12560, 12570, 12678, 12804, 12850, 12857, 12959, 13018, 13038, 13143, 13152, 13212, 13217, 13276, 13310, 13386, 13391, 13406, 13419, 13582, 13633, 13661, 13686, 13740, 13776, 13791, 13831, 13962, 14029, 14030, 14049, 14099, 14154, 14174, 14261, 14270, 14308, 14533, 14624, 14647, 14765, 14784, 14821, 14878, 14889, 15011, 15120, 15141, 15167, 15242, 15243, 15286, 15403, 15410, 15513, 15661, 15666, 15694, 15767, 15774, 15814, 16007, 16139, 16181, 16287, 16334, 16349, 16659, 16754, 16859, 16867, 16928, 16947, 16953]
    """

    """
    FRA - Best genes 400 px Ridge MSE
    Best pixels: [90, 98, 468, 543, 551, 566, 607, 627, 634, 663, 744, 808, 824, 829, 864, 977, 986, 1014, 1032, 1091, 1148, 1260, 1302, 1330, 1427, 1477, 1571, 1580, 1594, 1677, 1720, 1811, 1843, 1844, 1861, 1903, 1927, 1935, 2011, 2095, 2098, 2217, 2263, 2395, 2427, 2433, 2452, 2461, 2474, 2482, 2625, 2633, 2660, 2755, 2870, 2874, 2914, 2926, 2975, 3004, 3058, 3066, 3094, 3095, 3112, 3118, 3133, 3143, 3168, 3193, 3227, 3252, 3278, 3290, 3303, 3335, 3451, 3490, 3565, 3597, 3623, 3627, 3703, 3786, 3859, 3915, 3968, 4046, 4126, 4127, 4188, 4222, 4241, 4277, 4293, 4301, 4345, 4374, 4433, 4438, 4451, 4681, 4692, 4715, 4721, 4789, 4804, 4854, 4914, 4932, 4937, 4950, 5030, 5034, 5084, 5118, 5181, 5215, 5262, 5272, 5296, 5314, 5321, 5344, 5358, 5375, 5434, 5459, 5586, 5589, 5597, 5629, 5707, 5731, 5791, 5918, 5969, 6009, 6020, 6073, 6078, 6084, 6087, 6099, 6100, 6166, 6196, 6206, 6212, 6233, 6309, 6331, 6379, 6449, 6491, 6505, 6547, 6558, 6587, 6637, 6649, 6706, 6737, 6810, 6822, 6834, 6849, 6853, 6879, 6949, 6995, 7018, 7036, 7045, 7066, 7083, 7141, 7221, 7250, 7286, 7313, 7368, 7417, 7472, 7555, 7566, 7590, 7598, 7610, 7631, 7638, 7724, 7757, 7807, 7879, 7929, 7932, 7993, 8016, 8114, 8144, 8208, 8270, 8293, 8331, 8369, 8417, 8464, 8492, 8598, 8717, 8743, 8811, 8817, 8847, 8856, 8861, 8899, 8938, 8981, 8990, 9051, 9136, 9148, 9177, 9181, 9257, 9320, 9328, 9376, 9409, 9427, 9455, 9458, 9528, 9575, 9600, 9726, 9763, 9792, 9812, 9814, 9885, 9902, 9948, 10006, 10071, 10080, 10116, 10172, 10190, 10199, 10204, 10205, 10237, 10242, 10256, 10268, 10319, 10344, 10478, 10496, 10544, 10600, 10617, 10822, 10886, 11091, 11102, 11126, 11174, 11213, 11271, 11337, 11391, 11517, 11551, 11595, 11600, 11610, 11633, 11636, 11662, 11871, 11938, 11961, 11989, 12017, 12023, 12037, 12081, 12107, 12187, 12221, 12229, 12246, 12288, 12296, 12339, 12416, 12463, 12471, 12481, 12490, 12516, 12525, 12628, 12717, 12814, 12835, 12892, 12900, 12923, 12929, 12951, 12973, 13002, 13096, 13144, 13158, 13178, 13193, 13224, 13266, 13280, 13291, 13324, 13375, 13508, 13541, 13580, 13589, 13607, 13679, 13683, 13732, 13854, 13858, 13875, 13963, 13970, 13999, 14028, 14070, 14136, 14171, 14195, 14215, 14266, 14298, 14319, 14360, 14397, 14497, 14514, 14597, 14619, 14626, 14646, 14685, 14761, 14904, 14940, 14961, 15136, 15192, 15486, 15502, 15525, 15551, 15553, 15651, 15688, 15748, 15822, 15856, 15878, 15912, 16006, 16036, 16059, 16069, 16118, 16122, 16146, 16175, 16186, 16252, 16344, 16369, 16421, 16541, 16672, 16673, 16748, 16781, 16817, 16820, 16854, 16954]
    """

    """
    FRA - Best genes 500 px Ridge MSE
    Best Chromosome with 500 features: [5, 28, 83, 103, 110, 143, 151, 215, 328, 362, 378, 458, 640, 695, 764, 806, 821, 886, 913, 948, 964, 985, 1011, 1109, 1126, 1155, 1161, 1219, 1238, 1262, 1320, 1356, 1476, 1519, 1595, 1623, 1735, 1753, 1767, 1772, 1799, 1907, 1970, 2048, 2058, 2082, 2244, 2257, 2268, 2285, 2339, 2345, 2348, 2373, 2450, 2504, 2568, 2571, 2625, 2640, 2658, 2675, 2691, 2735, 2764, 2788, 2810, 2860, 2946, 2960, 3011, 3014, 3018, 3037, 3072, 3115, 3122, 3143, 3144, 3205, 3238, 3256, 3288, 3373, 3447, 3460, 3478, 3488, 3517, 3576, 3585, 3626, 3654, 3658, 3669, 3690, 3802, 3852, 3871, 3874, 3923, 3997, 4013, 4069, 4071, 4072, 4103, 4151, 4289, 4313, 4348, 4351, 4390, 4444, 4480, 4489, 4558, 4575, 4580, 4581, 4598, 4608, 4639, 4686, 4689, 4823, 4846, 4906, 4912, 5016, 5030, 5079, 5182, 5187, 5195, 5215, 5234, 5245, 5293, 5294, 5363, 5366, 5377, 5393, 5395, 5411, 5507, 5514, 5551, 5646, 5704, 5717, 5744, 5747, 5783, 5791, 5801, 5825, 5839, 5892, 5911, 5912, 5967, 5995, 6023, 6109, 6113, 6261, 6279, 6291, 6308, 6358, 6384, 6425, 6431, 6436, 6478, 6508, 6568, 6583, 6657, 6687, 6702, 6716, 6730, 6748, 6805, 6806, 6955, 6961, 7013, 7033, 7058, 7072, 7093, 7095, 7118, 7131, 7140, 7163, 7171, 7213, 7244, 7255, 7279, 7287, 7306, 7309, 7389, 7435, 7446, 7469, 7473, 7484, 7505, 7523, 7542, 7581, 7595, 7621, 7662, 7684, 7692, 7829, 7846, 7871, 7902, 7929, 7934, 7957, 7969, 7970, 7993, 8016, 8042, 8078, 8094, 8123, 8130, 8151, 8177, 8186, 8194, 8207, 8214, 8300, 8312, 8323, 8395, 8397, 8441, 8469, 8476, 8489, 8610, 8789, 9209, 9223, 9378, 9382, 9421, 9463, 9525, 9602, 9733, 9761, 9766, 9808, 9866, 9904, 9922, 9941, 9977, 9980, 9985, 10024, 10041, 10058, 10062, 10064, 10066, 10069, 10086, 10098, 10102, 10107, 10138, 10144, 10170, 10176, 10186, 10199, 10212, 10223, 10238, 10286, 10287, 10355, 10410, 10417, 10430, 10537, 10565, 10600, 10620, 10641, 10733, 10735, 10784, 10795, 10826, 10853, 10881, 10888, 10905, 10916, 10954, 11015, 11028, 11029, 11061, 11074, 11157, 11171, 11195, 11200, 11203, 11275, 11344, 11416, 11442, 11484, 11499, 11590, 11597, 11614, 11624, 11628, 11653, 11658, 11702, 11709, 11715, 11724, 11729, 11742, 11761, 11769, 11777, 11789, 11821, 11889, 11959, 12046, 12053, 12073, 12128, 12140, 12168, 12171, 12204, 12231, 12235, 12238, 12261, 12291, 12394, 12401, 12447, 12450, 12521, 12575, 12601, 12666, 12671, 12677, 12680, 12780, 12859, 12877, 12911, 12933, 13063, 13078, 13088, 13111, 13205, 13219, 13228, 13235, 13300, 13323, 13454, 13474, 13549, 13600, 13682, 13688, 13749, 13799, 13865, 13895, 13899, 13920, 13962, 13970, 13982, 13991, 14060, 14099, 14163, 14178, 14200, 14202, 14204, 14226, 14234, 14248, 14305, 14318, 14330, 14358, 14434, 14457, 14471, 14620, 14683, 14727, 14754, 14796, 14799, 14810, 14814, 14825, 14875, 14876, 14920, 14931, 14954, 14970, 15034, 15041, 15117, 15145, 15147, 15156, 15238, 15241, 15305, 15353, 15369, 15379, 15380, 15394, 15403, 15415, 15458, 15486, 15503, 15512, 15604, 15622, 15636, 15645, 15660, 15677, 15705, 15730, 15828, 15876, 15896, 15923, 15984, 15994, 16077, 16191, 16219, 16265, 16273, 16308, 16421, 16423, 16443, 16490, 16510, 16587, 16589, 16615, 16694, 16728, 16824, 16865, 16902, 16910, 16923, 16927, 16931, 16939, 16981, 16999]
    """

    """
    FRA - Best genes 200 px Lgbm MSE
    [61, 85, 96, 139, 193, 231, 280, 320, 471, 661, 696, 780, 1079, 1100, 1267, 1498, 1539, 1661, 1810, 1963, 2163, 2364, 2401, 2452, 2460, 2484, 2579, 2599, 2620, 2746, 2787, 2907, 2941, 2973, 3062, 3205, 3444, 3486, 3583, 3661, 3665, 3704, 3725, 3727, 3788, 3888, 3923, 4073, 4088, 4091, 4146, 4383, 4427, 4596, 4746, 5012, 5079, 5106, 5292, 5309, 5406, 5758, 5993, 6024, 6042, 6144, 6160, 6263, 6298, 6337, 6355, 6459, 6548, 6885, 6898, 7066, 7126, 7149, 7210, 7216, 7248, 7300, 7374, 7534, 7538, 7577, 7693, 7767, 8005, 8242, 8261, 8299, 8321, 8377, 8506, 8711, 8736, 8825, 8848, 8941, 9033, 9146, 9334, 9467, 9484, 9540, 9691, 9708, 9742, 9765, 9865, 10001, 10036, 10076, 10133, 10178, 10211, 10238, 10242, 10385, 10419, 10562, 10566, 10669, 10767, 10795, 10833, 11070, 11125, 11410, 11425, 11430, 11461, 11483, 11506, 11555, 11634, 11648, 11706, 11840, 11921, 12006, 12263, 12530, 12546, 12578, 12811, 12965, 12987, 13066, 13112, 13167, 13246, 13377, 13572, 13710, 13766, 13961, 14012, 14021, 14053, 14127, 14265, 14345, 14349, 14368, 14420, 14432, 14504, 14519, 14635, 14672, 14694, 14736, 14740, 14795, 14818, 14933, 14975, 15004, 15067, 15295, 15315, 15352, 15376, 15630, 15713, 15760, 15779, 15822, 15848, 16146, 16265, 16313, 16496, 16593, 16762, 16795, 16857, 16924]
    """

    """
    FRA - Best genes 300 px Ridge MSE Lgbm Activation MSE
    Best Chromosome with 300 features: [85, 160, 161, 195, 201, 232, 330, 332, 482, 579, 669, 890, 895, 898, 1008, 1285, 1300, 1307, 1316, 1339, 1573, 1633, 1666, 1787, 1915, 1969, 2000, 2080, 2123, 2324, 2389, 2443, 2454, 2482, 2541, 2563, 2642, 2710, 2722, 2761, 2770, 2799, 2820, 2850, 2862, 2971, 3105, 3153, 3168, 3306, 3311, 3348, 3392, 3425, 3630, 3647, 3677, 3771, 3801, 3945, 3964, 4206, 4360, 4473, 4477, 4507, 4535, 4542, 4562, 4777, 4829, 4926, 5049, 5065, 5103, 5122, 5195, 5216, 5356, 5385, 5400, 5530, 5532, 5568, 5617, 5726, 5805, 5902, 5997, 6000, 6016, 6024, 6069, 6329, 6365, 6376, 6401, 6405, 6415, 6417, 6438, 6440, 6461, 6509, 6597, 6643, 6644, 6706, 6724, 6747, 6770, 6793, 6817, 6923, 6948, 7024, 7078, 7091, 7117, 7143, 7145, 7173, 7243, 7264, 7362, 7446, 7505, 7538, 7579, 7608, 7619, 7653, 7775, 7798, 7813, 7932, 7964, 8101, 8114, 8246, 8306, 8401, 8699, 8894, 9076, 9143, 9177, 9208, 9287, 9315, 9353, 9400, 9402, 9403, 9482, 9610, 9625, 9633, 9772, 9818, 9836, 9851, 9884, 9923, 9966, 10043, 10102, 10150, 10321, 10329, 10359, 10380, 10408, 10437, 10472, 10550, 10784, 10820, 10852, 10925, 10998, 11002, 11013, 11036, 11042, 11239, 11266, 11303, 11322, 11327, 11335, 11403, 11406, 11449, 11463, 11520, 11570, 11578, 11630, 11656, 11709, 11783, 11785, 11909, 11925, 11962, 11991, 12056, 12097, 12098, 12105, 12112, 12119, 12183, 12239, 12241, 12360, 12438, 12520, 12570, 12622, 12656, 12678, 12697, 12804, 12815, 12850, 12940, 12959, 12994, 13018, 13038, 13125, 13143, 13152, 13276, 13323, 13406, 13419, 13497, 13553, 13582, 13633, 13666, 13686, 13689, 13740, 13766, 13776, 13791, 13804, 13831, 13858, 13962, 14049, 14154, 14174, 14177, 14308, 14395, 14429, 14533, 14625, 14647, 14676, 14770, 14821, 14878, 15011, 15012, 15141, 15167, 15174, 15242, 15279, 15286, 15374, 15403, 15421, 15424, 15513, 15521, 15552, 15634, 15661, 15694, 15774, 15814, 15956, 16007, 16139, 16179, 16287, 16300, 16447, 16556, 16617, 16659, 16880, 16915]
    """


if __name__ == "__main__":
    filename = "C:\\Users\\U81089\\OneDrive - Refinitiv\\Desktop\\Wind Code\\wind_files\\calib_data_FRA.mat"
    dataset = "FRA"

    # filename = "C:\\Users\\U81089\\OneDrive - Refinitiv\\Desktop\\Wind Code\\wind_files\\CAISO.mat"
    # dataset = "USACAISONP15"
    # dataset = "USACAISOSP15"

    # filename = "C:\\Users\\U81089\\OneDrive - Refinitiv\\Desktop\\Wind Code\\wind_files\\ERCOT.mat"
    # dataset = "USAERCOTNorthWest"
    # dataset = "USAERCOTSouthHouston"

    args = InitArgs(filename, dataset)
    main(args)



