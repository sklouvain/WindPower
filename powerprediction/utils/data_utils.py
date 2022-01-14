import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import time

import pandas as pd

from pyCdb import pycomm


def train_val_test_split(x, y, val_percentage=0.15, test_percentage=0.15):
    train_percentage = 1.0 - val_percentage - test_percentage
    j = int(train_percentage * y.shape[0])
    k = int((train_percentage + val_percentage) * y.shape[0])

    x_train, y_train = x[:j], y[:j]
    x_val, y_val = x[j:k], y[j:k]
    x_test, y_test = x[k:], y[k:]
    print(
        f"Train points: {y_train.shape[0]}, validation points: {y_val.shape[0]}, test points: {y_test.shape[0]}"
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


""" New util functions added by Tibi """


def get_font_dict(family='monospace', color='black', weight='normal', size=11):
    font = {
        'family': family,
        'color': color,
        'weight': weight,
        'size': size,
    }

    return font


def zero_num(y):
    zero_count = 0
    eps = 0.000001
    assert len(np.shape(y)) == 1

    for val in y:
        if val < eps:
            zero_count += 1

    return zero_count


def mape(y_true, y_pred):
    abs_list = [(abs(y_true[i] - y_pred[i]) / y_true[i]) for i in range(len(y_true)) if y_true[i] > 0]
    # print(abs_list)
    return sum(abs_list) / len(abs_list) * 100


def create_pos_dict_from_list(points):
    points_dict = dict()
    for i, point in enumerate(points):
        points_dict[point] = i
    return points_dict


def find_interval_pos(points, val):
    diff = abs(points[0] - points[1]) / 2 + 0.0001
    for i, point in enumerate(points):
        if abs(val - point) < diff:
            return i
    return None


def plot_map(sample_x, cmap='hot', interpolation='nearest', title=None, map_points=None, lat=None, lon=None, farms=None,
             min_coeff=0.13, max_coeff=0.13, mid_coeff=0.5):
    # print("Sample shape", np.shape(sample_x))
    reshaped_x = np.reshape(sample_x, np.shape(sample_x)[:2])
    min_sample = np.min(reshaped_x)
    max_sample = np.max(reshaped_x)

    x_shape = np.shape(sample_x)
    # print(x_shape[0], x_shape[1])
    # new_map = np.zeros((x_shape[0], x_shape[1]))

    map_points = map_points if map_points else set()
    count = -1
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            count += 1
            if count in map_points:
                reshaped_x[i, j] = max_sample + abs(max_sample) * max_coeff
                # plt.plot(i, j, 'bo')

    # plot_map(new_map)

    font = get_font_dict()

    extent = None
    if lat is not None and lon is not None:
        extent = (lon[0], lon[-1], lat[-1], lat[0])
        plt.ylabel('Latitude', fontdict=font)
        plt.xlabel('Longitude', fontdict=font)

        if farms is not None:
            # lat_dict = create_pos_dict_from_list(lat)
            # lon_dict = create_pos_dict_from_list(lon)

            for name, latitude, longitude in farms:
                lat_point = find_interval_pos(lat, latitude)
                if lat_point is None:
                    continue
                # lat_point = lat_dict[lat_point]
                lon_point = find_interval_pos(lon, longitude)
                if lon_point is None:
                    continue
                # lon_point = lon_dict[lon_point]

                if abs(reshaped_x[lat_point, lon_point] - (max_sample + abs(max_sample) * max_coeff)) < 0.01:
                    reshaped_x[lat_point, lon_point] = (1 - mid_coeff) * (min_sample - abs(min_sample) * min_coeff) +\
                                                       mid_coeff * (max_sample + abs(max_sample) * max_coeff)
                else:
                    reshaped_x[lat_point, lon_point] = min_sample - abs(min_sample) * min_coeff

    plt.title(title, fontdict=font)
    max_sample = max_sample * (1 + max_coeff)
    min_sample = min_sample * (1 - min_coeff)

    plt.imshow(reshaped_x, cmap=cmap, interpolation=interpolation, vmin=min_sample, vmax=max_sample, extent=extent)
    plt.colorbar()

    plt.show()


def plot_target(y, dates=None):
    font = get_font_dict()
    plt.title('Target values', fontdict=font)
    if dates is not None:
        plt.plot(dates, y, color='c')
    else:
        plt.plot(y, color='c')
    plt.show()


def plot_predictions(y_actual, y_predicted):
    # points = [(y_actual[i], y_predicted[i]) for i in range(len(y_actual))]
    font = get_font_dict()
    plt.scatter(y_actual, y_predicted, color='g', marker='v')
    plt.title("Actual vs Predicted y value", fontdict=font)
    plt.xlabel("Actual y", fontdict=font)
    plt.ylabel("Predicted y", fontdict=font)
    plt.show()


def convert_multidimensional_map_points(map_points):
    wind_points = [point // 2 for point in map_points if point % 2 == 0]
    temp_points = [point // 2 for point in map_points if point % 2 == 1]

    return set(wind_points), set(temp_points)


def permute_numpy_arrays(x, y):
    assert x.shape[0] == y.shape[0]
    perm_indices = np.random.RandomState(seed=17).permutation(x.shape[0])
    return x[perm_indices, :], y[perm_indices]


def shuffle_train_val(x_train, y_train, x_val, y_val):
    train_len = x_train.shape[0]
    full_x = np.concatenate((x_train, x_val), axis=0)
    full_y = np.concatenate((y_train, y_val), axis=0)
    full_x, full_y = permute_numpy_arrays(full_x, full_y)
    return full_x[:train_len, :], full_y[:train_len], full_x[train_len:, :], full_y[train_len:]


def compute_variance_from_data(x):
    nb_lines = np.shape(x)[1]
    nb_columns = np.shape(x)[2]

    var_matrix = np.zeros((nb_lines, nb_columns))
    x_reshaped = np.reshape(x, np.shape(x)[:3])

    for i in range(nb_lines):
        for j in range(nb_columns):
            arr = x_reshaped[:, i, j]
            curr_var = np.var(arr)
            var_matrix[i, j] = curr_var

    return var_matrix


def repair_coordinates_list(coord_list):
    for i in range(len(coord_list) - 1):
        if coord_list[i + 1] - coord_list[i] > 10:
            return coord_list[i + 1:] + coord_list[:i + 1]
    return coord_list


def get_coordinates_from_arrays(lat, lon):
    lat_set = set()
    for param in lat:
        lat_set.add(param[0])

    lat_list = list(lat_set)
    lat_list.sort()
    lat_list = repair_coordinates_list(lat_list)
    lat_list.reverse()

    lon_set = set()
    for param in lon:
        lon_set.add(param[0])

    lon_list = list(lon_set)
    lon_list.sort()
    lon_list = repair_coordinates_list(lon_list)

    return np.array(lat_list), np.array(lon_list)


def read_json_from_file(file_name, root_path="..\\store_files\\dataset_selected_points\\"):
    file = open(root_path + file_name)
    json_ob = json.load(file)
    file.close()
    return json_ob


def write_json_to_file(json_ob, json_name, indent=5, root_path="..\\store_files\\dataset_selected_points\\"):
    with open(root_path + json_name, "w") as outfile:
        json.dump(json_ob, outfile, indent=indent)


def convert_dms_to_decimal(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / 3600


def convert_decimal_to_dms(decimal):
    degrees = abs(int(decimal)) * np.sign(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = (decimal - degrees - minutes / 60) * 3600
    seconds = abs(seconds)
    minutes = abs(minutes)
    return degrees, minutes, seconds


def get_points_list_from_json(json_ob):
    farms_list = json_ob['farms']
    farms = []

    for farm in farms_list:
        name = farm["name"]
        lat_degrees = farm["lat_degrees"]
        lat_minutes = farm["lat_minutes"]
        lat_seconds = farm["lat_seconds"]
        lon_degrees = farm["lon_degrees"]
        lon_minutes = farm["lon_minutes"]
        lon_seconds = farm["lon_seconds"]

        latitude = convert_dms_to_decimal(lat_degrees, lat_minutes, lat_seconds)
        longitude = convert_dms_to_decimal(lon_degrees, lon_minutes, lon_seconds)
        print("Name:", farm['name'])
        print("Latitude:", lat_degrees, "°", lat_minutes, "\'", lat_seconds, "\"")
        print("Converted Latitude", latitude)
        print("Longitude", lon_degrees, "°", lon_minutes, "\'", lon_seconds, "\"")
        print("Converted Longitude", longitude)
        print()

        farms.append((name, latitude, longitude))

    return farms


def get_data_from_curves(curve_ids, start_date, end_date):
    # datatbl = pycomm.get_curves(112829829, f1, f1, f1, f2)
    # datatbl = pycomm.get_curves(112556827, f1, f2, f1, f2)
    # datatbl = pycomm.get_curves(103147166, f1, f2, f1, f2)

    datatbl = pycomm.get_curves(curve_ids, start_date, end_date, start_date, end_date)

    size = datatbl.size
    cid = datatbl['ID'][0]
    fdate = datatbl['FD']
    vdate = datatbl['VD']
    values = datatbl['V']

    print(size)
    print(cid)
    print("fdate", fdate[0])
    print("vdate", vdate[0])

    for i, value in enumerate(values):
        if i < len(values) - 10:
            continue

        print("Sample number: ", i, "Value: ", value, "Fdate", fdate[i], "Vdate", vdate[i])
        # if i == 10:
        #     break

    # print(value)

    # FRA, capacity: 110950845
    # FRA, production: 103147166
    # RWE, capacity: 112556827
    # RWE, production: 101679913

    return datatbl


def get_targets_from_datatable(datatbl):
    values = datatbl['V']

    curve_list = []
    for i, value in enumerate(values):
        curve_list.append(value)

    return np.array(curve_list)


def convert_data_to_hourly(y_prod, prod_tbl):
    prod_values = []
    prod_times = []
    cnt = 0
    last_hour = -1
    last_date = ""

    for i in range(len(y_prod)):
        curr_timestamp = prod_tbl['FD'][i]
        curr_hour = curr_timestamp.hour
        curr_date = str(curr_timestamp).split()[0]

        if curr_hour != last_hour or curr_date != last_date:
            if len(prod_values) != 0:
                prod_values[-1] /= cnt

            if last_hour != -1:
                last_datetime = prod_times[-1].to_pydatetime()
                aux_datetime = last_datetime + datetime.timedelta(hours=1)
                curr_datetime = curr_timestamp.to_pydatetime()

                while aux_datetime != curr_datetime:
                    prod_values.append(np.nan)
                    prod_times.append(pd.Timestamp(aux_datetime))
                    aux_datetime = aux_datetime + datetime.timedelta(hours=1)

            prod_values.append(y_prod[i])
            prod_times.append(curr_timestamp)
            cnt = 1
        else:
            prod_values[-1] += y_prod[i]
            cnt += 1

        last_date = curr_date
        last_hour = curr_hour

    prod_values[-1] /= cnt

    return np.array(prod_values), prod_times


def check_value_list(field_name, file_name, key_name="data_values",
                     root_path="..\\store_files\\initial_run_files\\"):
    input_name = file_name.split(".")[0]
    json_ob = read_json_from_file(file_name=file_name, root_path=root_path)
    fields = json_ob[key_name]
    if field_name not in fields:
        raise KeyError("Invalid value for {}, please make sure it is in: {}".format(input_name, fields))


def get_nb_of_digits(number):
    if number == 0:
        return 1

    number_cnt = 0
    while number > 0:
        number //= 10
        number_cnt += 1

    return number_cnt


def parse_date_from_file_name(file_name):
    num_list = [char if "0" <= char <= "9" else " " for char in file_name]
    new_str = "".join(num_list)
    numbers = new_str.split(" ")
    date_number = None
    for number in numbers:
        if len(number) and get_nb_of_digits(int(number)) == 10:
            date_number = number
            break
    return int(date_number[:4]), int(date_number[4:6]), int(date_number[6:8]), int(date_number[8:10])


"""
def compute_variance_from_matrix_list(x):
    assert np.shape(x) == 3, "x must have only 3 dimensions"

    nb_lines = np.shape(x)[1]
    nb_columns = np.shape(x)[2]
    var_matrix = np.zeros((nb_lines, nb_columns))

    for i in range(nb_lines):
        for j in range(nb_columns):
            arr = x[:, i, j]
            curr_var = np.var(arr)
            var_matrix[i, j] = curr_var

    return var_matrix


def compute_variance_from_data(x):
    var_matrix = []
    if len(np.shape(x)) == 4:
        for dim in range(np.shape(x)[3]):
            x_reshaped = x[:, :, :, dim]
            x_var = compute_variance_from_matrix_list(x_reshaped)
            var_matrix = var_matrix + x_var
    elif len(np.shape(x)) == 3:
        var_matrix = compute_variance_from_data(x)

    return var_matrix
"""
