import numpy as np
import pandas
import datetime

from powerprediction.utils.data_utils import read_json_from_file, get_data_from_curves
from powerprediction.utils.data_utils import get_targets_from_datatable, convert_data_to_hourly


def read_cdb_curves(region, subregion, date, hours_diff):
    year, month, day, hour = date
    # dates = getattr(dataset, "dates").copy()
    # start_date = pandas.to_datetime(dates[0]) - datetime.timedelta(days=1)
    # print("Start date:", start_date, type(start_date), dates[0], type(dates[0]))
    # end_date = pandas.to_datetime(dates[-1]) - datetime.timedelta(days=1)
    # print("End date:", end_date, type(end_date), dates[-1], type(dates[-1]))

    day -= 1
    start_date = datetime.datetime(year=year, month=month, day=day, hour=hour)
    end_date = start_date + datetime.timedelta(hours=hours_diff - 1)
    print("Start date:", start_date, type(start_date))
    print("End date:", end_date, type(end_date))

    key_name = region + "_" + subregion
    print("Json Key Name:", key_name)
    json_name = "wind_curves.json"
    json_ob = read_json_from_file(file_name=json_name, root_path="..\\store_files\\cdb_curves\\")
    capacity_curve_id = json_ob[key_name]['capacity']
    print("Capacity Curve ID:", capacity_curve_id)
    production_curve_id = json_ob[key_name]['production']
    print("Production Curve ID", production_curve_id)

    # y_prod_old = getattr(dataset, "production").copy()
    prod_tbl = get_data_from_curves(curve_ids=production_curve_id, start_date=start_date, end_date=end_date)
    print("Production table type:", type(prod_tbl))
    # prod_tbl.resample('D').avg()
    y_prod = get_targets_from_datatable(prod_tbl)
    # y_cap_old = getattr(dataset, "capacity").copy()
    cap_tbl = get_data_from_curves(curve_ids=capacity_curve_id, start_date=start_date, end_date=end_date)
    y_cap = get_targets_from_datatable(cap_tbl)

    # print("Mat file production shape:", np.shape(y_prod_old))
    print("Cdb production shape:", np.shape(y_prod))
    # print("Mat file capacity shape:", np.shape(y_cap_old))
    print("Cdb capacity shape:", np.shape(y_cap))

    y_prod_tib, times_tib = convert_data_to_hourly(y_prod, prod_tbl)

    print("Hourly converted production shape:", np.shape(y_prod_tib))

    # print("Capacity MAPE:", mape(y_cap_old, y_cap))

    # for i in range(len(y_prod_old)):
    # for i in range(100):
    #     if i >= len(y_prod_tib):
    #         print("Index", i, y_prod_old[i], dates[i])
    #     else:
    #         print("Index", i, y_prod_old[i], dates[i], "Tib", y_prod_tib[i], times_tib[i])

    for i in range(len(y_prod) - 10, len(y_prod)):
        print(y_prod[i], prod_tbl['FD'][i])

    # print("Production MAPE:", mape(y_prod_old, y_prod_tib))
    # print("Dates type", type(dates))

    print("Cap diffs:")
    # for i in range(len(y_cap_old)):
    #     if np.isnan(y_cap_old[i]) or np.isnan(y_cap[i]) or abs(y_cap_old[i] - y_cap[i]) > 0.1:
    #         print("Index:", i, y_cap_old[i], dates[i], y_cap[i], cap_tbl['FD'][i])
    #
    # print("Prod diffs:")
    # for i in range(len(y_prod_old)):
    #     if np.isnan(y_prod_old[i]) or np.isnan(y_prod_tib[i]) or abs(y_prod_old[i] - y_prod_tib[i]) > 0.1:
    #         print("Index:", i, y_prod_old[i], dates[i], y_prod_tib[i], times_tib[i])

    return y_cap, y_prod_tib
