import powerprediction
from powerprediction.utils.data_utils import read_json_from_file, check_value_list
from powerprediction.experiments.compute_region import compute_region

def read_initial_run_file():
    json_ob = read_json_from_file(
        file_name="initial_run_input.json",
        root_path="..\\store_files\\initial_run_files\\"
    )

    region = json_ob["region"]
    check_value_list(region, "region.json")
    subregion = json_ob["subregion"]
    check_value_list(subregion, "subregion.json", key_name=region)
    run_method = json_ob["run_method"]
    check_value_list(run_method, "run_method.json")

    return region, subregion, run_method


if __name__ == "__main__":
    region_name, subregion_name, run_method_name = read_initial_run_file()
    compute_region(region_name, subregion_name, run_method_name)




