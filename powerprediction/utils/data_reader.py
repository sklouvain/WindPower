from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Iterator
from itertools import product

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from einops import rearrange, repeat
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import time as tm

from powerprediction.utils.data_utils import train_val_test_split

basic_argparser = ArgumentParser()
basic_argparser.add_argument("filename", type=str, help="Matlab file containing the 'raw' data.")
basic_argparser.add_argument("--dataset", type=str, default="DK1on")

root_dir = Path(__file__).parent.parent.parent.resolve()
cache_dir = root_dir / ".cache"


def to_tensor(*arrs) -> Sequence[Tensor]:
    tenss = []
    for arr in arrs:
        tenss.append(Tensor(arr))
    return tuple(tenss)


@dataclass(frozen=True)
class DatasetContainer:
    name: str
    train_set: TensorDataset
    val_set: TensorDataset
    test_set: TensorDataset
    window_size: int
    add_coords: bool
    features: Sequence[str]

    @property
    def n_channels(self) -> int:
        return self.train_set[0][0].shape[0]

    def checkpoint_path(self, model: str) -> Path:
        return root_dir.joinpath("checkpoints", model, repr(self))

    @property
    def xy_dims(self) -> Tuple[int, int]:
        return tuple(self.train_set[0][0].shape[1:])

    def n_outputs(self) -> int:
        ds = self.train_set[0][0].shape
        return ds[1] * ds[2]

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=5)

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.val_set, shuffle=False, batch_size=batch_size, num_workers=4)

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    def __repr__(self) -> str:
        return f"{self.name}_{self.window_size}_{self.add_coords}_{'_'.join(self.features)}"

    def __str__(self) -> str:
        return repr(self)


@dataclass
class WindpowerContainer:
    name: str

    production: np.ndarray
    capacity: np.ndarray
    ratio: np.ndarray

    wind_u_100m: np.ndarray
    wind_v_100m: np.ndarray

    wind_speed_100m: np.ndarray
    wind_angle_100m: np.ndarray

    mask: np.ndarray

    dates: np.ndarray

    norm_data: Dict[str, np.float32]

    temperature: np.ndarray
    pressure: Optional[np.ndarray] = None

    wind_u_10m: Optional[np.ndarray] = None
    wind_v_10m: Optional[np.ndarray] = None
    wind_speed_10m: Optional[np.ndarray] = None
    wind_angle_10m: Optional[np.ndarray] = None

    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None

    def series(self, name: str) -> pd.Series:
        return pd.Series(self.get(name), index=self.get("dates"), name=name)

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self.production, self.capacity, self.ratio],
            index=self.get("dates"),
            columns=["production", "capacity", "ratio"],
        )

    def get(self, name: str, mask: bool = True) -> np.ndarray:
        data = getattr(self, name)
        if len(data.shape) != 3:
            return data
        mask = self.mask if mask else 1.0
        return data * mask

    def plot(self, plot_name: str, index: int) -> plt.figure:
        fig = plt.figure()
        plt.imshow(getattr(self, plot_name)[index])
        return fig

    def get_features(self, feat_names: Sequence[str], mask: bool = True) -> Dict[str, np.ndarray]:
        return {n: self.get(n, mask=mask) for n in feat_names}

    @staticmethod
    def coordinate_channels(shape: Tuple[int, int]) -> np.ndarray:
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        xr = repeat(x, "c -> c h", h=shape[1])
        yr = repeat(y, "c -> w c", w=shape[0])
        return np.stack([xr, yr], axis=-1)

    @staticmethod
    def add_coordinate_channels(x: np.ndarray) -> np.ndarray:
        # Assumes shape (batch_size, height, width, channels)
        shape = tuple(x.shape[1:3])
        coords = WindpowerContainer.coordinate_channels(shape)
        reps = repeat(coords, "w h c-> b w h c", b=x.shape[0])
        return np.concatenate([x, reps], axis=-1)

    def load_data(
        self,
        window_size: int = 0,
        features: Sequence[str] = ("wind_speed_100m",),
    ) -> Tuple[np.ndarray, np.ndarray]:

        y = self.ratio.copy()
        if window_size > 0 and y.shape[0]:
            y = y[window_size:(-window_size)]
        print("NaNs in y:", np.count_nonzero(np.isnan(y)))
        y[np.isnan(y)] = np.nanmean(y)

        mask = self.mask
        region_shape = mask.shape
        data_shape = (
            self.wind_speed_100m.shape[0],
            region_shape[0],
            region_shape[1],
            len(features),
            2 * window_size + 1,
        )
        print("Data shape:", data_shape)
        x = np.zeros(data_shape)

        for i, feat in enumerate(features):
            mu = self.norm_data[f"{feat}_mean"]
            sigma = self.norm_data[f"{feat}_std"]

            # Feature data
            v = getattr(self, feat).copy()
            print(f"NaNs in feature {feat}: {np.count_nonzero(np.isnan(v))}")
            print(f"Shape of input: {v.shape}")
            v[np.isnan(v)] = mu  # mean-imputation for NaNs
            v = ((v - mu) / sigma) * mask  # Normalization and masking

            for jjj in range(2 * window_size + 1):
                last_ind = data_shape[0] - window_size + jjj + 1
                x[:, :, :, i, jjj] = v[jjj:last_ind, :, :]

        print(f"Size of x in megabytes: {(x.nbytes / 2 ** 20):.2f}")
        x = rearrange(x, "samples height width features window -> samples height width (features window)")
        return x, y

    def get_train_val_test(self, window_size: int,
                           add_coordinates: bool, features: Sequence[str]) -> Sequence[np.array]:
        x, y = self.load_data(window_size=window_size, features=features)
        if add_coordinates:
            x = self.add_coordinate_channels(x)
        x = rearrange(x, "b h w c -> b c h w")
        # s = x.shape[1:]
        # n_channels = s[0]
        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_pytorch_datasets(self, window_size: int,
                             add_coordinates: bool, features: Sequence[str]) -> DatasetContainer:
        # x, y = self.load_data(window_size=window_size, features=features)
        # if add_coordinates:
        #     x = self.add_coordinate_channels(x)
        # x = rearrange(x, "b h w c -> b c h w")
        # # s = x.shape[1:]
        # # n_channels = s[0]
        # x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y)
        x_train, y_train, x_val, y_val, x_test, y_test = self.get_train_val_test(window_size, add_coordinates,
                                                                                 features)
        x_train, y_train, x_val, y_val, x_test, y_test = to_tensor(x_train, y_train, x_val, y_val, x_test, y_test)

        train_set = TensorDataset(x_train, y_train)
        val_set = TensorDataset(x_val, y_val)
        test_set = TensorDataset(x_test, y_test)
        return DatasetContainer(self.name, train_set, val_set, test_set, window_size, add_coordinates, features)

    def get_dataset_configurations(self) -> Iterator[DatasetContainer]:
        # exps = list(product(windows, add_coords, features))
        configs = [(0, False, ('wind_speed_100m',)),
                   (1, True, ('wind_speed_100m',)),
                   (1, True, ('wind_speed_100m', 'wind_speed_10m')),
                   (1, False, ('wind_speed_100m', 'wind_speed_10m')),
                   (0, True, ('wind_speed_100m', 'wind_speed_10m'))]
        for exp in configs:
            yield self.get_pytorch_datasets(*exp)


def create_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Create mask for region

    Some of the regions are not rectangular. This function create a mask that is zero
    everywhere outside of the region and one inside the region. This mask makes sure
    the region can be placed in a rectangular shape.

    Args:
        lat (numpy.ndarray): List of latitudes
        lon (numpy.ndarray): List of longitudes

    Returns:
        numpy.ndarray: Mask
    """

    lat = lat.squeeze()
    lon = lon.squeeze()

    lat_diffs = lat[1:] - lat[:-1]
    lon_diffs = lon[1:] - lon[:-1]

    lat_scale = np.abs((lat_diffs[lat_diffs != 0.0])).min()
    lon_scale = np.abs((lon_diffs[lon_diffs != 0.0])).min()

    lat_idx = (lat / lat_scale).astype(int)
    lon_idx = (lon / lon_scale).astype(int)

    lat_idx -= lat_idx.min()
    lon_idx -= lon_idx.min()

    h = lat_idx.max() + 1
    w = lon_idx.max() + 1

    print("lat length:", h)
    print("lon length:", w)

    mask = np.zeros(shape=(h, w), dtype=bool)

    mask[-lat_idx, lon_idx] = 1

    return mask


def _flip_data(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.all(mask):
        temp = np.zeros((mask.size, values.shape[1]), dtype=np.float32)
        temp[mask.T.ravel()] = values
        values = temp

    values = values.reshape(mask.shape[0], mask.shape[1], -1, order="F")
    values = np.moveaxis(values, -1, 0)

    return values


def print_keys(data, path=""):
    if hasattr(data, "keys"):
        for key in data.keys():
            print_keys(data[key], path + "/" + key)
    else:
        print(path, data.shape, data.dtype)


def _verify_dates(data: h5py.Group) -> Tuple[np.ndarray, slice, slice]:
    if "production/dates" not in data or "capacity/dates" not in data:
        return np.empty(shape=[0]), slice(None), slice(None)

    production_dates = data["production/dates"][0]
    capacity_dates = data["capacity/dates"][0]

    # check that production and capacity is aligned
    assert np.all(production_dates == capacity_dates)

    # refer to these dates by electricity_dates from now on
    electricity_dates = production_dates

    # Make sure all of the weather data is aligned
    prev = None
    for key in [
        "Temperature_2m",
        "Pressure_reduced_to_MSL",
        "U_component_of_wind_100m",
        "V_component_of_wind_100m",
        "U_component_of_wind_10m",
        "V_component_of_wind_10m",
    ]:
        if f"EC/{key}/dates" not in data:
            continue
        dates = data[f"EC/{key}/dates"][0]
        if prev is not None:
            assert np.all(dates == prev)  # type: ignore
        prev = dates

    weather_dates = dates

    # Dates are given in fractional days, but really have hourly frequency, so multiply by 24
    weather_dates *= 24
    electricity_dates *= 24

    # Convert to integer
    weather_dates = weather_dates.astype(int)
    electricity_dates = electricity_dates.astype(int)

    # Make sure that there are no gaps in the dates
    assert max(weather_dates[1:] - weather_dates[:-1]) == 1
    assert max(electricity_dates[1:] - electricity_dates[:-1]) == 1

    # Weather dates and electricity dates might not be aligned
    if weather_dates.size < electricity_dates.size:

        assert weather_dates[0] >= electricity_dates[0]
        assert weather_dates[-1] <= electricity_dates[-1]

        start_idx = electricity_dates.searchsorted(weather_dates[0], side="left")
        stop_idx = electricity_dates.searchsorted(weather_dates[-1], side="right")

        weather_slice = slice(None)
        electricity_slice = slice(start_idx, stop_idx)

    elif weather_dates.size > electricity_dates.size:

        assert weather_dates[0] <= electricity_dates[0]
        assert weather_dates[-1] >= electricity_dates[-1]

        start_idx = weather_dates.searchsorted(electricity_dates[0], side="left")
        stop_idx = weather_dates.searchsorted(electricity_dates[-1], side="right")

        weather_slice = slice(start_idx, stop_idx)
        electricity_slice = slice(None)

    else:
        if weather_dates[0] < electricity_dates[0]:
            offset = weather_dates.searchsorted(electricity_dates[0], side="left")

            weather_slice = slice(offset, None)
            electricity_slice = slice(0, -offset)

        elif weather_dates[0] > electricity_dates[0]:
            offset = electricity_dates.searchsorted(weather_dates[0], side="left")

            weather_slice = slice(0, -offset)
            electricity_slice = slice(offset, None)

        else:
            weather_slice = slice(None)
            electricity_slice = slice(None)

    assert np.all(weather_dates[weather_slice] == electricity_dates[electricity_slice])

    # Only call them dates from now on
    dates = weather_dates[weather_slice]

    # dates are relative to year 0
    dates = np.datetime64("0000-01-01 00") + dates * np.timedelta64(1, "h")

    return dates, weather_slice, electricity_slice


class ContainerBuilder:
    """
    Utility class to build a WindpowerContainer.
    """

    def __init__(self, filename: str, dataset_name: str):
        self.filename = filename
        self.dataset_name = dataset_name

        self.norm_data: Dict[str, np.float32] = {}
        self.args: Dict[str, Any] = {"name": dataset_name, "norm_data": self.norm_data}

        self.cache_dir = cache_dir / filename / dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.norm_data_path = self.cache_dir / "norm_train_data.npz"

    def add(self, key: str, value: Any):
        assert key not in self.args

        # Make sure all floating point arrays are float32
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            value = value.astype(np.float32)

        # Store numpy values to disk, in order to load them as memory maps,
        # to improve memory efficiency of loading data
        if isinstance(value, np.ndarray):
            path = self.cache_dir / f"{key}.npy"
            np.save(path, value)
            value = np.load(path, mmap_mode="r")

        self.args[key] = value

        return value

    def add_statistics(self, key: str, values: np.ndarray, mask: np.ndarray):
        if not np.all(mask):
            values = values[:, mask]

        # Only use the training portion to compute statistics
        values = values[: int(values.shape[0] * 0.7)]

        self.norm_data.update(
            {
                f"{key}_mean": np.mean(values),
                f"{key}_std": np.std(values),
                f"{key}_min": np.min(values),
                f"{key}_max": np.max(values),
            }
        )

    def create(self) -> WindpowerContainer:
        np.savez(self.norm_data_path, **self.norm_data)

        return WindpowerContainer(**self.args)

    def load_from_cache(self) -> WindpowerContainer:
        args: Dict[str, Any] = {"name": self.dataset_name}

        norm_data = np.load(self.norm_data_path)
        norm_data = dict(norm_data.items())

        args["norm_data"] = norm_data

        for key in [
            "production",
            "capacity",
            "ratio",
            "wind_u_100m",
            "wind_v_100m",
            "wind_speed_100m",
            "wind_angle_100m",
            "mask",
            "dates",
            "temperature",
        ]:
            path = self.cache_dir / f"{key}.npy"
            value = np.load(path, mmap_mode="r")
            args[key] = value

        for key in [
            "pressure",
            "wind_u_10m",
            "wind_v_10m",
            "wind_speed_10m",
            "wind_angle_10m",
            "lat",
            "lon",
        ]:
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                value = np.load(path, mmap_mode="r")
                args[key] = value

        return WindpowerContainer(**args)


def read_matlab(
    filename: str,
    dataset_name: str,
    force_recompute_cache: bool = False,
) -> WindpowerContainer:

    builder = ContainerBuilder(Path(filename).name, dataset_name)

    if not force_recompute_cache:
        try:
            # raise IOError
            container = builder.load_from_cache()
            print("Loaded from cache")
            return container
        except IOError:
            print("Failed to load cache")
            # Continue to read directly from file

    with h5py.File(filename, "r") as in_f:
        modelData = in_f["modelData"]
        for elem in modelData:
            print(elem)
        assert dataset_name in modelData

        data = modelData[dataset_name]

        print_keys(data)

        try:
            lat = data["EC/properties/lat_all"][:]
            lon = data["EC/properties/lon_all"][:]
        except Exception:
            lat = data["EC/properties/lat_to_keep"][:]
            lon = data["EC/properties/lon_to_keep"][:]

        lat = builder.add("lat", lat)
        lon = builder.add("lon", lon)

        mask = create_mask(lat, lon)
        mask = builder.add("mask", mask)

        dates, weather_slice, electricity_slice = _verify_dates(data)
        dates = builder.add("dates", dates)

        print("Reading electricity ...")

        capacity = np.empty(shape=[0])
        if "capacity/values" in data:
            capacity = data["capacity/values"][0, electricity_slice]
        capacity = builder.add("capacity", capacity)

        production = np.empty(shape=[0])
        if "production/values" in data:
            production = data["production/values"][0, electricity_slice]
        production = builder.add("production", production)

        ratio = np.empty(shape=[0])
        if "capacity/values" in data and "production/values" in data:
            ratio = production / capacity
        ratio = builder.add("ratio", ratio)

        for height in ["10m", "100m"]:
            u_key = f"EC/U_component_of_wind_{height}/values"
            if u_key not in data:
                u_key = f"EC/U_component_of_wind_{height}"

            v_key = f"EC/V_component_of_wind_{height}/values"
            if v_key not in data:
                v_key = f"EC/V_component_of_wind_{height}"

            if u_key not in data:
                print(f"{height} wind data not present")
                continue

            print(f"Reading wind_u_{height} ...")
            wind_u = _flip_data(data[u_key][:, weather_slice], mask)
            wind_u[:, ~mask] = 0
            wind_u = builder.add(f"wind_u_{height}", wind_u)
            print(tm.perf_counter())

            print(f"Reading wind_v_{height} ...")
            wind_v = _flip_data(data[v_key][:, weather_slice], mask)
            wind_v[:, ~mask] = 0
            wind_v = builder.add(f"wind_v_{height}", wind_v)
            print(tm.perf_counter())

            print("Computing speed and angle ...")
            wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
            print(tm.perf_counter())
            wind_angle = np.arctan2(wind_u, wind_v)
            print(tm.perf_counter())

            wind_speed[:, ~mask] = 0
            wind_angle[:, ~mask] = 0

            builder.add_statistics(f"wind_speed_{height}", wind_speed, mask)

            wind_speed = builder.add(f"wind_speed_{height}", wind_speed)
            wind_angle = builder.add(f"wind_angle_{height}", wind_angle)

        print("Reading temperature ...")
        temperature = np.empty(shape=[0, 0])
        if "EC/Temperature_2m/values" in data or "EC/Temperature_2m" in data:
            if "EC/Temperature_2m/values" not in data:
                temperature = _flip_data(data["EC/Temperature_2m"][:, weather_slice], mask)
            else:
                temperature = _flip_data(data["EC/Temperature_2m/values"][:, weather_slice], mask)
            temperature[:, ~mask] = 0
            builder.add_statistics("temperature", temperature, mask)
        temperature = builder.add("temperature", temperature)

        # ERA5 data does not have pressure data
        if "EC/Pressure_reduced_to_MSL/values" in data or "EC/Pressure_reduced_to_MSL" in data:
            print("Reading pressure ...")
            if "EC/Pressure_reduced_to_MSL/values" not in data:
                pressure = _flip_data(data["EC/Pressure_reduced_to_MSL"][:, weather_slice], mask)
            else:
                pressure = _flip_data(data["EC/Pressure_reduced_to_MSL/values"][:, weather_slice], mask)
            pressure[:, ~mask] = 0
            builder.add_statistics("pressure", pressure, mask)
            pressure = builder.add("pressure", pressure)

        return builder.create()


@dataclass
class DatasetReference:
    filepath: str
    dataset: str

    def __post_init__(self):
        filepath = root_dir.joinpath('dataset', self.filepath)
        self.filepath = str(filepath)

    def read_dataset(self) -> WindpowerContainer:
        return read_matlab(self.filepath, self.dataset)

    def __repr__(self) -> str:
        return f"Dataset reference with file {self.filepath} and dataset {self.dataset}"

    def checkpoint_path(self, model: str) -> Path:
        return Path(self.filepath).with_suffix('').joinpath(self.dataset, "checkpoints", model)


if __name__ == "__main__":
    args = basic_argparser.parse_args()

    if args.dataset == "ALL":
        with h5py.File(args.filename, "r") as in_f:
            modelData = in_f["modelData"]
            datasets = list(modelData.keys())
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print(dataset)
        dataset = read_matlab(args.filename, dataset)
        x, y = dataset.load_data()
        print()
