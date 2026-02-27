import glob
import json
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from obstore.store import LocalStore
from tqdm import tqdm
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry

from icechunk import (
    ManifestConfig,
    ManifestSplitCondition,
    ManifestSplittingConfig,
    ManifestSplitDimCondition,
    Repository,
    RepositoryConfig,
    VirtualChunkContainer,
    local_filesystem_store,
    s3_storage,
)

warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=zarr.errors.UnstableSpecificationWarning)

# TODO:
# - Add functionality to remove data (e.g. timestamps older than 2 yrs)
# - Add function to clean up datastore (remove unused files)
# - Add error handling and other user feedback
# - Add function to spot check data
# - Add verification of dataset and ic config file - raise errors for bad or missing
#   values
# - Add option in config for all variables in one NC file
# - Add option in config for multiple timestamps in one NC file


class IcechunkInterface:

    def __init__(self, dataset_key: str) -> None:
        self.ic_config = self.__read_config("ic_config.json")
        self.dataset_config = self.__read_config(
            f"{self.ic_config.get("dataset_configs_dir")}/{dataset_key}.json"
        )

        self.__configure_logger(dataset_key)

        storage_config = self.__get_storage(dataset_key)

        if not Repository.exists(storage_config):
            split_config = ManifestSplittingConfig.from_dict(
                {
                    ManifestSplitCondition.AnyArray(): {
                        ManifestSplitDimCondition.DimensionName("time"): 365
                    }
                }
            )
            repo_config = RepositoryConfig(
                manifest=ManifestConfig(splitting=split_config),
            )
            virtual_container = VirtualChunkContainer(
                url_prefix="file:///data/", store=local_filesystem_store(path="/data/")
            )
            repo_config = RepositoryConfig()
            repo_config.set_virtual_chunk_container(virtual_container)
            self.repo = Repository.create(storage_config, config=repo_config)
            self.repo.save_config()
            self.logger.info("\n***\n Created repository \n***")
        else:
            self.repo = Repository.open(
                storage_config, authorize_virtual_chunk_access={"file:///data/": None}
            )

        self.initialized = len(list(self.repo.ancestry(branch="main"))) > 1

    def __get_storage(self, dataset_key) -> s3_storage:
        s3_config = self.ic_config.get("s3_config")
        return s3_storage(
            bucket=s3_config.get("bucket"),
            prefix=dataset_key,
            region="us-east-1",
            access_key_id=s3_config.get("user"),
            secret_access_key=s3_config.get("password"),
            endpoint_url=s3_config.get("url"),
            allow_http=True,
            force_path_style=True,
        )

    def __read_config(self, path: str) -> dict:
        with open(path, "r") as f:
            config = json.load(f)
        return config

    def __configure_logger(self, dataset_key: str) -> None:

        log_dir = self.ic_config.get("log_directory")

        os.makedirs(log_dir, exist_ok=True)

        self.log_path = f"{log_dir}{dataset_key}.log"

        file_handler = logging.FileHandler(self.log_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            handlers=[file_handler],
        )
        self.logger = logging.getLogger("ic_interface")

    def get_repo_timestamps(self, branch: str = "main") -> np.ndarray:
        session = self.repo.readonly_session(branch)
        try:
            with xr.open_zarr(
                session.store, consolidated=False, decode_times=False
            ) as ds:
                timestamps = ds.time.values
        except zarr.errors.GroupNotFoundError:
            timestamps = []
        return timestamps

    def get_time_chunk_map(self, branch: str = "main") -> np.ndarray:
        timestamps = self.get_repo_timestamps(branch)

        session = self.repo.readonly_session(branch)
        with xr.open_zarr(session.store, consolidated=False, decode_times=False) as ds:
            timestamps = ds.time.data

        return {int(ts): i for i, ts in enumerate(timestamps)}

    def create_branch(self, branch_id: str, from_init: bool = False) -> None:
        """ """
        branches = list(self.repo.list_branches())
        if branch_id in branches:
            return

        ancestry_list = list(self.repo.ancestry(branch="main"))
        snapshot = ancestry_list[0]
        if from_init:
            snapshot = ancestry_list[-1]

        self.repo.create_branch(branch_id, snapshot_id=snapshot.id)

    def delete_branch(self, branch_id: str) -> None:
        """ """
        self.repo.delete_branch(branch_id)

    def set_branch_ref(self, branch: str, snapshot_id: str) -> None:
        """ """
        self.repo.reset_branch(branch, snapshot_id=snapshot_id)

    def get_virtual_file_list(self) -> set:
        """
        Gets a set of all netcdf files used in virtual chunk references from the repo.
        """
        session = self.repo.readonly_session("main")
        chunks = session.all_virtual_chunk_locations()

        file_list = [c.replace("file://", "") for c in chunks]

        return set(file_list)

    def get_timestamp_index(self, timestamp: int, repo_timestamps: list) -> int | None:

        time_chunk_index = None

        idx = np.argwhere(repo_timestamps == timestamp).flatten()
        if len(idx) > 0:
            time_chunk_index = int(idx[0])

        return time_chunk_index

    @staticmethod
    def nc_file_var_ts(nc_file: str, drop_vars=None) -> list:
        """
        Extracts timestamps and variable names from NetCDF files.

        Args:
            row: a pandas series containing NetCDF files

        Returns:
            tuple: a list of variables and timestamps.
        """
        with xr.open_dataset(
            nc_file, drop_variables=drop_vars, decode_times=False
        ) as ds:
            return [nc_file, list(ds.data_vars), ds.time.data]

    def get_nc_file_info(
        self,
        nc_files: list | None = [],
        start_date: str | None = None,
        end_date: str | None = None,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        """
        Generate a dataframe of forcast metadata and file paths for a list of NetCDF
        files or for each file in the datastore. Can be filtered to a specific period
        with optional start_date and end_date args.

        Args:
            nc_files: a list of NetCDF files to process. This will process all files in
                    the datastore if not provided.
            start_date: the date of the initial forecast to be added to the repo.
            end_date: the date of the last forecast to be added to the repo.
            skip_existing: ignore files that are aready referenced by the repo.

        Returns:
            list: a list of inputs that can be passed to write_timestamp formatted
                for multiprocessing
        """
        path_template = self.dataset_config["data_path_template"]
        path_filters = self.dataset_config.get("path_filters")
        variables = self.dataset_config.get("variables")
        drop_vars = self.dataset_config.get("drop_vars")

        if len(nc_files) == 0:
            nc_files = glob.glob(path_template)

        if path_filters:
            nc_files = [
                f for f in nc_files if all([k not in str(f) for k in path_filters])
            ]

        if skip_existing:
            current_nc_files = self.get_virtual_file_list()
            nc_files = [f for f in nc_files if str(f) not in current_nc_files]

        print("Scanning NetCDF data.")
        file_data = []
        max_workers = int(os.cpu_count() / 2)
        with tqdm(total=len(nc_files)) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.nc_file_var_ts, nc_file=nc_file, drop_vars=drop_vars
                    )
                    for nc_file in nc_files
                ]

                for future in as_completed(futures):
                    try:
                        file_data.append(future.result())
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(
                            f"Could not extract NetCDF variable data. \n    {e}"
                        )

        nc_info = pd.DataFrame(file_data, columns=["file", "variable", "timestamp"])
        nc_info = nc_info.explode("variable").explode("timestamp", ignore_index=True)
        nc_info = nc_info.sort_values(by=["timestamp", "variable", "file"]).reset_index(
            drop=True
        )
        nc_info.drop_duplicates(
            subset=["timestamp", "variable"], keep="last", inplace=True
        )

        # filter forecast data by given dates
        if start_date:
            start_timestamp = cftime.date2num(
                datetime.strptime(start_date, "%Y%m%d"),
                self.dataset_config["time_dim_units"],
            )
            nc_info = nc_info[nc_info["datetime"] >= start_timestamp]

        if end_date:
            end_timestamp = cftime.date2num(
                datetime.strptime(end_date, "%Y%m%d"),
                self.dataset_config["time_dim_units"],
            )
            nc_info = nc_info[nc_info["datetime"] <= end_timestamp]

        # ensure each timestamp includes all variables
        ts_file_counts = nc_info.groupby(["timestamp"])["file"].transform("size")
        if variables:
            nc_info = nc_info[(ts_file_counts == len(variables))]
        else:
            nc_info = nc_info[(ts_file_counts == ts_file_counts.max())]

        if self.dataset_config.get("latest_only"):
            nc_info = nc_info.loc[nc_info["timestamp"] == nc_info["timestamp"].max()]

        # group by file and timestamp
        nc_info = nc_info.groupby("file").agg(list).reset_index()
        nc_info["timestamp"] = nc_info["timestamp"].map(lambda ts: tuple(np.unique(ts)))
        nc_info = nc_info.groupby("timestamp").agg(list).reset_index()
        nc_info["variable"] = nc_info["variable"].apply(
            lambda v: np.unique(np.concat(v))
        )

        return nc_info

    def append_timestamps(self, timestamps: int | list, branch: str = "main") -> None:
        """ """
        if not hasattr(timestamps, "__len__"):
            timestamps = [timestamps]

        session = self.repo.writable_session(branch)
        dataset = zarr.open_group(session.store)
        for array_key in dataset:
            array = dataset[array_key]
            if array_key == "time":
                array.append(timestamps, axis=0)
            else:
                dims = array.metadata.dimension_names
                try:
                    time_idx = dims.index("time")
                except ValueError:
                    continue
                shape = list(array.shape)
                shape[time_idx] = shape[time_idx] + len(timestamps)
                array.resize(shape)
        commit_msg = (
            "Extended time dimension with timestamp(s) "
            f"{timestamps[0]} - {timestamps[-1]} ({len(timestamps)})."
        )
        session.commit(commit_msg)
        self.logger.info(commit_msg)

    def initialize_repo_arrays(
        self, nc_file_info: pd.DataFrame, branch: str = "main"
    ) -> None:
        """
        Initializes dataset coordinate and variable arrays in repository.

        Args:
            nc_file_info: A DataFrame containing meta data for the NC files being
                            written.
        Returns:
            None
        """

        timestamps = nc_file_info.timestamp.explode("timestamp").unique()

        ts_mask = nc_file_info.timestamp.apply(lambda ts: timestamps.min() in ts)
        ts_data = nc_file_info[ts_mask]

        nc_files = ts_data["file"].values[0]
        parser_type = self.dataset_config.get("parser")
        drop_vars = self.dataset_config.get("drop_vars")

        match parser_type:
            case "nc3":
                parser = NetCDF3Parser()
            case _:
                parser = HDFParser()

        file_urls = [f"file://{file}" for file in nc_files]
        store = LocalStore(prefix="/data/")
        registry = ObjectStoreRegistry({file_url: store for file_url in file_urls})

        session = self.repo.writable_session(branch)
        with open_virtual_mfdataset(
            urls=file_urls,
            parser=parser,
            registry=registry,
            drop_variables=drop_vars,
            compat="override",
            decode_times=False,
        ) as vds:
            vds.vz.to_icechunk(session.store)
            timestamps = [ts for ts in timestamps if ts not in vds.time.data]
            commit_msg = (
                f"Initialized data arrays with timestamp(s) {vds.time.data.astype(int)}"
            )

        session.commit(commit_msg)
        self.logger.info(commit_msg)

        if len(timestamps) > 0:
            self.append_timestamps(timestamps)
