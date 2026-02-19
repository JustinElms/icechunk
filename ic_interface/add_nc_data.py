import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ic_interface import IcechunkInterface
from write_chunk_refs import write_chunk_refs


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    max_workers = int(os.cpu_count() / 2)

    parser = argparse.ArgumentParser(
        description=(
            "Initialze and/or append data to the dataset Icechunk repository."
            " One of -/dir/--nc_dir or -nc/--nc_files must be provided."
        )
    )
    parser.add_argument("key", help="The dataset key.", type=str)
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip any files referenced in the dataset repo. (optional)",
    )
    parser.add_argument(
        "--start_date",
        help="The earliest forecast date to include in YYYYMMDD format. (optional)",
        default=None,
    )
    parser.add_argument(
        "--end_date",
        help="The latest forecast date to include in YYYYMMDD format. (optional)",
        default=None,
    )
    file_args = parser.add_mutually_exclusive_group(required=True)
    file_args.add_argument(
        "-dir",
        "--nc_dir",
        help=(
            "The path to directory containing the NetCDF or other format data files to"
            " append."
        ),
        default=None,
        type=str,
    )    
    file_args.add_argument(
        "-nc",
        "--nc_files",
        help=(
            "The path to the NetCDF or other format data files to append. If not"
            "provided this script will  add all files in the datastore to the "
            "repository. (optional)"
        ),
        nargs="*",
        default=[],
        type=str,
    )

    args = parser.parse_args()
    ic_interface = IcechunkInterface(args.key)
    dataset_config = ic_interface.dataset_config
    initialized = ic_interface.initialized

    file_handler = logging.FileHandler(ic_interface.log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[file_handler],
    )
    logger = logging.getLogger("init_ic_dataset")
    logger.setLevel(logging.INFO)

    nc_files = args.nc_files
    if args.nc_dir:
        nc_files = list(Path(args.nc_dir).rglob("*.nc"))

    nc_file_info = ic_interface.get_nc_file_info(
        nc_files, args.start_date, args.end_date, args.skip_existing
    )

    if not ic_interface.initialized:
        ic_interface.initialize_repo_arrays(nc_file_info)

    if "timestamp" in nc_file_info.columns:
        nc_file_info["time_chunk_index"] = None
        while any(nc_file_info["time_chunk_index"].isnull()):
            time_chunk_map = ic_interface.get_time_chunk_map()
            nc_file_info["time_chunk_index"] = nc_file_info["timestamp"].apply(
                time_chunk_map.get
            )
            new_timestamps = nc_file_info.loc[
                nc_file_info["time_chunk_index"].isnull(), "timestamp"
            ].values
            if len(new_timestamps) == 0:
                continue
            ic_interface.append_timestamps(new_timestamps)

        timestamps = sorted(nc_file_info.timestamp.unique())
        ts_groups = np.array_split(timestamps, len(timestamps) / max_workers / 2)

        for ts_group in ts_groups:

            session = ic_interface.repo.writable_session("main")
            with ProcessPoolExecutor(
                max_workers=max_workers,
            ) as executor:
                futures = []
                for timestamp in ts_group:
                    ts_data = nc_file_info.loc[nc_file_info.timestamp == timestamp]
                    nc_files = ts_data.file.values[0]
                    time_chunk_index = ts_data.time_chunk_index.values[0]

                    futures.append(
                        executor.submit(
                            write_chunk_refs,
                            session=session.fork(),
                            dataset_config=dataset_config,
                            nc_files=nc_files,
                            time_chunk_index=time_chunk_index,
                        )
                    )

                remote_sessions = [f.result() for f in futures]

            session.merge(*remote_sessions)
            commit_msg = (
                f"Wrote {len(ts_group)} timestamps {ts_group[0]} - {ts_group[-1]}"
            )
            session.commit(commit_msg)
            logger.info(commit_msg)
