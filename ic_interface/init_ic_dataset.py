import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ic_interface import IcechunkInterface
from write_chunk_refs import write_chunk_refs


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    max_workers = int(os.cpu_count() / 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="The dataset key.", type=str)
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip files that match timestamps in the dataset repo. (optional)",
    )
    parser.add_argument(
        "--start_date",
        help="The earliest timestamp date to include in YYYYMMDD format. (optional)",
        default=None,
    )
    parser.add_argument(
        "--end_date",
        help="The latest timestamp date to include in YYYYMMDD format. (optional)",
        default=None,
    )
    parser.add_argument(
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

    file_handler = logging.FileHandler(ic_interface.log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[file_handler],
    )
    logger = logging.getLogger("init_ic_dataset")
    logger.setLevel(logging.INFO)

    ic_interface = IcechunkInterface(args.key)
    initialized = ic_interface.initialized

    nc_file_info = ic_interface.get_nc_file_info(
        args.nc_files, args.start_date, args.end_date, args.skip_existing
    )

    if not ic_interface.initialized:
        ic_interface.initialize_repo_arrays(nc_file_info)

    if "timestamp" in nc_file_info.columns:

        time_chunk_map = ic_interface.get_time_chunk_map()
        nc_file_info["time_chunk_index"] = nc_file_info["timestamp"].apply(
            time_chunk_map.get
        )

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
            session.commit(f"Wrote timestamps {ts_group[0]} - {ts_group[-1]}")
            print(f"Wrote timestamps {ts_group[0]} - {ts_group[-1]}")
