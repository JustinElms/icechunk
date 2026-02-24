import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ic_interface import IcechunkInterface
from write_chunk_refs import write_chunk_refs


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    max_workers = int(os.cpu_count() / 2)

    parser = argparse.ArgumentParser(
        description=(
            "Initialze and/or append NetCDF data to the dataset Icechunk repository."
            " Only one of -/dir/--nc_dir or -nc/--nc_files should be provided. If "
            "neither are provided then all NetCDF files in the datastore will be added "
            "to the repository."
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
    file_args = parser.add_mutually_exclusive_group(required=False)
    file_args.add_argument(
        "-dir",
        "--nc_dir",
        help=(
            "The path to the directory containing the NetCDF or other format data "
            "files to append. (optional)"
        ),
        default=None,
        type=str,
    )
    file_args.add_argument(
        "-nc",
        "--nc_files",
        help=(
            "The path to the NetCDF or other format data files to append. (optional)"
        ),
        nargs="*",
        default=[],
        type=str,
    )

    args = parser.parse_args()
    ic_interface = IcechunkInterface(args.key)
    dataset_config = ic_interface.dataset_config
    initialized = ic_interface.initialized
    ic_interface.logger.info("\n***\n Startting add_nc_data.py script. \n***")

    nc_files = args.nc_files
    if args.nc_dir:
        nc_files = list(Path(args.nc_dir).rglob("*.nc"))

    nc_file_info = ic_interface.get_nc_file_info(
        nc_files, args.start_date, args.end_date, args.skip_existing
    )

    branch = "main"
    if not ic_interface.initialized:
        ic_interface.initialize_repo_arrays(nc_file_info)
    else:
        branch = f"append_{datetime.now().isoformat()}"
        if branch in ic_interface.repo.list_branches():
            ic_interface.delete_branch(branch)
        ic_interface.create_branch(branch)

    timestamps = nc_file_info.timestamp.explode("timestamp").unique()
    time_chunk_map = {ts: None for ts in timestamps}
    while not any(time_chunk_map.values()):
        time_chunk_map = {**time_chunk_map, **ic_interface.get_time_chunk_map(branch)}
        new_timestamps = [
            ts for ts in time_chunk_map.keys() if time_chunk_map[ts] is None
        ]

        if len(new_timestamps) == 0:
            continue
        ic_interface.append_timestamps(new_timestamps, branch)

    nc_file_info["time_chunk_map"] = nc_file_info.timestamp.apply(
        lambda ts: {t: time_chunk_map[t] for t in ts}
    )

    idx_groups = np.array_split(
        nc_file_info.index, len(nc_file_info.index) / max_workers / 2
    )

    with tqdm(total=len(nc_file_info)) as pbar:
        for idx_group in idx_groups:
            group_df = nc_file_info.loc[idx_group]
            group_ts = np.concat(group_df.timestamp.to_numpy())
            try:
                session = ic_interface.repo.writable_session(branch)
                nc_files = [str(f) for files in group_df.file.values for f in files]
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                ) as executor:
                    futures = []
                    for idx, row in group_df.iterrows():
                        futures.append(
                            executor.submit(
                                write_chunk_refs,
                                session=session.fork(),
                                dataset_config=dataset_config,
                                nc_files=row.file,
                                time_chunk_map=row.time_chunk_map,
                            )
                        )

                    remote_sessions = [f.result() for f in futures]

                session.merge(*remote_sessions)
                commit_msg = (
                    f"Wrote {len(group_ts)} timestamps "
                    f"{int(group_ts.min())} - {int(group_ts.max())}"
                )
                session.commit(commit_msg)
                ic_interface.logger.info(commit_msg)
                ic_interface.logger.info(f"Files added:\n    {'\n    '.join(nc_files)}")
            except Exception as e:
                ic_interface.logger.error(e)
                ic_interface.logger.info(
                    f"Error in files:\n    {'\n    '.join(nc_files)}"
                )
            pbar.update(len(idx_group))

    if branch != "main":
        snapshot_id = ic_interface.repo.lookup_branch(branch)
        ic_interface.set_branch_ref("main", snapshot_id)
        ic_interface.delete_branch(branch)
