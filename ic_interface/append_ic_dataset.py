import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from ic_interface import IcechunkInterface
from write_chunk_refs import write_chunk_refs


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    max_workers = int(os.cpu_count() / 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="The dataset key.", type=str)
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

    branch = f"append_{datetime.now().isoformat()}"
    if branch in ic_interface.repo.list_branches():
        ic_interface.delete_branch(branch)
    ic_interface.create_branch(branch)

    nc_file_info = ic_interface.get_nc_file_info(args.nc_files)

    if "timestamp" not in nc_file_info.columns:
        raise KeyError(
            "Timestamp values missing. Exiting."
        )  # provide better feedback than this

    nc_file_info["time_chunk_index"] = None
    while any(nc_file_info["time_chunk_index"].isnull()):
        time_chunk_map = ic_interface.get_time_chunk_map(branch)
        nc_file_info["time_chunk_index"] = nc_file_info["timestamp"].apply(
            time_chunk_map.get
        )
        new_timestamps = nc_file_info.loc[
            nc_file_info["time_chunk_index"].isnull(), "timestamp"
        ].values
        if len(new_timestamps) == 0:
            continue
        ic_interface.append_timestamps(new_timestamps, branch)

    session = ic_interface.repo.writable_session(branch)
    with ProcessPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        futures = []
        timestamps = nc_file_info.timestamp.unique()
        for idx, row in nc_file_info.iterrows():
            futures.append(
                executor.submit(
                    write_chunk_refs,
                    session=session.fork(),
                    dataset_config=dataset_config,
                    nc_files=row.file,
                    time_chunk_index=row.time_chunk_index,
                )
            )

        remote_sessions = [f.result() for f in futures]

    session.merge(*remote_sessions)
    commit_msg = (
        f"Wrote {len(nc_file_info)} timestamps "
        f"{int(nc_file_info.timestamp.min())} - {int(nc_file_info.timestamp.max())}"
    )
    session.commit(commit_msg)
    logger.info(commit_msg)

    snapshot_id = ic_interface.repo.lookup_branch(branch)
    ic_interface.set_branch_ref("main", snapshot_id)
    ic_interface.delete_branch(branch)
