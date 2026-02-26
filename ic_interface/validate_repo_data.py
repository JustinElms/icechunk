import argparse
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import QueueListener

import numpy as np
import xarray as xr
from icechunk import Session
from tqdm import tqdm

from ic_interface import IcechunkInterface


def worker_configurer(queue: mp.Queue, log_path: str):
    """Configuration for the worker processes."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)


def listener_configurer(log_path: str):
    """Configuration for the main process listener."""
    root = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.setLevel(logging.INFO)


def validate_repo_data(session: Session, nc_file: str, drop_vars: list | None) -> None:
    ic_ds = xr.open_zarr(
        session.store, consolidated=False, decode_times=False, decode_cf=False
    )
    with xr.open_dataset(
        nc_file, drop_variables=drop_vars, decode_times=False, decode_cf=False
    ) as ds:
        ic_subset = ic_ds.sel(time=ds.time)
        for var in ds.data_vars:
            try:
                xr.testing.assert_equal(ds[var], ic_subset[var])
            except AssertionError as e:
                logging.critical("Validation failed for file \n    " f"{str(nc_file)}")
                logging.critical(e)


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    max_workers = int(os.cpu_count() / 2)

    parser = argparse.ArgumentParser(
        description=(
            "Validate repository data by comparing a random subset of NetCDF files to "
            "repository data. This essentially ensures that the virtual chunk "
            "references are correctly associated with the correct NetCDF file and "
            "timestamp."
        )
    )
    parser.add_argument("key", help="The dataset key.", type=str)
    parser.add_argument(
        "-c",
        "--coverage",
        help="The percentage of NetCDF files to test. Accepts integers from 0 to 100.",
        default=5,
        type=int,
    )

    args = parser.parse_args()
    ic_interface = IcechunkInterface(args.key)
    ic_interface.logger.info("\n***\n Starting validate_repo_data.py script. \n***")
    dataset_config = ic_interface.dataset_config
    drop_vars = dataset_config.get("drop_vars")

    log_queue = mp.Manager().Queue(-1)
    listener = QueueListener(log_queue, listener_configurer(ic_interface.log_path))
    listener.start()

    nc_files = np.array(list(ic_interface.get_virtual_file_list()))
    n_files = len(nc_files)

    rng = np.random.default_rng()
    sample_size = int(n_files * args.coverage / 100)
    if sample_size < 1:
        sample_size = 1
    sample_idx = rng.choice(np.arange(n_files), size=sample_size)
    sample_files = nc_files[sample_idx]

    sample_groups = np.array_split(sample_files, np.ceil(sample_size / max_workers / 2))

    print("Validating repository NetCDF data.")
    with tqdm(total=sample_size) as pbar:
        for sample_group in sample_groups:
            session = ic_interface.repo.readonly_session("main")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=worker_configurer,
                initargs=(
                    log_queue,
                    ic_interface.log_path,
                ),
            ) as executor:

                futures = [
                    executor.submit(
                        validate_repo_data,
                        session=session,
                        nc_file=nc_file,
                        drop_vars=drop_vars,
                    )
                    for nc_file in sample_group
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        ic_interface.logger.error(f"Exception raise in validation: {e}")
                        ic_interface.logger.error(
                            f"Error in files:\n    {'\n    '.join(sample_group)}"
                        )
            pbar.update(len(sample_group))

    listener.stop()
    ic_interface.logger.info("\n***\n Finished validate_repo_data.py script. \n***")
