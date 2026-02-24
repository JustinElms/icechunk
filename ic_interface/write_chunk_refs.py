from icechunk import (
    ForkSession,
    Session,
    VirtualChunkSpec,
)
from obstore.store import LocalStore
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry

COORDINATE_VARIABLES = [
    "nav_lat_u",
    "nav_lon_u",
    "nav_lat_v",
    "nav_lon_v",
    "nav_lat",
    "nav_lon",
    "latitude_u",
    "longitude_u",
    "latitude_v",
    "longitude_v",
    "latitude",
    "longitude",
    "lat",
    "lon",
]


def write_chunk_refs(
    *,
    session: Session | ForkSession = None,
    dataset_config: dict = {},
    nc_files: list = [],
    time_chunk_map: dict = {},
) -> None:
    """
    Writes virtual chunk references from the given NetCDF files to the repository using
    the provided session object. Commits must be handled externally using the returned
    session so that this function can be used with muliprocessing.

    args:
        session: An Icechunk Session or ForSession object.
        dataset_config: A dict containing the dataset configuration.
        nc_files: A Pandas Dataframe containing NetCDF files and metadata.
        time_chunk_map: A mapping of timestamps to their related chunks.

    returns:
        session: The Icechunk Session or ForSession object.
    """

    drop_vars = dataset_config.get("drop_vars")
    parser_type = dataset_config.get("parser_type")

    match parser_type:
        case "nc3":
            parser = NetCDF3Parser()
        case _:
            parser = HDFParser()

    file_urls = [f"file://{file}" for file in nc_files]
    store = LocalStore(prefix="/data/")
    registry = ObjectStoreRegistry({file_url: store for file_url in file_urls})

    with open_virtual_mfdataset(
        urls=file_urls,
        parser=parser,
        registry=registry,
        drop_variables=drop_vars,
        compat="override",
        decode_times=False,
    ) as vds:

        # check if any coordinates are erroniously included in data variables
        coord_data_vars = [v for v in vds.data_vars if v in COORDINATE_VARIABLES]
        for cdv in coord_data_vars:
            vds = vds.set_coords(cdv)

        for variable in vds.data_vars:
            chunks = []
            for manifest_idx, spec in vds[variable].data.manifest.dict().items():
                index = list(map(int, manifest_idx.split(".")))
                time_dim_idx = vds[variable].dims.index("time")
                timestamp = vds.time.data[index[time_dim_idx]]
                index[time_dim_idx] = int(time_chunk_map[timestamp])
                chunks.append(VirtualChunkSpec(index, *spec.values()))
            session.store.set_virtual_refs(variable, chunks)

    return session
