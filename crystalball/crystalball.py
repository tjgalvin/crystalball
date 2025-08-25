#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import warnings
from contextlib import ExitStack
from importlib.metadata import version
from time import time
from typing import Literal, Any

import dask
import dask.array as da
from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import wsclean_predict
from africanus.util.dask_util import EstimatingProgressBar
from dask.array import PerformanceWarning
import dask.delayed
from dask.distributed import Client, progress
from daskms import xds_from_ms, xds_from_table, xds_to_table, dataset
from distributed import LocalCluster
from loguru import logger as log

import crystalball.logger_init  # noqa
from crystalball.budget import get_budget
from crystalball.filtering import filter_datasets, select_field_id
from crystalball.ms import ms_preprocess
from crystalball.region import load_regions
from crystalball.wsclean import WSCleanModel, import_from_wsclean


def support_tables(ms: str, tables: list[str], compute: bool=True) -> dict[str, Any]:
    """Load in the measurement set tables as xarray DataSets

    Args:
        ms (str): The measurement set to load the tables from
        tables (list[str]): The specific tables to load
        compute (bool, optional): Whether the tables should be evaluated on the fly or simply returned. Defaults to True.

    Returns:
        dict[str, Any]: The set of tables, where the key is their name and value is the xrray dataset
    """
    log.info("Loading the support tabls")
    def _loader(dataset: Any) -> Any:
        """Tricksey trick"""
        if compute:
            return dataset.compute(priority=9999)
        return dataset
    
    open_tables = {}
    for t in tables:
        log.info(f"Loading {t=}")
        dataset = xds_from_table(
                    "::".join((ms, t)),
                    group_cols="__row__"
                )
        open_tables[t] = [_loader(ds) for ds in dataset]
    
    log.info(f"{len(tables)} loaded with {compute=}")
    return open_tables
    
    # return {t: [
    #     _loader(ds) 
    #     for ds in xds_from_table(
    #                 "::".join((ms, t)),
    #                 group_cols="__row__")
    #     ]
    #     for t in tables
    # }


def fill_correlations(vis, pol):
    """
    Expands single correlation produced by wsclean_predict to the
    full set of correlations.

    Parameters
    ----------
    vis : :class:`dask.array.Array`
        dask array of visibilities of shape :code:`(row, chan, 1)`
    pol : :class:`xarray.Dataset`
        MS Polarisation dataset.

    Returns
    -------
    vis : :class:`dask.array.Array`
        dask array of visibilities of shape :code:`(row, chan, corr)`
    """

    corrs = pol.NUM_CORR.data[0]

    assert vis.ndim == 3

    if corrs == 1:
        return vis
    elif corrs == 2:
        vis = da.concatenate([vis, vis], axis=2)
        return vis.rechunk({2: corrs})
    elif corrs == 4:
        zeros = da.zeros_like(vis)
        vis = da.concatenate([vis.MODEL_DATA.data, zeros, zeros, vis.MOEL_DATA.data], axis=2)
        return vis.rechunk({2: corrs})
    else:
        raise ValueError("MS Correlations %d not in (1, 2, 4)" % corrs)


def source_model_to_dask(source_model: str, chunks: int) -> WSCleanModel:
    """Convert the wsclean text file source list (the BB style) to a set of dask arrays

    Args:
        source_model (str): The path to the wsclean text file
        chunks (int): The desired chunk size to split the source model across

    Returns:
        WSCleanModel: The columns as a dask array in an appropriate chunked form
    """
    # Create chunked dask arrays from wsclean model arrays
    sm = source_model

    radec_chunks = (chunks,) + sm.radec.shape[1:]
    spi_chunks = (chunks,) + sm.spi.shape[1:]
    gauss_chunks = (chunks,) + sm.gauss_shape.shape[1:]

    return WSCleanModel(
        da.from_array(sm.source_type, chunks=chunks),
        da.from_array(sm.radec, chunks=radec_chunks),
        da.from_array(sm.flux, chunks=chunks),
        da.from_array(sm.spi, chunks=spi_chunks),
        da.from_array(sm.ref_freq, chunks=chunks),
        da.from_array(sm.log_poly, chunks=chunks),
        da.from_array(sm.gauss_shape, chunks=gauss_chunks)
    )

def construct_client(
    exitstack: ExitStack,
    num_workers: int = 0,
    scheduler: Literal["threads", "distributed"] = "threads",
    address: str | None = None,
    workers: int = 1,
) -> Client:
    """Create a dask client based on inputs. By default the client is a thread pool. 

    Args:
        exitstack (ExitStack): An exit stack to perform removal / shutdown
        num_workers (int, optional): Desired number of workers. Defaults to 0.
        scheduler (Literal[&quot;threads&quot;, &quot;distributed&quot;], optional): The type of client to start. Defaults to "threads".
        address (str | None, optional): Connection string to existing distributed schedular. Defaults to None.
        workers (int, optional): Number of workers. Defaults to 1.

    Raises:
        ValueError: Unknown schedular type

    Returns:
        Client: Dask client
    """

    if scheduler not in ["threads", "distributed"]:
        raise ValueError(f"Unknown scheduler type: {scheduler}")

    # Following Quartical's constuction of a dask client
    if scheduler == "threads":
        log.info("Initializing dask client using threads scheduler.")
        exitstack.enter_context(dask.config.set(num_workers=num_workers))
        return None

    address = address or os.environ.get("DASK_SCHEDULER_ADDRESS")
    if address:
        log.info(
            f"Initializing distributed client using scheduler address: "
            f"{address}"
        )
        client = exitstack.enter_context(Client(address))

    else:
        log.info("Initializing distributed client using LocalCluster.")
        cluster = LocalCluster(
            processes=workers > 1,
            n_workers=workers,
            threads_per_worker=num_workers,
            memory_limit=0,
        )
        cluster = exitstack.enter_context(cluster)
        client = exitstack.enter_context(Client(cluster))

    client.wait_for_workers(workers)

    log.info("Distributed client sucessfully initialized.")
    return client

def compute_chunk_sizes(
    ms: str, 
    source_model_xds: list[dataset.Dataset],
    num_workers: int,
    memory_fraction: float,
    ms_datatype: Any,
    ms_rows: int,
    client: Client   
) -> tuple[int, int]:
    """Compute chunk sizes for the data and source. This will load the data
    to acquire its dimensionality.

    Args:
        ms (str): The measurement set to consider
        source_model_xds (list[dataset.Dataset]): The loaded source model
        num_workers (int): The number of workers to use
        memory_fraction (float): The appropriate memory fraction each worker should target
        ms_datatype (Any): The size of the MS data
        ms_rows (int): Number of rows in the measurement set
        client (Client): The dask client being used

    Returns:
        tuple[int, int]: The row chunk size and model chunk size
    """
    # Get the support tables
    tables = support_tables(
        ms=ms, 
        tables=["SPECTRAL_WINDOW", "POLARIZATION"],
    )

    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    
    max_num_chan = max([ss.NUM_CHAN.data[0] for ss in spw_ds])
    max_num_corr = max([ss.NUM_CORR.data[0] for ss in pol_ds])

    # Perform resource budgeting
    nsources = source_model_xds.source_type.shape[0]
    row_chunks, model_chunks = get_budget(
        nr_sources=nsources,
        nr_rows=ms_rows, 
        nr_chans=max_num_chan, 
        nr_corrs=max_num_corr, 
        data_type=ms_datatype, 
        num_workers=num_workers,
        memory_fraction=memory_fraction,
        client=client,
    )
    return row_chunks, model_chunks


def create_predict_graph(
        ms: str,
        source_model: WSCleanModel | str,
        output_column: str = "MODEL_DATA",
        field: str | None = None,
        row_chunks: int = 0,
        model_chunks: int = 0,
        client: Client = None
) -> list[dataset.Dataset]:
    """Create the dask work graph for execution

    Args:
        ms (str): The measurement set to create the model data for
        source_model (WSCleanModel): The wsclean model
        output_column (str, optional): Output column to write to. Defaults to "MODEL_DATA".
        field (str | None, optional): The field to insert data for. Defaults to None.
        row_chunks (int, optional): How many rows to process in a chunk. Defaults to 0.
        model_chunks (int, optional): How many model components to process in a chynk. Defaults to 0.
        within (str | None, optional): Region limiting prediction for. Defaults to None.

    Returns:
        list[dataset.Dataset]: The dask delayed objects to execute
    """
    pkg_version = version("crystalball")
    log.info(f"Crystalball version {pkg_version}")
    
    if isinstance(source_model, str):
        source_model = import_from_wsclean(
                source_model
            )
    
    log.info("Converting source model to dask arrays")
    source_model = source_model_to_dask(source_model, model_chunks)

    tables = support_tables(
        ms=ms, 
        tables=["FIELD", "DATA_DESCRIPTION", "SPECTRAL_WINDOW", "POLARIZATION"],
        compute=False
    )
    
    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]
    
    # List of write operations
    writes = []

    datasets = xds_from_ms(
        ms,
        columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
        group_cols=["FIELD_ID", "DATA_DESC_ID"],
        chunks={"row": row_chunks}
    )

    field_id = select_field_id(field_ds, field)

    log.info(f"The {field_id=}")
    for xds in filter_datasets(datasets, field_id):
        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]
        frequency = spw.CHAN_FREQ.data[0]

        lm = radec_to_lm(source_model.radec, field.PHASE_DIR.data[0][0])

        with warnings.catch_warnings():
            # Ignore dask chunk warnings emitted when going from 1D
            # inputs to a 2D space of chunks
            warnings.simplefilter('ignore', category=PerformanceWarning)
            vis = wsclean_predict(
                xds.UVW.data,
                lm,
                source_model.source_type,
                source_model.flux,
                source_model.spi,
                source_model.log_poly,
                source_model.ref_freq,
                source_model.gauss_shape,
                frequency
            )

        vis = fill_correlations(vis, pol)

        log.info('Field {0} DDID {1:d} rows {2} chans {3} corrs {4}',
                 field.NAME.values[0],
                 xds.DATA_DESC_ID,
                 vis.shape[0], vis.shape[1], vis.shape[2])

        # Assign visibilities to MODEL_DATA array on the dataset
        xds = xds.assign(
            **{output_column: (("row", "chan", "corr"), vis)}
        )
        # Create a write to the table
        log.info("Creating writes object")
        write = xds_to_table(xds, ms, [output_column])
        # Add to the list of writes
        writes.append(write)

    log.info("Work graphy constructed!")
    return writes

def predict(
        ms: str,
        sky_model: str = "sky-model.txt",
        output_column: str = "MODEL_DATA",
        field: str | None = None,
        row_chunks: int = 0,
        model_chunks: int = 0,
        within: str | None = None,
        points_only: bool = False,
        num_sources: int = 0,
        num_workers: int = 0,
        memory_fraction: float = 0.1,
        client: Client | None = None,
):
    
    pkg_version = version("crystalball")
    log.info(f"Crystalball version {pkg_version}")

    # get inclusion regions
    include_regions = load_regions(within) if within else []

    # Import source data from WSClean component list
    # See https://wsclean.readthedocs.io/en/latest/component_list.html
    log.info(f"Loading in {sky_model=}")
    source_model = import_from_wsclean(
        sky_model,
        include_regions=include_regions,
        point_only=points_only,
        num=num_sources or None
    )

    # Add output column if it isn't present
    ms_rows, ms_datatype = ms_preprocess(
        ms_name=ms,
        output_column=output_column
    )

    if row_chunks == 0 and model_chunks == 0:
        row_chunks, model_chunks = compute_chunk_sizes(
            ms=ms, source_model_xds=source_model, num_workers=num_workers, ms_rows=ms_rows,
            memory_fraction=memory_fraction, ms_datatype=ms_datatype, client=client
        )
    
    writes = create_predict_graph(
         ms=ms,
        source_model=source_model,
        output_column=output_column,
        field=field,
        row_chunks=row_chunks,
        model_chunks=model_chunks,
        client=client
    )

    tick = time()
    with ExitStack() as stack:
        if sys.stdout.isatty():
            # Default progress bar in user terminal
            stack.enter_context(EstimatingProgressBar())
        else:
            # Log progress every 5 minutes
            stack.enter_context(EstimatingProgressBar(minimum=2 * 60, dt=5))

        # Submit all graph computations in parallel
        if client is not None:
            log.info("Starting executions")
            future_list = client.compute(writes, optimize_graph=False)
            progress(future_list)
        else:
            dask.compute(writes)

    tock = time()
    time_taken = tock - tick
    time_unit = "sec"
    if time_taken > 60:
        time_taken /= 60
        time_unit = "min"
    if time_taken > 60:
        time_taken /= 60
        time_unit = "hr"
    log.info(f"Elapsed time: {time_taken:0.1f} {time_unit}")

    log.info("Finished")


def predict_cli() -> None:
    """The CLI entry point into crystalball
    """
    # Parse application args
    args = create_parser().parse_args([a for a in sys.argv[1:]])

    with ExitStack() as stack:
        # Set up dask client
        client = construct_client(
            stack,
            num_workers=args.num_workers,
            scheduler=args.scheduler,
            address=args.address,
            workers=1,
        )
        # Run application script
        return predict(
            ms=args.ms,
            sky_model=args.sky_model,
            output_column=args.output_column,
            field=args.field,
            row_chunks=args.row_chunks,
            model_chunks=args.model_chunks,
            within=args.within,
            points_only=args.points_only,
            num_sources=args.num_sources,
            num_workers=args.num_workers,
            memory_fraction=args.memory_fraction,
            client=client,
        )


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. "
                        "Default is 'sky-model.txt'")
    p.add_argument("-o", "--output-column", default="MODEL_DATA",
                   help="Output visibility column. Default is '%(default)s'")
    p.add_argument("-f", "--field", type=str,
                   help="The field name or id to be predicted. "
                         "If not provided, only a single field "
                         "may be present in the MS")
    p.add_argument("-rc", "--row-chunks", type=int, default=0,
                   help="Number of rows of input MS that are processed in "
                        "a single chunk. If 0 it will be set automatically. "
                        "Default is 0.")
    p.add_argument("-mc", "--model-chunks", type=int, default=0,
                   help="Number of sky model components that are processed in "
                        "a single chunk. If 0 it wil be set automatically. "
                        "Default is 0.")
    p.add_argument("-w", "--within", type=str,
                   help="Optional. Give JS9 region file. Only sources within "
                        "those regions will be included.")
    p.add_argument("-po", "--points-only", action="store_true",
                   help="Select only point-type sources.")
    p.add_argument("-ns", "--num-sources", type=int, default=0, metavar="N",
                   help="Select only N brightest sources.")
    p.add_argument("-mf", "--memory-fraction", type=float, default=0.1,
                   help="Fraction of system RAM that can be used. "
                        "Used when setting automatically the "
                        "chunk size. Default in 0.1.")
    
    dask_group = p.add_argument_group("Dask options")
    dask_group.add_argument("-j", "--num-workers", type=int, default=0, metavar="N",
                            help="Explicitly set the number of worker threads.")
    dask_group.add_argument(
        "--scheduler",
        type=str,
        choices=["threads", "distributed"],
        help="Dask scheduler type.",
        default="threads",
    )
    dask_group.add_argument("--address", type=str, help="Dask scheduler address.", default=None)

    return p



if __name__ == "__main__":
    predict_cli()