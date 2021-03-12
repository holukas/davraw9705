import cli
from calibration_factors import calibration_vs_profile
import os
from pathlib import Path

import pandas as pd

import _version
import files
import logger
import plots
import setup
from convertrawfile import ConvertRawFile


# todo check conversion vs WE
# todo make end messages
# DONE availability heatmap
# DONE aggs plots
# DONE hires plots (for each hour)
# DONE better timestamp generation for raw files
# DONE check if any rows are deleted during conversion, must not happen
# DONE correction v wind component
# todo add instrument info to header
# todo unresolved issue in file Davos970107_07.aux, too many columns in first line


def main(indir, outdir):
    cwd = Path(os.path.dirname(os.path.abspath(__file__)))  # Current working directory
    indir = Path(indir)
    outdir = Path(outdir)
    run_id = setup.generate_run_id()  # Run ID

    # Output folders
    outdir_run = outdir / f"OUT_{run_id}"
    outdir_run_rawdata_ascii = outdir_run / 'raw_data_ascii'
    outdir_run_log = outdir_run / 'log'
    outdir_run_plots = outdir_run / 'plots'
    outdir_run_plots_hires = outdir_run_plots / 'hires'
    outdir_run_plots_agg = outdir_run_plots / 'agg'
    create_dirs = [outdir_run, outdir_run_rawdata_ascii, outdir_run_log, outdir_run_plots,
                   outdir_run_plots_hires, outdir_run_plots_agg]
    for cd in create_dirs:
        if not os.path.exists(cd):
            os.makedirs(cd)

    # Logger
    log = logger.setup_logger(run_id=run_id, logdir=outdir_run_log)
    log.info(f"DAVRAW9705")
    log.info(f"version: {_version.__version__} / {_version.__date__}")
    log.info(f"run id: {run_id}")
    log.info(f"")

    # Search files
    found_pairs_dict = files.search_all(search_root=indir, log=log)

    # # Plot availability heatmap  # todo act
    # plots.availability_heatmap(found_rawfiles_dict=found_pairs_dict,
    #                            filename_datefrmt='Davos%y%m%d_%H.raw',
    #                            outdir=outdir_run_plots,
    #                            log=log)

    # Load calibration factors from profile comparisons
    _filepath = Path(cwd) / 'calibration_factors/Davos-V2-factors-for-EC-postcalibration.dat'
    co2_profile_factors_df = calibration_vs_profile.get_factors(scalar='co2',
                                                                filename=_filepath,
                                                                delimiter=' ', logger=log)
    _filepath = Path(cwd) / 'calibration_factors/Davos-V2-factors-for-EC-H2O-postcalibration.dat'
    h2o_profile_factors_df = calibration_vs_profile.get_factors(scalar='h2o',
                                                                filename=_filepath,
                                                                delimiter=',', logger=log)
    profile_factors_df = pd.concat([co2_profile_factors_df, h2o_profile_factors_df], axis=1)  # Merge

    # File conversion loop
    numfiles = len(found_pairs_dict)
    filecounter = 0
    stats_coll_df = pd.DataFrame()
    for filename, filepaths in found_pairs_dict.items():
        filecounter += 1
        filestats_df = ConvertRawFile(raw_file=filename,
                                      raw_file_fullpath=filepaths[0],
                                      aux_file_fullpath=filepaths[1],
                                      indir=indir,
                                      outdir_rawdata_ascii=outdir_run_rawdata_ascii,
                                      outdir_plots_hires=outdir_run_plots_hires,
                                      log=log, run_id=run_id,
                                      numfiles=numfiles,
                                      filecounter=filecounter,
                                      profile_factors_df=profile_factors_df).get_filestats()

        # Collect aggregated stats for each file
        if filecounter == 1:
            stats_coll_df = filestats_df.copy()
        else:
            stats_coll_df = stats_coll_df.append(filestats_df)

    # Save aggregated stats
    outpath = outdir_run_plots_agg / f"stats_agg_{run_id}.csv"
    log.info("")
    log.info("/" * 60)
    log.info(f"Saving stats collection to {outpath}")

    # Insert timestamp as separate column for output file
    timestamp_col = ('TIMESTAMP', '[yyyy-mm-dd HH:MM:SS]', '-')
    # stats_coll_df[timestamp_col] = stats_coll_df.index
    stats_coll_df.insert(0, timestamp_col, stats_coll_df.index)
    stats_coll_df.to_csv(f"{outpath}", index=False)
    stats_coll_df = stats_coll_df.drop(timestamp_col, 1)  # Remove timestamp column, index still available for plotting

    # Plot aggregated stats
    plots.aggs_ts(stats_coll_df=stats_coll_df, outdir_plots_agg=outdir_run_plots_agg, logger=log)

    log.info("")
    log.info("")
    log.info("")
    log.info("=" * 60)
    log.info("DAVRAW9705 finished.")
    log.info("=" * 60)


if __name__ == '__main__':
    args = cli.get_args()
    main(indir=args.source_dir, outdir=args.output_dir)
    # main(indir=r'F:\CH-DAV\[CALC]_1997-2005\X-TEST-1997',
    #      outdir_run=r'F:\CH-DAV\[CALC]_1997-2005\00 - DAWRAW9705_output')
