import os
from pathlib import Path

import files
import logger
import setup
from convertrawfile import ConvertRawFile


# todo availability heatmap, other plots aggs, hires
# DONE todo better timestamp generation for raw files
# DONE todo check if any rows are deleted during conversion, must not happen
# DONE todo correction v wind component
# todo add instrument info to header
# todo rename columns to better names: co2, h2o, ...
# todo unresolved issue in file Davos970107_07.aux, too many columns in first line


def main():
    indir = Path(r'L:\Dropbox\luhk_work\programming\DAVRAW9705_Convert_DAV_old_raw_data\_comparison')
    outdir = Path(r'L:\Dropbox\luhk_work\programming\DAVRAW9705_Convert_DAV_old_raw_data\_comparison\out')
    logdir = Path(r'L:\Dropbox\luhk_work\programming\DAVRAW9705_Convert_DAV_old_raw_data\_comparison\out\log')

    create_dirs = [outdir, logdir]
    for cd in create_dirs:
        if not os.path.exists(cd):
            os.makedirs(cd)

    # Logger
    run_id = setup.generate_run_id()
    log = logger.setup_logger(run_id=run_id, logdir=logdir)

    # Search files
    found_rawfiles_dict = files.search_all(dir=indir, rawfile_id='*.raw', log=log)

    # File conversion loop
    for filename, filepaths in found_rawfiles_dict.items():
        ConvertRawFile(raw_file=filename,
                       raw_file_fullpath=filepaths[0],
                       aux_file_fullpath=filepaths[1],
                       indir=indir, outdir=outdir,
                       log=log)


if __name__ == '__main__':
    main()
