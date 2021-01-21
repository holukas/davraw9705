import os
from pathlib import Path

import logger
import setup
from readrawfile import ReadRawFile

# todo unresolved issue in file Davos970107_07.aux, too many columns in first line

indir = Path(r'L:\Dropbox\luhk_work\programming\DAVRAW9705_Convert_DAV_old_raw_data\original_raw_and_aux')
outdir = Path(r'L:\Dropbox\luhk_work\programming\DAVRAW9705_Convert_DAV_old_raw_data\raw_data_ascii')
logdir = Path(r'L:\Dropbox\luhk_work\programming\DAVRAW9705_Convert_DAV_old_raw_data\raw_data_ascii\log')

create_dirs = [outdir, logdir]
for cd in create_dirs:
    if not os.path.exists(cd):
        os.makedirs(cd)

run_id = setup.generate_run_id()
log = logger.setup_logger(run_id=run_id, logdir=logdir)

rawfiles_orig = os.listdir(indir)

for raw_file in rawfiles_orig:
    found_aux = False
    if raw_file.endswith('.raw'):
        ReadRawFile(raw_file=raw_file, indir=indir, outdir=outdir, logger=logger, rawfiles_orig=rawfiles_orig)
