import fnmatch
import os
from pathlib import Path


def search_all(dir, rawfile_id, log):
    """Search .raw and corresponding .aux files in dir that match file id"""
    log.info("Searching for files ...")
    filepairs_dict = {}
    for root, dirs, found_files in os.walk(dir):
        for idx, file in enumerate(found_files):

            if fnmatch.fnmatch(file, rawfile_id):
                aux_filepath = None

                # Found .raw
                raw_filepath = Path(root) / file

                # Search .aux
                aux_needed_file = f'{raw_filepath.stem}.aux'
                if aux_needed_file in found_files:
                    aux_filepath = Path(root) / aux_needed_file

                # If both files available, store info in dict
                if aux_filepath:
                    filepairs_dict[file] = [raw_filepath, aux_filepath]
                    log.info(f"[FOUND RAW/AUX PAIR]    {file}:")
                    log.info(f"[FOUND RAW/AUX PAIR]        {filepairs_dict[file][0]}")
                    log.info(f"[FOUND RAW/AUX PAIR]        {filepairs_dict[file][1]}")

    log.info("=" * 30)
    log.info(f"{len(filepairs_dict)} .raw files matching {rawfile_id} were found in {dir}")
    log.info("=" * 30)
    return filepairs_dict
