import datetime as dt
import fnmatch
import os
from collections import OrderedDict
from pathlib import Path

import pandas as pd


def search_files_in_subfolder(search_root, file_id, log):
    foundfiles_dict = {}
    for root, dirs, found_files in os.walk(search_root):
        # current_walk_directory = Path(root).stem
        for idx, file in enumerate(found_files):
            if fnmatch.fnmatch(file, file_id):
                filepath = Path(root) / file
                foundfiles_dict[file] = filepath
                log.info(f"[FILE SEARCH {file_id}]    Found {filepath}")
    foundfiles_dict = OrderedDict(sorted(foundfiles_dict.items()))
    return foundfiles_dict


def search_all(search_root, log):
    """Search .raw and corresponding .aux files in dir that match file id"""
    log.info("Searching for files ...")

    # Search .raw files in 'Raw' subfolder
    found_raw_dict = search_files_in_subfolder(search_root=search_root, file_id='*.raw', log=log)

    # Search .aux files in 'Aux' subfolder
    found_aux_dict = search_files_in_subfolder(search_root=search_root, file_id='*.aux', log=log)

    # Pair .raw files with their aux file
    found_pairs_dict = {}
    for rawfile, raw_filepath in found_raw_dict.items():
        aux_needed_file = f'{Path(rawfile).stem}.aux'
        if aux_needed_file in found_aux_dict:
            aux_filepath = found_aux_dict[aux_needed_file]
            found_pairs_dict[rawfile] = [raw_filepath, aux_filepath]
            log.info(f"[FOUND RAW/AUX PAIR]    RAW: {raw_filepath}")
            log.info(f"                        AUX: {aux_filepath}")

    log.info("")
    log.info("/" * 120)
    log.info("=" * 120)
    log.info("FILE SEARCH RESULTS:")
    log.info(f"Found {len(found_pairs_dict)} .raw files and respective .aux files in {search_root}")
    log.info("=" * 120)
    return found_pairs_dict


def parse_csv_file(filepath, skip_rows_list, header_rows_list, header_section_rows_list, error_bad_lines=True):
    """ Read file into df. """
    # print(filepath)

    # Check data
    # ----------
    more_data_cols_than_header_cols, num_missing_header_cols, \
    header_cols_list, generated_missing_header_cols_list = \
        compare_len_header_vs_data(filepath=filepath,
                                   skip_rows_list=skip_rows_list,
                                   header_rows_list=header_rows_list)

    # Read data file
    # --------------
    # Header section is skipped while reading, column names from header_cols_list are used instead
    data_df = pd.read_csv(filepath,
                          skiprows=header_section_rows_list,
                          header=None,
                          names=header_cols_list,
                          # na_values=settings_dict['DATA_NA_VALUES'],
                          # encoding='utf-8',
                          delimiter='\t',
                          dtype=None,
                          skip_blank_lines=True,
                          engine='python',
                          encoding='unicode_escape',
                          index_col=False,
                          error_bad_lines=error_bad_lines)

    return data_df, generated_missing_header_cols_list


def compare_len_header_vs_data(filepath, skip_rows_list, header_rows_list):
    """
    Check whether there are more data columns than given in the header.

    If not checked, this would results in an error when reading the csv file
    with .read_csv, because the method expects an equal number of header and
    data columns. If this check is True, then the difference between the length
    of the first data row and the length of the header row(s) can be used to
    automatically generate names for the missing header columns.
    """
    # Check number of columns of the first data row after the header part
    skip_num_lines = len(header_rows_list) + len(skip_rows_list)
    first_data_row_df = pd.read_csv(filepath,
                                    skiprows=skip_num_lines,
                                    header=None,
                                    nrows=1,
                                    engine='python',
                                    delimiter='\t',
                                    index_col=False)
    len_data_cols = first_data_row_df.columns.size

    # Check number of columns of the header part
    header_cols_df = pd.read_csv(filepath,
                                 skiprows=skip_rows_list,
                                 header=header_rows_list,
                                 nrows=0,
                                 engine='python',
                                 delimiter='\t')
    len_header_cols = header_cols_df.columns.size

    # Check if there are more data columns than header columns
    if len_data_cols > len_header_cols:
        more_data_cols_than_header_cols = True
        num_missing_header_cols = len_data_cols - len_header_cols
    else:
        more_data_cols_than_header_cols = False
        num_missing_header_cols = 0

    # Generate missing header columns if necessary
    header_cols_list = header_cols_df.columns.to_list()
    generated_missing_header_cols_list = []
    sfx = make_timestamp_microsec_suffix()
    if more_data_cols_than_header_cols:
        for m in list(range(1, num_missing_header_cols + 1)):
            missing_col = (f'unknown_{m}-{sfx}', '[-unknown-]')
            generated_missing_header_cols_list.append(missing_col)
            header_cols_list.append(missing_col)

    return more_data_cols_than_header_cols, num_missing_header_cols, header_cols_list, generated_missing_header_cols_list


def make_timestamp_microsec_suffix():
    now_time_dt = dt.datetime.now()
    now_time_str = now_time_dt.strftime("%H%M%S%f")
    run_id = f'{now_time_str}'
    # log(name=make_run_id.__name__, dict={'run id': run_id}, highlight=False)
    return run_id
