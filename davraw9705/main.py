import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

import func
import logger
import setup

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

        # GET DATE FROM FILENAME
        raw_file_year = int(raw_file[5:7])
        if 95 <= raw_file_year <= 99:
            raw_file_year += 1900
        else:
            raw_file_year += 2000
        raw_file_month = int(raw_file[7:9])
        raw_file_day = int(raw_file[9:11])
        this_date = dt.datetime(raw_file_year, raw_file_month, raw_file_day)

        # READ FOUND *.raw FILE
        raw_file_fullpath = os.path.realpath(os.path.join(indir, raw_file))
        aux_needed_file = os.path.splitext(raw_file)[0] + '.aux'

        RAW_CONTENTS = func.read_file(file_fullpath=raw_file_fullpath, header=3)  # read file
        RAW_CONTENTS, filled_date_range, raw_file_starttime, raw_starttime_year, raw_starttime_month, raw_starttime_day, raw_starttime_hour, raw_starttime_minute \
            = func.create_raw_timestamp(df=RAW_CONTENTS, keep_timestamp_col=True)  # make timestamp
        this_startdatetime = dt.datetime(int(raw_starttime_year), int(raw_starttime_month), int(raw_starttime_day),
                                         int(raw_starttime_hour), int(raw_starttime_minute))

        # THE H2O AND CO2 HAD SOME AMPLIFIER INSTALLED AT CERTAIN TIMES
        period1_end = dt.datetime(1998, 9, 22, 11, 0)
        period2_end = dt.datetime(1998, 12, 15, 15, 0)
        if this_startdatetime > period2_end:
            h2o_multiplier = 0.33500
            co2_multiplier = 1
        elif period1_end < this_startdatetime <= period2_end:
            h2o_multiplier = 0.33500
            co2_multiplier = 0.33425
        else:
            h2o_multiplier = 1
            co2_multiplier = 1

        # CONVERT *.raw FILE
        # convert to floats so we can multiply w/ floats
        RAW_CONTENTS.iloc[:, 0] = RAW_CONTENTS.iloc[:, 0].astype(float)
        RAW_CONTENTS.iloc[:, 1] = RAW_CONTENTS.iloc[:, 1].astype(float)
        RAW_CONTENTS.iloc[:, 2] = RAW_CONTENTS.iloc[:, 2].astype(float)
        RAW_CONTENTS.iloc[:, 3] = RAW_CONTENTS.iloc[:, 3].astype(float)
        RAW_CONTENTS.iloc[:, 4] = RAW_CONTENTS.iloc[:, 4].astype(float)
        RAW_CONTENTS.iloc[:, 5] = RAW_CONTENTS.iloc[:, 5].astype(float)

        RAW_CONTENTS.iloc[:, 0] *= 0.01  # 0 = u: wind speeds in cm s-1 --> m s-1
        RAW_CONTENTS.iloc[:, 1] *= 0.01  # 1 = v
        RAW_CONTENTS.iloc[:, 2] *= 0.01  # 2 = w
        RAW_CONTENTS.iloc[:,
        3] *= 0.02  # 3 = T: must be speed of sound data information; Solent R2/R2A sonics multiply the SOS (measured in m s-1)
        #       w/ a factor of 50 and then report them as a positive integer value todo
        RAW_CONTENTS.iloc[:,
        4] *= 1 * co2_multiplier  # 4 = C = CO2 concentration in mV, not in the units specified in the file! todo
        RAW_CONTENTS.iloc[:,
        5] *= 1 * h2o_multiplier  # 5 = Xw = H2O vapor concentration in mV, not in the units specified in the file! todo

        # FIND CORRESPONDING *.aux FILE
        for aux_file in rawfiles_orig:
            if aux_file == aux_needed_file:
                # print("FOUND AUX FILE " + aux_file)
                aux_file_fullpath = os.path.realpath(os.path.join(indir, aux_file))
                # print(" Reading file contents ...")
                AUX_CONTENTS = func.read_file(file_fullpath=aux_file_fullpath, header=1)  # read file
                # print(" Creating TIMESTAMP ...")
                AUX_CONTENTS, aux_file_starttime = func.create_aux_timestamp_and_interpolate(
                    df=AUX_CONTENTS, filled_date_range=filled_date_range, keep_timestamp_col=True)  # make timestamp
                found_aux = True
                print('\nFOUND PAIR: ' + raw_file + ' + ' + aux_file)

        if found_aux == False:
            print('(!)NO PAIR: ' + raw_file + ' + no matching aux file found')

        print("     * multiplier in raw: co2:{} h2o:{}".format(co2_multiplier, h2o_multiplier))

        # MERGE FILES
        # filled_date_range = pd.date_range(dataframe.index[0], dataframe.index[-1], freq='50L')  # generate continuous date range and re-index data
        # dataframe = dataframe.reindex(filled_date_range, fill_value=-9999)  # apply new continuous index to data
        if found_aux:
            MERGED_CONTENTS = pd.concat([RAW_CONTENTS, AUX_CONTENTS], axis=1)
            MERGED_CONTENTS.fillna(inplace=True, method='bfill')
            MERGED_CONTENTS.fillna(inplace=True, method='ffill')
        else:
            MERGED_CONTENTS = RAW_CONTENTS
            MERGED_CONTENTS['Pa_[kPa]'] = -9999
            MERGED_CONTENTS['Plic_[kPa]'] = -9999
            MERGED_CONTENTS['Ta_[degC]'] = -9999
            MERGED_CONTENTS['Tlic_[degC]'] = -9999
            MERGED_CONTENTS['Xa_[mmol / mol]'] = -9999
            MERGED_CONTENTS['Rhoa_[kg / m3]'] = -9999

            # STRUCTURE OF MERGED_CONTENTS
            #   0: u
            #   1: v
            #   2: w
            #   3: Ts
            #   4: CO2
            #   5: H2O
            #   6: Pa
            #   7: Plic
            #   8: Ta
            #   9: Tlic
            #  10: Xa
            #  11: Rhoa

        # ALL DATA ARE NOW IN 1 df
        # CONVERSION TO USEFUL UNITS
        MERGED_CONTENTS.replace('None', np.nan,
                                inplace=True)  # so we can do calculations w/o None (None is generated after removing NULL BYTES)
        MERGED_CONTENTS.replace(-9999, np.nan, inplace=True)  # so we can do calculations w/o -9999
        MERGED_CONTENTS = func.calibration_coefficients \
                (
                df=MERGED_CONTENTS,
                h2o_col=5,
                co2_col=4,
                Pa_col=6,
                Plic_col=7,
                Ta_col=8,
                Tlic_col=9
            )
        MERGED_CONTENTS.replace(np.nan, -9999, inplace=True)  # so we only numerics in file

        # SAVE FILE
        new_filename = os.path.join(outdir, raw_file_starttime + '.csv')
        MERGED_CONTENTS.to_csv(new_filename, index=False)
        # contents.to_csv(new_filename, index=False)
        print("     * saved in file {}".format(new_filename))

# logfile_out.close()
