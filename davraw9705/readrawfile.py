import datetime as dt
import os

import numpy as np
import pandas as pd


class ReadRawFile():
    def __init__(self, raw_file, indir, outdir, logger, rawfiles_orig):
        self.indir = indir
        self.outdir = outdir
        self.logger = logger
        self.raw_file = raw_file
        self.rawfiles_orig = rawfiles_orig

        self.raw_file_fullpath = os.path.realpath(os.path.join(indir, raw_file))
        self.aux_needed_file = os.path.splitext(raw_file)[0] + '.aux'

        self.cur_datetime = self.get_datetime_from_filename()
        self.RAW_df = self.read_file(filepath=self.raw_file_fullpath, header=3)

        # Make timestamp
        self.RAW_df, self.filled_date_range, \
        self.raw_file_starttime, self.raw_starttime_year, \
        self.raw_starttime_month, self.raw_starttime_day, \
        self.raw_starttime_hour, self.raw_starttime_minute = \
            self.create_raw_timestamp(df=self.RAW_df, keep_timestamp_col=True)

        # Start datetime of file
        self.cur_startdatetime = dt.datetime(int(self.raw_starttime_year), int(self.raw_starttime_month),
                                             int(self.raw_starttime_day), int(self.raw_starttime_hour),
                                             int(self.raw_starttime_minute))

        # Convert to units
        self.RAW_df = self.convert_data()

        # Find respective aux file
        self.AUX_df, self.found_aux = self.find_aux_file()

        # Merge raw and aux data
        self.MERGED_df = self.merge_raw_aux(raw_df=self.RAW_df.copy(),
                                            aux_df=self.AUX_df.copy())

        self.MERGED_df = self.x(df=self.MERGED_df.copy())

        self.save()

    def save(self):
        # SAVE FILE
        new_filename = os.path.join(self.outdir, self.raw_file_starttime + '.csv')
        self.MERGED_df.to_csv(new_filename, index=False)
        # contents.to_csv(new_filename, index=False)
        print("     * saved in file {}".format(new_filename))

    def x(self, df):
        # ALL DATA ARE NOW IN 1 df
        # CONVERSION TO USEFUL UNITS
        # so we can do calculations w/o None (None is generated after removing NULL BYTES):
        df.replace('None', np.nan, inplace=True)
        df.replace(-9999, np.nan, inplace=True)  # so we can do calculations w/o -9999
        df = self.calibration_coefficients(df=df,
                                           h2o_col=5,
                                           co2_col=4,
                                           Pa_col=6,
                                           Plic_col=7,
                                           Ta_col=8,
                                           Tlic_col=9)
        df.replace(np.nan, -9999, inplace=True)  # so we only numerics in file
        return df

    def calibration_coefficients(self, df, h2o_col, co2_col, Pa_col, Plic_col, Ta_col, Tlic_col):
        Pa = df.iloc[:, Pa_col]  # air pressure
        Plic = df.iloc[:, Plic_col]  # licor pressure
        Ta = df.iloc[:, Ta_col]  # air temperature
        Tlic = df.iloc[:, Tlic_col]  # licor temperature
        h2o = df.iloc[:, h2o_col]  # get h2o data
        co2 = df.iloc[:, co2_col]  # get co2 data

        # from Werner Eugster's script wsl2cdef V 1.8
        # calibration coefficients for the LiCOR 6262 in use

        # (1) for H2O
        LICOR_Sw = 1.000  # span correction for H2O - a value of 1.0 means that there is no span correction
        LICOR_a1_h2o = 8.1020e-3  # calibration coefficients for H2O
        LICOR_a2_h2o = -8.1540e-8  # calibration coefficients for H2O
        LICOR_a3_h2o = 1.2110e-9  # calibration coefficients for H2O
        LICOR_T0_h2o = 32.15  # calibration temperature
        LICOR_P0_h2o = 101.3  # standard pressure (LiCOR uses this in the computation)
        LICOR_aw = 1.5  # aw seems to be a constant with value 1.5

        # (2) for CO2
        LICOR_Sc = 1.027330  # span correction for CO2 - a value of 1.0 means that there is no span correction
        LICOR_a1_co2 = 0.14225  # calibration coefficients for CO2
        LICOR_a2_co2 = 5.9733e-6  # calibration coefficients for CO2
        LICOR_a3_co2 = 9.0892e-9  # calibration coefficients for CO2
        LICOR_a4_co2 = -1.2783e-12  # calibration coefficients for CO2
        LICOR_a5_co2 = 8.5851e-17  # calibration coefficients for CO2
        LICOR_T0_co2 = 32.15  # calibration temperature
        LICOR_P0_co2 = 101.3  # standard pressure (LiCOR uses this in the computation)

        # scaling factor to correct concentration readings based on the intercalibration experiment from 28.09.2004
        # we only determined the factor for CO2 concentration data,
        # but most likely we should be using the same or a similar factor also for H2O
        SCALING_FACTOR_co2 = 1.114
        SCALING_FACTOR_h2o = 1.114

        # this is step 1 of 3 according to the LiCOR manual:
        # water vapor concentration in ambient air.
        #   Vwmeas in mV is assumed to be in h2o
        #   Vwzero in mV is not measured and is assumed to be zero
        h2o_calibrated = h2o * LICOR_Sw  # no zero offset considered here!
        h2o_calibrated = h2o_calibrated * (LICOR_P0_h2o / Plic)
        h2o_calibrated = LICOR_a1_h2o * h2o_calibrated + \
                         LICOR_a2_h2o * h2o_calibrated ** 2 + \
                         LICOR_a3_h2o * h2o_calibrated ** 3  # this should be mmol mol-1, approx. 9 [WE: g m-3]
        h2o_calibrated = h2o_calibrated * ((Tlic + 273.15) / (LICOR_T0_h2o + 273.15))
        # $h2o_calibrated corresponds to "w" in the LiCOR sample calculation

        # since WE Version 1.7: correct for underpressure effects using a linear scaling factor derived on 28.09.2004 with an intercalibration
        # experiment carried out in situ at Davos Seehornwald
        h2o_calibrated *= SCALING_FACTOR_h2o

        # this is step 2 of 3 according to the LiCOR manual:
        # water vapor corrected CO2 in ambient air.
        #   Vcmeas in mV is assumed to be in $co2
        #   Vczero in mV is not measured and is assumed to be zero
        X = 1 + (LICOR_aw - 1) * h2o_calibrated / 1000
        co2_calibrated = co2 * LICOR_Sc  # no zero offset considered here!
        co2_calibrated = co2_calibrated * LICOR_P0_co2 / (Plic * X)
        co2_calibrated = LICOR_a1_co2 * co2_calibrated + \
                         LICOR_a2_co2 * co2_calibrated ** 2 + \
                         LICOR_a3_co2 * co2_calibrated ** 3 + \
                         LICOR_a4_co2 * co2_calibrated ** 4 + \
                         LICOR_a5_co2 * co2_calibrated ** 5  # this should MOST LIKELY be umol mol-1, approx. 370 [WE: mmol m-3]
        co2_calibrated = X * co2_calibrated * (Tlic + 273.15) / (LICOR_T0_co2 + 273.15)

        # since Version 1.7: correct for underpressure effects using a linear scaling factor derived on 28.09.2004 with an intercalibration
        # experiment carried out in situ at Davos Seehornwald
        co2_calibrated *= SCALING_FACTOR_co2

        df.iloc[:, h2o_col] = h2o_calibrated
        df.iloc[:, co2_col] = co2_calibrated

        return df

    def merge_raw_aux(self, raw_df, aux_df):
        # MERGE FILES
        # filled_date_range = pd.date_range(dataframe.index[0], dataframe.index[-1], freq='50L')  # generate continuous date range and re-index data
        # dataframe = dataframe.reindex(filled_date_range, fill_value=-9999)  # apply new continuous index to data
        if self.found_aux:
            MERGED_df = pd.concat([raw_df, aux_df], axis=1)
            MERGED_df.fillna(inplace=True, method='bfill')
            MERGED_df.fillna(inplace=True, method='ffill')
        else:
            MERGED_df = raw_df
            MERGED_df['Pa_[kPa]'] = -9999
            MERGED_df['Plic_[kPa]'] = -9999
            MERGED_df['Ta_[degC]'] = -9999
            MERGED_df['Tlic_[degC]'] = -9999
            MERGED_df['Xa_[mmol / mol]'] = -9999
            MERGED_df['Rhoa_[kg / m3]'] = -9999

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
        return MERGED_df

    def create_aux_timestamp_and_interpolate(self, df, filled_date_range, keep_timestamp_col):
        df['Date'].fillna(method='ffill', inplace=True)  # forward-fill available times
        df['Date'] = df['Date'].str.replace(' ', '')  # remove whitespace for successful parsing

        try:
            df['Year'] = pd.to_datetime(df['Date'], format='%d.%m.%y').dt.year
            df['Month'] = pd.to_datetime(df['Date'], format='%d.%m.%y').dt.month
            df['Day'] = pd.to_datetime(df['Date'], format='%d.%m.%y').dt.day
        except:
            df['Year'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.year
            df['Month'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.month
            df['Day'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.day

        df['Time'].fillna(method='ffill', inplace=True)  # forward-fill available times
        df['Time'] = df['Time'].str.replace(' ', '')  # remove whitespace for successful parsing
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
        df['Minute'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute
        df['Second'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.second

        # df['Index'] = df['Index'].subtract(1).multiply(50000)  # conversion from index number to microseconds
        # df['Index'].replace(1000000, 0, inplace=True)

        df['temp'] = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']].apply(lambda s: dt.datetime(*s), axis=1)
        df.drop_duplicates(subset='temp', keep='last', inplace=True)
        df.insert(0, 'TIMESTAMP', df['temp'])

        firstdatetime = df['TIMESTAMP'].iloc[0]

        year = str(firstdatetime.year).zfill(4)
        month = str(firstdatetime.month).zfill(2)
        day = str(firstdatetime.day).zfill(2)
        hour = str(firstdatetime.hour).zfill(2)
        minute = str(firstdatetime.minute).zfill(2)
        starttime_str = '{}{}{}{}{}'.format(year, month, day, hour, minute)

        df.drop(['Date', 'Time', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'temp'], axis=1, inplace=True)
        if keep_timestamp_col:
            df.set_index('TIMESTAMP', inplace=True)
        else:
            df.drop(['TIMESTAMP'], axis=1, inplace=True)

        # print(" Generating continuous timestamp from " + str(df.index[0]) + " until " + str(df.index[-1]))
        # generate continuous date range and re-index data
        # filled_date_range = pd.date_range(df.index[0], df.index[-1], freq=this_file_freq)
        df = df.sort_index()
        df = df.reindex(filled_date_range, fill_value=-9999, method='nearest')  # apply new continuous index to data

        # df to numeric
        df = df.apply(pd.to_numeric, args=('coerce',))
        df = df.astype(float)

        # # interpolate between values
        # df.replace(-9999, np.nan, inplace=True)
        # df = df.apply(pd.Series.interpolate)
        # df.replace(np.nan, -9999, inplace=True)  # should not be necessary

        return df, starttime_str

    def find_aux_file(self):
        """FIND CORRESPONDING *.aux FILE"""
        AUX_CONTENTS = None
        found_aux = False
        for aux_file in self.rawfiles_orig:
            if aux_file == self.aux_needed_file:
                aux_file_fullpath = os.path.realpath(os.path.join(self.indir, aux_file))
                AUX_CONTENTS = self.read_file(filepath=aux_file_fullpath, header=1)  # read file
                AUX_CONTENTS, aux_file_starttime = self.create_aux_timestamp_and_interpolate(
                    df=AUX_CONTENTS, filled_date_range=self.filled_date_range,
                    keep_timestamp_col=True)  # make timestamp
                found_aux = True
                print('\nFOUND PAIR: ' + self.raw_file + ' + ' + aux_file)

        if found_aux == False:
            # AUX_CONTENTS = pd.DataFrame()
            print('(!)NO PAIR: ' + self.raw_file + ' + no matching aux file found')

        return AUX_CONTENTS, found_aux

    def get_multipliers(self):
        """the h2o and co2 had some amplifier installed at certain times"""

        # CO2 and H2O
        period1_end = dt.datetime(1998, 9, 22, 11, 0)
        period2_end = dt.datetime(1998, 12, 15, 15, 0)
        if self.cur_startdatetime > period2_end:
            co2_multiplier = 1
            h2o_multiplier = 0.33500
        elif period1_end < self.cur_startdatetime <= period2_end:
            co2_multiplier = 0.33425
            h2o_multiplier = 0.33500
        else:
            co2_multiplier = 1
            h2o_multiplier = 1

        # Wind speeds u, v, w
        uvw_multiplier = 0.01

        # Speed of sound
        # Solent R2/R2A sonics multiply the SOS (measured in m s-1) w/ a factor of 50
        # and then report them as a positive integer value todo
        sos_multiplier = 0.02

        return uvw_multiplier, sos_multiplier, co2_multiplier, h2o_multiplier

    def convert_data(self):
        """Conversion of *.raw files"""
        df = self.RAW_df.copy()

        # Multpliers for variables conversion to needed units
        uvw_multiplier, sos_multiplier, co2_multiplier, h2o_multiplier = self.get_multipliers()
        print("     * multiplier in raw: co2:{} h2o:{}".format(co2_multiplier, h2o_multiplier))

        # Convert to floats so we can multiply w/ floats
        df.iloc[:, 0] = df.iloc[:, 0].astype(float)
        df.iloc[:, 1] = df.iloc[:, 1].astype(float)
        df.iloc[:, 2] = df.iloc[:, 2].astype(float)
        df.iloc[:, 3] = df.iloc[:, 3].astype(float)
        df.iloc[:, 4] = df.iloc[:, 4].astype(float)
        df.iloc[:, 5] = df.iloc[:, 5].astype(float)

        # Wind speeds
        # cm s-1 --> m s-1
        df.iloc[:, 0] *= uvw_multiplier  # 0 = u
        df.iloc[:, 1] *= uvw_multiplier  # 1 = v
        df.iloc[:, 2] *= uvw_multiplier  # 2 = w

        # Speed of sound
        df.iloc[:, 3] *= sos_multiplier

        # CO2
        # 4 = C = CO2 concentration in mV, not in the units specified in the file! todo
        df.iloc[:, 4] *= 1 * co2_multiplier

        # H2O
        # 5 = Xw = H2O vapor concentration in mV, not in the units specified in the file! todo
        df.iloc[:, 5] *= 1 * h2o_multiplier

        return df

    def create_raw_timestamp(self, df, keep_timestamp_col):
        # CREATE A TIMESTAMP FROM INFO WRITTEN IN THE FILE
        # NOTE: the continuous timestamp in these files is wrong (duplicates); but the starting time is correct

        # CHECK AND PREPARE DATE COL
        # some files have unnecessary whitespace or letters in the Date col that need to be deleted
        df['Date'].fillna(method='ffill', inplace=True)  # forward-fill available times
        df['Date'] = df['Date'].str.replace(' ', '')  # remove whitespace for successful parsing
        df['Date'] = df['Date'].replace('[a-z]', '-9999', regex=True)  # remove letters for successful parsing

        # CHECK AND PREPARE TIME COL (although it is not needed anymore b/c we build our own timestamp
        df['Time'].fillna(method='ffill', inplace=True)
        df['Time'] = df['Time'].str.replace(' ', '')
        # df['Time'].ix[df['Time'].str.len() != 8]  # see: http://stackoverflow.com/questions/21556744/pandas-remove-rows-whose-date-does-not-follow-specified-format
        df['Time'][
            df['Time'].str.len() != 8] = '-9999'  # lines that are not in the Time format HH:MM:SS will be removed

        # df['Minute'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute
        # df['Second'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.second

        # CHECK AND PREPARE INDEX COL
        df['Index'] = df['Index'].subtract(1).multiply(50000)  # conversion from index number to microseconds
        df['Index'].replace(1000000, 0, inplace=True)
        df['Index'] = df['Index'].replace(np.nan, '-9999')  # NaN can appear after removal of NULL BYTES

        # REMOVE ERROR LINES FROM DATAFRAME todo?
        df = df[df['Date'] != '-9999']
        df = df[df['Time'] != '-9999']
        df = df[df['Index'] != -9999]

        # READ DATE INFORMATION
        df['Year'] = pd.to_datetime(df['Date'], format='%d.%m.%y').dt.year
        df['Month'] = pd.to_datetime(df['Date'], format='%d.%m.%y').dt.month
        df['Day'] = pd.to_datetime(df['Date'], format='%d.%m.%y').dt.day
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
        df['Index'] = df['Index'].astype(int)

        # BUILD COLUMN WITH OUR OWN MINUTES, SECONDS AND MICROSECONDS
        frequency = 0.048019  # microseconds; 20.825 Hz = 1 value every 0.048019207 seconds
        df['runtime_second'] = df.index
        df['runtime_second'] = (df[
                                    'runtime_second'] * frequency) - frequency  # subtract frequency so the column starts with zero seconds
        df['Minute'] = (df['runtime_second'] / 60).astype(int)
        df['Second'] = (df['runtime_second'] - (df['Minute'] * 60)).astype(int)
        df['Microsecond'] = df['runtime_second'] - (df['Minute'] * 60) - df['Second']
        df['Microsecond'] = (df['Microsecond'] * 1000000).astype(int)

        df = df[df['Minute'] < 60]  # sometimes file timestamp goes longer than 1 hour, not needed

        # BUILD FULL DATETIME COLUMN
        # NOTE: timestamp is not completely correct, there are duplicates written in the timestamp
        df['TIMESTAMP'] = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Microsecond']].apply(
            lambda s: dt.datetime(*s), axis=1)
        first_datetime = df['TIMESTAMP'].iloc[0]  # the first datetime entry in the data file

        # BUILD STARTTIME STRING FOR FILENAME
        raw_starttime_year = str(first_datetime.year).zfill(4)
        raw_starttime_month = str(first_datetime.month).zfill(2)
        raw_starttime_day = str(first_datetime.day).zfill(2)
        raw_starttime_hour = str(first_datetime.hour).zfill(2)
        raw_starttime_minute = str(first_datetime.minute).zfill(2)
        starttime_str = '{}{}{}{}{}'.format(raw_starttime_year, raw_starttime_month, raw_starttime_day,
                                            raw_starttime_hour,
                                            raw_starttime_minute)

        # REMOVE OR KEEP COLS
        df.drop(
            ['Date', 'Time', 'Index', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Microsecond',
             'runtime_second'],
            axis=1, inplace=True)
        if keep_timestamp_col:
            df.set_index('TIMESTAMP', inplace=True)
        else:
            df.drop(['TIMESTAMP'], axis=1, inplace=True)

        # DF TO NUMERIC
        df = df.apply(pd.to_numeric, args=('coerce',))
        df = df.astype(float)

        df = df.sort_index()
        filled_date_range = df.index

        return df, filled_date_range, starttime_str, raw_starttime_year, raw_starttime_month, raw_starttime_day, raw_starttime_hour, raw_starttime_minute

    def read_file(self, filepath, header):
        try:
            df = pd.read_csv(filepath, header=header, delimiter='\t', engine='python')
            # df = pd.read_csv(file_fullpath, header=header, delimiter='\t', engine='python', tupleize_cols=True)
        except:
            try:
                # some files (very few) have "Error: line contains NULL byte"
                # this means there is NUL in the file when looked at in a text editor
                # with option "Show All Characters" or similar seems there is way to
                # solve this following the code from http://osdir.com/ml/python-pydata/2012-05/msg00044.html:
                fi = open(filepath, 'r')
                data = fi.read()
                fi.close()
                fo = open(filepath + '_NULLBYTES-REMOVED', 'w')
                fo.write(data.replace('\x00', ''))  # remove NULL BYTES
                fo.close()

                df = pd.read_csv(filepath + '_NULLBYTES-REMOVED', header=header, delimiter='\t',
                                 engine='python')
                # df = pd.read_csv(file_fullpath + '_NULLBYTES-REMOVED', header=header, delimiter='\t', engine='python', tupleize_cols=True)
            except:
                df = pd.read_csv(filepath + '_NULLBYTES-REMOVED', header=header, delimiter='\t',
                                 error_bad_lines=False)
                # df = pd.read_csv(file_fullpath + '_NULLBYTES-REMOVED', header=header, delimiter='\t', tupleize_cols=True, error_bad_lines=False)

        # DELETE EMPTY COLUMNS
        # some years have too many tabs in the file, more tabs than data columns
        # empty tabs are read with col name 'Unnamed*'
        # we use this info to delete empty data columns, so our data file only contains actual data values
        for col in df.columns:
            if 'Unnamed:' in col:
                df = df.drop(col, 1)
                # df = df.rename(columns={col: col.split('_')[0]})

        # PUT UNITS IN HEADER
        # df.to_csv("del.csv")
        units = df.iloc[0]
        df = df.reindex(df.index.drop(0))
        for idx, col in enumerate(df.columns):
            if str(units[idx]) != 'nan':
                if (col == 'T') and (
                        units[idx] == '[degC]'):  # for T the wrong units are in the original header file
                    df = df.rename(columns={col: 'SOS_[m/s]'})
                else:
                    df = df.rename(columns={col: col + '_' + str(units[idx])})

        return df

    def get_datetime_from_filename(self):
        """Get datetime from filename"""
        raw_file_year = int(self.raw_file[5:7])
        if 95 <= raw_file_year <= 99:
            raw_file_year += 1900
        else:
            raw_file_year += 2000
        raw_file_month = int(self.raw_file[7:9])
        raw_file_day = int(self.raw_file[9:11])
        cur_datetime = dt.datetime(raw_file_year, raw_file_month, raw_file_day)
        return cur_datetime
