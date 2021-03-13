import datetime as dt
import os
import time

import numpy as np
import pandas as pd

import files
import plots
import stats

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 20)


class ConvertRawFile():
    filename_parsing_string = 'Davos%y%m%d_%H'

    KAPPA = 1.402  # adiabatic exponent c_p/c_v
    RGAS = 287.64  # universal gas constant for dry air, J/kg/K

    # Calibration coefficients
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

    # scaling factor to correct concentration readings based on the intercalibration
    # experiment from 28.09.2004
    # we only determined the factor for CO2 concentration data,
    # but most likely we should be using the same or a similar factor also for H2O
    SCALING_FACTOR_co2 = 1.114
    SCALING_FACTOR_h2o = 1.114

    def __init__(self, raw_file, indir, outdir_rawdata_ascii, log, raw_file_fullpath, aux_file_fullpath, run_id,
                 numfiles, filecounter, outdir_plots_hires, profile_factors_df, outdir_run_temp):
        self.indir = indir
        self.outdir = outdir_rawdata_ascii
        self.log = log
        self.raw_file = raw_file
        self.raw_file_fullpath = raw_file_fullpath
        self.aux_file_fullpath = aux_file_fullpath
        self.run_id = run_id
        self.numfiles = numfiles
        self.filecounter = filecounter
        self.outdir_plots_hires = outdir_plots_hires
        self.profile_factors_df = profile_factors_df
        self.outdir_run_temp=outdir_run_temp

        self.filestats_df = pd.DataFrame()
        self.set_colnames()  # Define column names in raw file

        self.raw_filename_start_time = \
            self.get_file_start_time(ext='raw',
                                     filename_parsing_string=self.filename_parsing_string,
                                     raw_file=self.raw_file)

        self.raw_filename_date = pd.to_datetime(self.raw_filename_start_time.date())

        self.run()

    @staticmethod
    def get_file_start_time(ext, filename_parsing_string, raw_file):
        """Get file start time from filename"""
        _parser = f"{filename_parsing_string}.{ext}"
        return dt.datetime.strptime(raw_file, _parser)

    def run(self):
        tic = time.time()  # Measure processing time for file

        self.log.info("")
        perc = (self.filecounter / self.numfiles) * 100
        self.log.info(f"{self.raw_file}  (File #{self.filecounter} of {self.numfiles})  ({perc:.1f}%)")

        self.log.info("=" * 50)
        self.log.info(f"    raw path:  {self.raw_file_fullpath}")
        self.log.info(f"    aux path:  {self.aux_file_fullpath}")
        self.log.info(f"    start time from filename:  {self.raw_filename_start_time}")

        # Get data from .raw file
        RAW_df, filled_date_range, rawfile_starttime_str = \
            self.get_raw_file_data()

        # Get data from .aux file
        AUX_df = \
            self.get_aux_file_data(filled_date_range=filled_date_range)

        # Merge .raw and .aux data
        MERGED_df = \
            self.merge_raw_aux(raw_df=RAW_df,
                               aux_df=AUX_df)

        # Corrections
        MERGED_df = self.apply_calibration_coefficients(df=MERGED_df.copy())
        # MERGED_df = self.apply_corrections(df=MERGED_df.copy())  # todo act

        # Make multi-row column index from tuples
        MERGED_df.columns = pd.MultiIndex.from_tuples(MERGED_df.columns)

        # data_start_dt = MERGED_df.index[0]
        # data_end_dt = MERGED_df.index[-1]
        # data_expected_end_dt = self.raw_filename_start_time + pd.Timedelta(hours=1)
        # if data_end_dt > data_expected_end_dt:
        # file_duration = MERGED_df.index[0] - MERGED_df.index[-1]
        # self.raw_filename_start_time

        # hh_groups = MERGED_df.groupby(pd.Grouper(freq='30T'))  # Half-hourly groups in hourly file
        # for name, group in hh_groups:
        #     print(group.index[0])
            # group.to_csv(string.replace(str(name), ':', '_') + '.csv')

        # # Plot high-res data  # todo act
        # plots.high_res_ts(df=MERGED_df, outfile=self.raw_file, outdir=self.outdir_plots_hires, logger=self.log)

        # Stats collection
        self.filestats_df = stats.calc(df=MERGED_df.copy(), bin_filedate=self.raw_filename_start_time, logger=self.log)

        # Save data
        self.save_file(df=MERGED_df, rawfile_starttime_str=rawfile_starttime_str)

        time_needed = time.time() - tic
        self.log.info(f"    (Time needed: {time_needed:.2f}s)")

    def apply_corrections(self, df):
        """Correct measurements based on comparison IRGA vs profile"""

        co2_factor = self.profile_factors_df.loc[self.raw_filename_date]['co2_factor']
        h2o_factor = self.profile_factors_df.loc[self.raw_filename_date]['h2o_factor']

        self.log.info(f"    Applying correction factors from IRGA vs profile comparison "
                      f"to CO2 and H2O measurements: CO2 * {co2_factor}  |  H2O * {h2o_factor} ...")

        co2_corrected = df[self.co2_col].multiply(co2_factor)
        h2o_corrected = df[self.h2o_col].multiply(h2o_factor)

        co2_corrected.name = self.co2_col
        h2o_corrected.name = self.h2o_col

        df.update(co2_corrected)
        df.update(h2o_corrected)

        return df

    def get_filestats(self):
        return self.filestats_df

    def set_colnames(self):
        """
        Set column names for converted files

        Column names already include the correct units after conversion.
        Note that the original raw data files contain column names that
        show wrong information, e.g. CO2 is recorded as mV signal but
        later converted to µmol/mol.
        """

        # Raw data columns
        self.u_col = ('U', '[m/s]')
        self.v_col = ('V', '[m/s]')
        self.w_col = ('W', '[m/s]')
        self.t_sonic_col = ('T_SONIC', '[degC]')  # In degC after conversions

        # Conversion of CO2 and H2O signals is tricky
        # The units given in the header are wrong.
        # Given values in the .raw files are voltages in mV.
        # Conversions are documented in the LI-6262 manual.
        # Below the final units for CO2 and H2O after conversions in this script are given.

        # CO2 concentration, in umol/mol after conversion, according to manual.
        # I assume this is dry mole fraction, i.e. mixing ratio
        self.co2_col = ('CO2', '[µmol/mol]')

        # H2O vapor concentration, in mmol/mol after conversion, according to manual
        self.h2o_col = ('H2O', '[mmol/mol]')

        self.pa_col = ('Pa', '[kPa]')  # Air pressure
        self.plic_col = ('Plic', '[kPa]')  # IRGA pressure (Licor)
        self.ta_col = ('Ta', '[degC]')  # Air temperature
        self.tlic_col = ('Tlic', '[degC]')  # IRGA temperature (Licor)
        self.xa_col = ('Xa', '[mmol / mol]')
        self.rhoa_col = ('Rhoa', '[kg / m3]')

        # Additional columns
        self.timestamp_col = ('TIMESTAMP', '[yyyy-mm-dd HH:MM:SS.]')
        self.index_col = ('Index', '[-]')
        self.date_col = ('Date', '[-]')
        self.time_col = ('Time', '[-]')
        self.datetime_col = ('DATETIME', '[yyyy-mm-dd HH:MM:SS]')
        self.total_runtime_col = ('Runtime', '[s]')

        self.year_col = ('Year', '[yyyy]')
        self.month_col = ('Month', '[mm]')
        self.day_col = ('Day', '[dd]')
        self.hour_col = ('Hour', '[HH]')
        self.minute_col = ('Minute', '[MM]')
        self.second_col = ('Second', '[SS]')
        self.runtime_second_col = ('runtime_second', '[sec]')
        self.microsecond_col = ('Microsecond', '[usec]')

    def get_raw_file_data(self):
        """Read .raw files and prepare data"""

        # Read raw file
        RAW_df = self.read_file(filepath=self.raw_file_fullpath,
                                header_rows_list=[0, 1],  # Rows AFTER skip rows considered
                                skip_rows_list=[0, 1, 2],
                                header_section_rows_list=[0, 1, 2, 3, 4])

        # Add empty units
        RAW_df = self.add_empty_units(df=RAW_df)

        # Remove unnamed columns
        RAW_df = self.remove_unnamed_cols(df=RAW_df)

        # Make timestamp
        RAW_df, filled_date_range, rawfile_starttime_str = \
            self.create_raw_timestamp(df=RAW_df)

        # Rename colnames
        RAW_df = self.sanitize_raw_colnames(df=RAW_df)

        # Convert to units
        RAW_df = self.convert_raw_to_units(df=RAW_df)

        return RAW_df, filled_date_range, rawfile_starttime_str

    def get_aux_file_data(self, filled_date_range):
        """Find respective .aux file, read it and prepare data"""

        AUX_df = self.read_file(filepath=self.aux_file_fullpath,
                                header_rows_list=[0, 1],  # Rows AFTER skip rows considered
                                skip_rows_list=[0],
                                header_section_rows_list=[0, 1, 2])

        # Add empty units, rename T and remove unnamed columns
        AUX_df = self.add_empty_units(df=AUX_df)
        AUX_df = self.remove_unnamed_cols(df=AUX_df)

        # Make timestamp
        AUX_df = \
            self.create_aux_timestamp_and_interpolate(df=AUX_df,
                                                      filled_date_range=filled_date_range,
                                                      keep_timestamp_col=True)  # make timestamp
        return AUX_df

    def sanitize_raw_colnames(self, df):
        """Rename raw data columns"""

        # aux_cols = [('Pa', '[kPa]'), ('Plic', '[kPa]'), ('Ta', '[degC]'),
        #             ('Tlic', '[degC]'), ('Xa', '[mmol/mol]'), ('Rhoa', '[kg/m3]')]

        # Original column names and renamed column names
        # For T the wrong original units are in the original header file
        # (original units are sonic temperature x 50).
        orig_colnames_dict = {
            ('U', '[m/s]'): self.u_col,
            ('V', '[m/s]'): self.v_col,
            ('W', '[m/s]'): self.w_col,
            ('Sound', '[m/50s]'): self.t_sonic_col,
            ('T', '[degC]'): self.t_sonic_col,
            ('C', '[mV]'): self.co2_col,
            ('C', '[µmol/mol]'): self.co2_col,
            ('Xw', '[mV]'): self.h2o_col,
            ('Xw', '[mmol/mol]'): self.h2o_col
        }

        # Rename columns
        new_colnames = []
        for orig_col in df.columns:
            if orig_col in orig_colnames_dict:
                new_col = orig_colnames_dict[orig_col]
                if new_col != orig_col:
                    self.log.info(f"    Renamed column {orig_col} to {new_col}")
            else:
                new_col = orig_col
                self.log.info(f"    (!)WARNING: Column {new_col} is not a defined raw data column.")
            new_colnames.append(new_col)

        df.columns = new_colnames
        return df

    def add_empty_units(self, df):
        """Rename units for columns where units were empty"""
        new_colnames = []
        for col in df.columns:
            varname = col[0]
            varunits = col[1]
            if "Unnamed:" in varunits:
                varunits = "[-]"
            new_colnames.append((varname, varunits))
        df.columns = new_colnames
        return df

    def remove_unnamed_cols(self, df):
        """
        Identify unnamed columns

        Some years have too many tabs in the file, more tabs than data columns
        empty tabs are read with col name 'Unnamed*' (or 'unkown') we use this info to delete
        empty data columns, so our data file only contains actual data values.
        Remove columns with empty (unnamed) variable name
        """
        for col in df.columns:
            varname = col[0]
            if "Unnamed:" in varname:
                df = df.drop(col, 1)
                self.log.info(f"    (!) Removed unnamed column {col}")
            elif "unknown" in varname:
                df = df.drop(col, 1)
                self.log.info(f"    (!) Removed unknown column {col}")
        return df

    def save_file(self, df: pd.DataFrame, rawfile_starttime_str: str):
        # SAVE FILE
        new_filename = os.path.join(self.outdir, 'CH-DAV_' + rawfile_starttime_str + '.csv.gz')
        # df.to_csv(new_filename, index=True)
        df.to_csv(new_filename, index=False, compression='gzip')
        self.log.info(f"    --> saved merged RAW/AUX data file {new_filename}")

    # def apply_h2o_corrections(self, df):
    #     """
    #
    #     Use the parameters attached to correct our measured H2O concentrations (H2O_EC)
    #     for up to August 2005. I used a simple linear regression with Nabel_H2O to get
    #     these parameters.
    #
    #     H2O.corrected = slope * H2O_EC + intercept
    #
    #     TIMESTAMP.START	         TIMESTAMP.END          slope       intercept   Adjusted R-squared
    #     1997-01-01 00:30:00 	 1998-01-01 00:00:00 	1.087378	24.757392	0.9723
    #     1998-01-01 00:30:00 	 1999-01-01 00:00:00 	1.171352	1.241446	0.9994
    #     1999-01-01 00:30:00 	 2000-01-01 00:00:00 	1.1655515	2.7512004	0.9893
    #     2000-01-01 00:30:00 	 2001-01-01 00:00:00 	1.1617209	3.9202374	0.9977
    #     2001-01-01 00:30:00 	 2002-01-01 00:00:00 	1.1473303	4.2208861	0.9978
    #     2002-01-01 00:30:00 	 2003-01-01 00:00:00 	1.1644168	5.9501428	0.9945
    #     2003-01-01 00:30:00 	 2004-01-01 00:00:00 	1.1580541	1.0469319	0.9953
    #     2004-01-01 00:30:00 	 2005-01-01 00:00:00 	1.304316	-12.997544	0.9485
    #     2005-01-01 00:30:00 	 2005-08-09 16:30:00 	1.2226907	-3.389643	0.9976
    #
    #     """
    #
    #     if pd.to_datetime('1997-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('1998-01-01 00:00:00'):
    #         slope = 1.087378
    #         intercept = 24.757392
    #
    #     elif pd.to_datetime('1998-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('1999-01-01 00:00:00'):
    #         slope = 1.171352
    #         intercept = 1.241446
    #
    #     elif pd.to_datetime('1999-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2000-01-01 00:00:00'):
    #         slope = 1.1655515
    #         intercept = 2.7512004
    #
    #     elif pd.to_datetime('2000-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2001-01-01 00:00:00'):
    #         slope = 1.1617209
    #         intercept = 3.9202374
    #
    #     elif pd.to_datetime('2001-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2002-01-01 00:00:00'):
    #         slope = 1.1473303
    #         intercept = 4.2208861
    #
    #     elif pd.to_datetime('2002-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2003-01-01 00:00:00'):
    #         slope = 1.1644168
    #         intercept = 5.9501428
    #
    #     elif pd.to_datetime('2003-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2004-01-01 00:00:00'):
    #         slope = 1.1580541
    #         intercept = 1.0469319
    #
    #     elif pd.to_datetime('2004-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2005-01-01 00:00:00'):
    #         slope = 1.304316
    #         intercept = -12.997544
    #
    #     elif pd.to_datetime('2005-01-01 00:30:00') <= self.raw_filename_start_time <= \
    #             pd.to_datetime('2005-08-09 16:30:00'):
    #         slope = 1.2226907
    #         intercept = -3.389643
    #
    #     else:
    #         slope = 1
    #         intercept = 0
    #
    #     h2o_measured = df[self.h2o_col].copy()
    #     h2o_corrected = h2o_measured.multiply(slope).add(intercept)
    #     self.log.info(f"    Corrected H2O values H2O_corrected = {slope} * H2O + {intercept}")
    #
    #     # Update data in df with calibrated data
    #     h2o_corrected.name = self.h2o_col  # Set name to enable update of data
    #     df.update(h2o_corrected)
    #
    #     return df

    def apply_calibration_coefficients(self, df):
        df.replace('None', np.nan, inplace=True)
        df.replace(-9999, np.nan, inplace=True)  # so we can do calculations w/o -9999

        Pa = df[self.pa_col].squeeze()  # air pressure, .squeeze() converts to Series
        Plic = df[self.plic_col].squeeze()  # licor pressure
        Ta = df[self.ta_col].squeeze()  # air temperature
        Tlic = df[self.tlic_col].squeeze()  # licor temperature
        h2o = df[self.h2o_col].squeeze()  # get h2o data
        co2 = df[self.co2_col].squeeze()  # get co2 data

        # H2O correction
        # --------------
        # this is step 1 of 3 according to the LiCOR manual:
        # water vapor concentration in ambient air.
        #   Vwmeas in mV is assumed to be in h2o
        #   Vwzero in mV is not measured and is assumed to be zero
        h2o_calibrated = h2o * self.LICOR_Sw  # no zero offset considered here!
        h2o_calibrated = h2o_calibrated * (self.LICOR_P0_h2o / Plic)
        h2o_calibrated = self.LICOR_a1_h2o * h2o_calibrated + \
                         self.LICOR_a2_h2o * h2o_calibrated ** 2 + \
                         self.LICOR_a3_h2o * h2o_calibrated ** 3  # this should be mmol mol-1, approx. 9 [WE: g m-3]
        h2o_calibrated = h2o_calibrated * ((Tlic + 273.15) / (self.LICOR_T0_h2o + 273.15))
        # $h2o_calibrated corresponds to "w" in the LiCOR sample calculation

        # since WE Version 1.7: correct for underpressure effects using a linear
        # scaling factor derived on 28.09.2004 with an intercalibration
        # experiment carried out in situ at Davos Seehornwald
        h2o_calibrated *= self.SCALING_FACTOR_h2o

        # CO2 correction
        # --------------
        # this is step 2 of 3 according to the LiCOR manual:
        # water vapor corrected CO2 in ambient air.
        #   Vcmeas in mV is assumed to be in $co2
        #   Vczero in mV is not measured and is assumed to be zero
        X = 1 + (self.LICOR_aw - 1) * h2o_calibrated / 1000
        co2_calibrated = co2 * self.LICOR_Sc  # no zero offset considered here!
        co2_calibrated = co2_calibrated * self.LICOR_P0_co2 / (Plic * X)
        co2_calibrated = self.LICOR_a1_co2 * co2_calibrated + \
                         self.LICOR_a2_co2 * co2_calibrated ** 2 + \
                         self.LICOR_a3_co2 * co2_calibrated ** 3 + \
                         self.LICOR_a4_co2 * co2_calibrated ** 4 + \
                         self.LICOR_a5_co2 * co2_calibrated ** 5  # this should MOST LIKELY be umol mol-1, approx. 370 [WE: mmol m-3]
        co2_calibrated = X * co2_calibrated * (Tlic + 273.15) / (self.LICOR_T0_co2 + 273.15)

        # since Version 1.7: correct for underpressure effects using a linear scaling
        # factor derived on 28.09.2004 with an intercalibration experiment carried out
        # in situ at Davos Seehornwald
        co2_calibrated *= self.SCALING_FACTOR_co2

        # Update data in df with calibrated data
        h2o_calibrated.name = self.h2o_col  # Set name to enable update of data
        co2_calibrated.name = self.co2_col
        df.update(h2o_calibrated)
        df.update(co2_calibrated)
        # df.loc[:, [self.h2o_col]] = h2o_calibrated
        # df.loc[:, [self.co2_col]] = co2_calibrated

        df.replace(np.nan, -9999, inplace=True)  # so we only numerics in file

        return df

    def merge_raw_aux(self, raw_df, aux_df):
        """Merge raw and aux files"""
        self.log.info(f"    Merging raw file ({len(raw_df)} rows) with aux file ({len(aux_df)} rows) ...")
        MERGED_df = pd.concat([raw_df, aux_df], axis=1)
        MERGED_df.fillna(inplace=True, method='bfill')
        MERGED_df.fillna(inplace=True, method='ffill')
        return MERGED_df

    def create_aux_timestamp_and_interpolate(self, df, filled_date_range, keep_timestamp_col):
        self.log.info(f"    Adjusting timestamp of aux file and upscaling data (linear interpolation) ...")
        df[self.date_col].fillna(method='ffill', inplace=True)  # forward-fill available times
        df[self.date_col] = df[self.date_col].str.replace(' ', '')  # remove whitespace for successful parsing

        try:
            df[self.year_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%y').dt.year
            df[self.month_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%y').dt.month
            df[self.day_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%y').dt.day
        except:
            df[self.year_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%Y').dt.year
            df[self.month_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%Y').dt.month
            df[self.day_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%Y').dt.day

        df[self.time_col].fillna(method='ffill', inplace=True)  # forward-fill available times
        df[self.time_col] = df[self.time_col].str.replace(' ', '')  # remove whitespace for successful parsing
        df[self.hour_col] = pd.to_datetime(df[self.time_col], format='%H:%M:%S').dt.hour
        df[self.minute_col] = pd.to_datetime(df[self.time_col], format='%H:%M:%S').dt.minute
        df[self.second_col] = pd.to_datetime(df[self.time_col], format='%H:%M:%S').dt.second

        # df['Index'] = df['Index'].subtract(1).multiply(50000)  # conversion from index number to microseconds
        # df['Index'].replace(1000000, 0, inplace=True)

        df['temp'] = df[[self.year_col, self.month_col, self.day_col,
                         self.hour_col, self.minute_col, self.second_col]].apply(
            lambda s: dt.datetime(*s), axis=1)
        df.drop_duplicates(subset='temp', keep='last', inplace=True)
        df.insert(0, self.timestamp_col, df['temp'])

        # Get first date and build string for filename (outfile)
        first_datetime = df[self.timestamp_col].iloc[0].to_pydatetime()  # the first datetime entry in the data file
        starttime_str = first_datetime.strftime("%Y%m%d%H%M")
        _readable_starttime_str = first_datetime.strftime("%Y-%m-%d %H:%M:%S")

        df.drop([self.date_col, self.time_col, self.year_col, self.month_col, self.day_col,
                 self.hour_col, self.minute_col, self.second_col, 'temp'], axis=1, inplace=True)
        if keep_timestamp_col:
            df.set_index(self.timestamp_col, inplace=True)
        else:
            df.drop([self.timestamp_col], axis=1, inplace=True)

        # Resample (upsample) the low-res aux data to match high-res raw data, interpolate
        df = df.sort_index()
        df = df.resample('0.01S').interpolate(method='linear', axis=0, limit=None, inplace=False)
        df = df.reindex(filled_date_range, tolerance='0.048sec', method='nearest')  # apply new continuous index to data

        # df = df.reindex(filled_date_range, fill_value=-9999, method='nearest')  # Old method
        # tidx = pd.date_range(df.index.min(), df.index.max(), freq='0.048S')
        # df = df.fillna(method='backfill', inplace=False)  # Back-fill to get values for start of file
        # df = df.fillna(method='ffill', inplace=False)  # Forward-fill to make sure there are no missing vals

        # df to numeric
        df = df.apply(pd.to_numeric, args=('coerce',))
        df = df.astype(float)

        # df.replace(-9999, np.nan, inplace=True)
        # df = df.apply(pd.Series.interpolate)
        # df.replace(np.nan, -9999, inplace=True)  # should not be necessary

        return df

    def get_multipliers(self):
        """the h2o and co2 had some amplifier installed at certain times"""

        # CO2 and H2O
        # The H2O and CO2 had some amplifier installed at certain times.
        # Use the following multiplier to correct for this:
        #
        # TIME_PERIOD			H2O_MULTIPLIER	CO2_MULTIPLIER
        # --------------------------------	--------------	--------------
        # until 22.09.1998 11:00		1.00000		1.00000
        # after this until 15.12.1998 15:00	0.33500		0.33425
        # after 15.12.1998 15:00		0.33500		1.00000
        period1_end = dt.datetime(1998, 9, 22, 11, 0)
        period2_end = dt.datetime(1998, 12, 15, 15, 0)
        if self.raw_filename_start_time > period2_end:  # after 15.12.1998 15:00
            co2_multiplier = 1
            h2o_multiplier = 0.33500
        elif period1_end < self.raw_filename_start_time <= period2_end:
            # after 22.09.1998 11:00 and until 15.12.1998 15:00
            co2_multiplier = 0.33425
            h2o_multiplier = 0.33500
        else:  # until 22.09.1998 11:00
            co2_multiplier = 1
            h2o_multiplier = 1

        # Wind speeds u, v, w
        uw_multiplier = 0.01

        # Note: in the CDEF files, generated by using the conversion script 'wsl2cdef', the
        # y-axis of the R2A is mirrored (multiplied by -1).
        # To my understanding in the CDEF files the wind direction is not defined according
        # to the instrument north arrow, but along the u wind speed axis. However, when using
        # raw data in EddyPro mirroring is not necessary. The sonic is oriented with 270° North
        # offset in relation to the North arrow. We can use this info directly in EddyPro for
        # calculations of the correct wind direction. Wind directions look correct with these
        # settings (270°, Axis).
        v_multiplier = 0.01

        # Speed of sound
        # Solent R2/R2A sonics multiply the SOS (measured in m s-1) w/ a factor of 50
        # and then report them as a positive integer value
        sos_multiplier = 0.02

        return uw_multiplier, v_multiplier, sos_multiplier, co2_multiplier, h2o_multiplier

    def convert_raw_to_units(self, df):
        """Conversion of *.raw file variables to correct units"""

        # Multpliers for variables conversion to needed units
        uw_multiplier, v_multiplier, sos_multiplier, co2_multiplier, h2o_multiplier = \
            self.get_multipliers()
        self.log.info("    multiplier in raw: co2:{} h2o:{}".format(co2_multiplier, h2o_multiplier))

        # Convert to floats so we can multiply w/ floats
        df = df.apply(pd.to_numeric)

        # Wind speeds
        # cm s-1 --> m s-1
        df.iloc[:, 0] *= uw_multiplier  # 0 = u
        df.iloc[:, 1] *= v_multiplier  # 1 = v
        df.iloc[:, 2] *= uw_multiplier  # 2 = w

        # Sonic temperature
        # m s-1 x 50 --> deg C
        df.iloc[:, 3] *= sos_multiplier  # To speed of sound
        df.iloc[:, 3] = (df.iloc[:, 3] * df.iloc[:, 3]) / self.KAPPA / self.RGAS - 273.15  # To deg C

        # CO2
        # 4 = C = CO2 concentration in mV, not in the units specified in the file! todo
        df.iloc[:, 4] *= 1 * co2_multiplier

        # H2O
        # 5 = Xw = H2O vapor concentration in mV, not in the units specified in the file! todo
        df.iloc[:, 5] *= 1 * h2o_multiplier

        return df

    def create_raw_timestamp(self, df):
        """
        Create timestamp for raw data with the start time in the filename as the basis
        """

        # Fill in *start* date and *start* time from filename
        df[self.timestamp_col] = self.raw_filename_start_time

        # Create total runtime in seconds
        frequency = 0.048000000768  # seconds; 20.833333 Hz = 1 value every 0.048000000768 seconds
        df[self.total_runtime_col] = df.index
        df[self.total_runtime_col] = df[self.total_runtime_col].multiply(frequency)

        # Add the total runtime to the start datetime of the file
        df[self.timestamp_col] = df[self.timestamp_col] + pd.to_timedelta(df[self.total_runtime_col], unit='s')

        # Get first date and build string for filename (outfile)
        first_datetime = df[self.timestamp_col].iloc[0].to_pydatetime()  # the first datetime entry in the data file
        starttime_str = first_datetime.strftime("%Y%m%d%H%M")
        _readable_starttime_str = first_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Remove unneeded columns
        df.drop([self.date_col, self.time_col, self.index_col, self.total_runtime_col],
                axis=1, inplace=True)
        df.set_index(self.timestamp_col, inplace=True)

        # Make sure data are numeric
        df = df.apply(pd.to_numeric, args=('coerce',))
        df = df.astype(float)

        # Sort by index, but this should not be necessary
        df = df.sort_index()
        filled_date_range = df.index

        self.log.info(f"    data start time:  {_readable_starttime_str}")

        return df, filled_date_range, starttime_str

    def read_file(self, filepath, skip_rows_list: list, header_rows_list: list,
                  header_section_rows_list: list):
        try:
            # df = pd.read_csv(filepath, header=header, delimiter='\t', engine='python', index_col=None)
            # df = pd.read_csv(file_fullpath, header=header, delimiter='\t', engine='python', tupleize_cols=True)

            df, _ = files.parse_csv_file(filepath=filepath,
                                         skip_rows_list=skip_rows_list,
                                         header_rows_list=header_rows_list,
                                         header_section_rows_list=header_section_rows_list)


        except Exception as e:
            self.log.info(f"    (!)WARNING Exception caught in file {filepath}")
            self.log.info(f"    (!)WARNING     Error message: \"{e}\"")
            self.log.info(f"    (!)WARNING     Now removing NULLBYTES from file ...")

            # Some files (very few) have "Error: line contains NULL byte"
            # this means there is NUL in the file when looked at it in a text editor
            # with option "Show All Characters" or similar seems there is way to
            # solve this following the code from:
            #   http://osdir.com/ml/python-pydata/2012-05/msg00044.html:
            # Update Mar 2021: I think there are only two files between 1997-2005 where
            # this exception occurs with the current way of reading files:
            #   'Davos010920_13.raw' and 'Davos000827_08.raw'

            # Read problematic file
            fi = open(filepath, 'r')  # Open the problematic file for reading
            data = fi.read()  # Read file
            fi.close()

            # Remove NULLBYTES from problematic data and write to new file
            # This step saves the new file to the temp folder of this run.

            # Assign new name
            _problematic_file_name = filepath.name
            _file_nullbytes = self.outdir_run_temp /  f"{_problematic_file_name}_NULLBYTES-REMOVED.aux"
            # _file_nullbytes = f"{filepath}_NULLBYTES-REMOVED.aux"
            fo = open(_file_nullbytes, 'w')  # Open new file
            fo.write(data.replace('\x00', ''))  # Remove NULLBYTES and write to new file
            fo.close()

            self.log.info(f"    (!)WARNING     Done.")
            self.log.info(f"    (!)WARNING     Trying to parse NULLBYTES-REMOVED ...")

            try:
                # df = pd.read_csv(_file_nullbytes, header=header_rows_list, delimiter='\t', engine='python')
                df, _ = files.parse_csv_file(filepath=_file_nullbytes,
                                             skip_rows_list=skip_rows_list,
                                             header_rows_list=header_rows_list,
                                             header_section_rows_list=header_section_rows_list)
                self.log.info(f"    (!)WARNING     Success! Reading file {_file_nullbytes} worked.")
            except Exception as e:
                self.log.info(f"    (!)WARNING     Failed!")
                self.log.info(f"    (!)WARNING     Trying to parse NULLBYTES-REMOVED file by skipping error lines ...")
                # df = pd.read_csv(_file_nullbytes, header=header_rows_list, delimiter='\t', error_bad_lines=False)
                df, _ = files.parse_csv_file(filepath=_file_nullbytes,
                                             skip_rows_list=skip_rows_list,
                                             header_rows_list=header_rows_list,
                                             header_section_rows_list=header_section_rows_list,
                                             error_bad_lines=False)
                self.log.info(f"    (!)WARNING     Success! Skipping bad lines while reading file "
                              f"{_file_nullbytes} worked.")

        self.log.info(f"    {filepath.name}  columns:  {len(df.columns)}  /  rows:  {len(df)}")
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

        # Identify unnamed columns
        # Some years have too many tabs in the file, more tabs than data columns
        # empty tabs are read with col name 'Unnamed*' we use this info to delete
        # empty data columns, so our data file only contains actual data values.

        # # Rename units for columns where units were empty
        # new_colnames = []
        # for col in df.columns:
        #     varname = col[0]
        #     varunits = col[1]
        #     if "Unnamed:" in varunits:
        #         varunits = "[-]"
        #
        #     new_colnames.append((varname, varunits))
        # df.columns = new_colnames
        #
        # # Remove columns with empty (unnamed) variable name
        # for col in df.columns:
        #     varname = col[0]
        #     if "Unnamed:" in varname:
        #         df = df.drop(col, 1)  # todo check if works
        #
        # # PUT UNITS IN HEADER
        # # units = df.iloc[0]
        # df = df.reindex(df.index.drop(0))
        # for idx, col in enumerate(df.columns):
        #     if str(units[idx]) != 'nan':
        #         if (col == 'T') and (
        #                 units[idx] == '[degC]'):  # for T the wrong units are in the original header file
        #             df = df.rename(columns={col: 'SOS_[m/s]'})
        #         else:
        #             df = df.rename(columns={col: col + '_' + str(units[idx])})

        # CREATE A TIMESTAMP FROM INFO WRITTEN IN THE FILE
        # NOTE: the continuous timestamp in these files is wrong (duplicates); but the starting time is correct
        # # CHECK AND PREPARE DATE COL
        # # some files have unnecessary whitespace or letters in the Date col that need to be deleted
        # df[self.date_col].fillna(method='ffill', inplace=True)  # forward-fill available times
        # df[self.date_col] = df[self.date_col].str.replace(' ', '')  # remove whitespace for successful parsing
        # # remove letters for successful parsing
        # df[self.date_col] = df[self.date_col].replace('[a-z]', '-9999', regex=True)

        # # CHECK AND PREPARE TIME COL
        # # (although it is not needed anymore b/c we build our own timestamp
        # df[self.time_col].fillna(method='ffill', inplace=True)
        # df[self.time_col] = df[self.time_col].str.replace(' ', '')
        # # df['Time'].ix[df['Time'].str.len() != 8]
        # # see: http://stackoverflow.com/questions/21556744/pandas-remove-rows-whose-date-does-not-follow-specified-format

        # # lines that are not in the Time format HH:MM:SS will be removed
        # df[self.time_col][df[self.time_col].str.len() != 8] = '-9999'  # todo check if nec
        # # df['Minute'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute
        # # df['Second'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.second

        # # CHECK AND PREPARE INDEX COL
        # # conversion from index number to microseconds
        # df[self.index_col] = df[self.index_col].subtract(1).multiply(50000)
        # df[self.index_col].replace(1000000, 0, inplace=True)
        # df[self.index_col] = df[self.index_col].replace(np.nan, '-9999')  # NaN can appear after removal of NULL BYTES

        # # REMOVE ERROR LINES FROM DATAFRAME todo?
        # df = df[df[self.date_col] != '-9999']
        # df = df[df[self.time_col] != '-9999']
        # df = df[df[self.index_col] != -9999]

        # # READ DATE INFORMATION
        # df[self.year_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%y').dt.year
        # df[self.month_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%y').dt.month
        # df[self.day_col] = pd.to_datetime(df[self.date_col], format='%d.%m.%y').dt.day
        # df[self.hour_col] = pd.to_datetime(df[self.time_col], format='%H:%M:%S').dt.hour
        # df[self.index_col] = df[self.index_col].astype(int)

        # # BUILD COLUMN WITH OUR OWN MINUTES, SECONDS AND MICROSECONDS
        # frequency = 0.048019  # microseconds; 20.825 Hz = 1 value every 0.048019207 seconds
        # df[self.runtime_second_col] = df.index
        # # subtract frequency so the column starts with zero seconds
        # df[self.runtime_second_col] = \
        #     (df[self.runtime_second_col] * frequency)
        # df[self.minute_col] = (df[self.runtime_second_col] / 60).astype(int)
        # df[self.second_col] = (df[self.runtime_second_col] - (df[self.minute_col] * 60)).astype(int)
        # df[self.microsecond_col] = df[self.runtime_second_col] - (df[self.minute_col] * 60) - df[self.second_col]
        # df[self.microsecond_col] = (df[self.microsecond_col] * 1000000).astype(int)

        # df = df[df[self.minute_col] < 60]  # sometimes file timestamp goes longer than 1 hour, not needed

        # # BUILD FULL DATETIME COLUMN
        # # NOTE: timestamp is not completely correct, there are duplicates written in the timestamp
        # df[self.timestamp_col] = df[[self.year_col, self.month_col, self.day_col,
        #                              self.hour_col, self.minute_col, self.second_col,
        #                              self.microsecond_col]].apply(lambda s: dt.datetime(*s), axis=1)
        # first_datetime = df[self.timestamp_col].iloc[0]  # the first datetime entry in the data file

        # BUILD STARTTIME STRING FOR FILENAME
        # raw_starttime_year = str(first_datetime.year).zfill(4)
        # raw_starttime_month = str(first_datetime.month).zfill(2)
        # raw_starttime_day = str(first_datetime.day).zfill(2)
        # raw_starttime_hour = str(first_datetime.hour).zfill(2)
        # raw_starttime_minute = str(first_datetime.minute).zfill(2)
        # starttime_str = '{}{}{}{}{}'.format(raw_starttime_year, raw_starttime_month, raw_starttime_day,
        #                                     raw_starttime_hour,
        #                                     raw_starttime_minute)
