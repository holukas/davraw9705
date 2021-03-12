import pandas as pd

def get_factors(scalar, filename, delimiter, logger):
    """
    Get calibration factors from the comparison of IRGA CO2/H2O concentrations to
    independent profile measurements

    Factors are later applied to measured CO2 and H2O raw data.
    """
    logger.info(f"[CALIBRATION FACTORS] Reading calibration factors from file {filename} ...")

    df = pd.read_csv(filename, header=0, delimiter=delimiter, engine='python')

    _start_year = pd.to_datetime(df['year.from'], format='%Y')
    _start_timedelta_days = pd.to_timedelta(df['doy.from'], unit='D')
    _end_year = pd.to_datetime(df['year.to'], format='%Y')
    _end_timedelta_days = pd.to_timedelta(df['doy.to'], unit='D')

    df['start_date'] = _start_year + _start_timedelta_days
    df['end_date'] = _end_year + _end_timedelta_days - pd.Timedelta(1, unit='D')

    _start_date = df['start_date'].iloc[0]
    _end_date = df['end_date'].iloc[-1]

    date_range = pd.date_range(start=_start_date, end=_end_date)
    colname = f'{scalar}_factor'  # Factor column name
    factors_df = pd.DataFrame(index=date_range, columns=[colname])

    for row in df.iterrows():
        row_data = row[1]
        factors_df.loc[row_data['start_date'], colname] = row_data['factor.from']
        factors_df.loc[row_data['end_date'], colname] = row_data['factor.to']

    factors_df = factors_df.apply(pd.to_numeric, args=('raise',))
    factors_df = factors_df.interpolate(method='linear', axis=0, limit=None)
    return factors_df