import numpy as np
import pandas as pd


def calc(df, bin_filedate, logger):
    """Calculate stats for raw data"""
    logger.info("    Calculating file stats ...")

    if df.empty:
        # In case there are no data, create df with one row of NaNs
        df = pd.DataFrame(index=[0], columns=df.columns)

    # Replace missing values -9999 with NaNs for correct stats calcs
    df.replace(-9999, np.nan, inplace=True)

    df['index'] = bin_filedate
    df.sort_index(axis=1, inplace=True)  # lexsort for better performance
    aggs = ['count', 'min', 'max', 'mean', 'std', 'median', q01, q05, q95, q99]
    df = df.groupby('index').agg(aggs)
    return df


def q01(x):
    return x.quantile(0.01)


def q05(x):
    return x.quantile(0.05)


def q95(x):
    return x.quantile(0.95)


def q99(x):
    return x.quantile(0.99)
