# DAVRAW9705

## v1.0.0 / XX Mar 2021
- optimized code
- new: added table for CO2 and H2O correction factors, from comparison IRGA vs profile
- new: time series plots of aggregated merged files (raw + aux)
- new: high-resolution time series plots for each merged file (raw + aux)
- new: file data availability plot (heatmap) from available .raw files
- changed: added temp output folder that is used to store NULLBYTES files to avoid an
  error when saving the files e.g. on the server where we have no direct write access
- changed: .aux file values are now linearly interpolated instead of fill with nearest
- changed: output compressed (gzip) half-hourly files
- changed: implement better search for raw and aux files
- changed: reading irregular files should now work better
- changed: time resolution of the sonic is now assumed to be 20.833333 Hz, following the info
  found from the original programmer of the data acquisition
- changed: all rows in raw files are kept
- changed: timestamp for each row in the high-resolution raw data files is now generated
  by combining the start time as given in the filename and the total runtime of the file
- bug: reading aux file crashed if one row contained dates (str) instead of data
