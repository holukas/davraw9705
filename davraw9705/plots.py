import datetime as dt
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggs_ts(stats_coll_df, outdir_plots_agg, logger):
    """Plot aggregated values for each file"""
    logger.info("[PLOT AGGREGATES]    Plotting aggregated data from stats collection ...")
    stats_coll_df.replace(-9999, np.nan, inplace=True)
    stats_coll_df.sort_index(axis=1, inplace=True)  # lexsort for better performance
    stats_coll_df.sort_index(axis=0, inplace=True)

    # Get only var name, units and instrument from 3-row MultiIndex,
    # this means that the row with agg info is skipped here, but then used later during plotting
    vars = list(zip(stats_coll_df.columns.get_level_values(0), stats_coll_df.columns.get_level_values(1)))
    vars = set(vars)

    for var in vars:
        logger.info(f"    Plotting {var} ...")
        var_df = stats_coll_df[var].copy()

        gs = gridspec.GridSpec(2, 2)  # rows, cols
        gs.update(wspace=0.1, hspace=0, left=0.03, right=0.99, top=0.99, bottom=0.01)
        fig = plt.Figure(facecolor='white', figsize=(32, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])

        ax1.plot_date(var_df.index, var_df['median'], alpha=.5, c='#455A64', label='median')
        ax1.fill_between(x=var_df.index, y1=var_df['q95'], y2=var_df['q05'],
                         alpha=.2, color='#5f87ae', label='5-95th percentile')
        ax1.errorbar(var_df.index, var_df['mean'], var_df['std'],
                     marker='o', mec='black', mfc='None', color='black', capsize=0,
                     label='mean +/- std', alpha=.2)
        try:
            ax1.set_ylim(var_df['q01'].min(), var_df['q99'].max())
        except ValueError:
            pass
        ax2.plot_date(var_df.index, var_df['count'], alpha=1, c='#37474F', label='count')

        text_args = dict(verticalalignment='top',
                         size=14, color='black', backgroundcolor='none', zorder=100)
        ax1.text(0.01, 0.96, f"{var[0]} {var[1]}", transform=ax1.transAxes, horizontalalignment='left',
                 **text_args)

        _default_format(ax=ax1, width=1, length=2, txt_ylabel=var[0], txt_ylabel_units=var[1])
        _default_format(ax=ax2, width=1, length=2, txt_ylabel='counts')

        # ahx.axhline(0, color='black', ls='-', lw=1, zorder=1)
        font = {'family': 'sans-serif', 'size': 10}
        ax1.legend(frameon=True, loc='upper right', prop=font).set_zorder(100)

        outfile = outdir_plots_agg / f"stats_agg_{var[0]}.png"
        fig.savefig(f"{outfile}", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=150)
        logger.info(f"    --> saved aggregated time series plot: {outfile}")


def availability_heatmap(found_rawfiles_dict, filename_datefrmt, outdir, log):
    """
    Plot data availability from datetime info in filenames

    Kudos:
        https://matplotlib.org/3.2.2/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    log.info("[FILE AVAILABILITY PLOT] Plotting availability heatmap ...")

    # Prepare data for plot
    plot_df = pd.DataFrame()
    for rawfile, pair_filepaths in found_rawfiles_dict.items():
        rawfile_date = dt.datetime.strptime(rawfile, filename_datefrmt)
        plot_df.loc[rawfile_date, 'date'] = rawfile_date.date()
        plot_df.loc[rawfile_date, 'raw+aux_filesize'] = (os.path.getsize(pair_filepaths[0]) / 1000000) \
                                                        + (os.path.getsize(pair_filepaths[1]) / 1000000)  # in MB

    agg_plot_df = plot_df.groupby('date').agg('sum')
    agg_plot_df.index = pd.to_datetime(agg_plot_df.index)
    plot_range = pd.date_range(agg_plot_df.index[0], agg_plot_df.index[-1], freq='1D')
    agg_plot_df = agg_plot_df.reindex(plot_range)

    agg_plot_df['day'] = agg_plot_df.index.day
    agg_plot_df['month'] = agg_plot_df.index.month
    agg_plot_df['year'] = agg_plot_df.index.year
    agg_plot_df['year-month'] = agg_plot_df.index.strftime('%Y-%m')
    agg_plot_df = agg_plot_df.pivot("year-month", "day", "raw+aux_filesize")
    days = [str(xx) for xx in agg_plot_df.columns]
    months = [str(yy) for yy in agg_plot_df.index]

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.set_title("Filesizes of *.raw and *.aux data files per day")

    # Colormap
    cmap = plt.get_cmap('RdYlBu')
    agg_plot_df = np.ma.masked_invalid(agg_plot_df)  # Mask NaN as missing
    cmap.set_bad(color='#EEEEEE', alpha=1.)  # Set missing data to specific color

    # Data
    im = ax.imshow(agg_plot_df, cmap=cmap, aspect='equal', vmin=0)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, format='%.0f')
    cbar.ax.set_ylabel('Filesize [MB]', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(days)))
    ax.set_yticks(np.arange(len(months)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(days)
    ax.set_yticklabels(months)

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)
    ax.set_xticks(np.arange(agg_plot_df.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(agg_plot_df.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Save
    out_file = outdir / f"file_availability_heatmap"
    plt.savefig(f"{out_file}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return None


def high_res_ts(df, outfile, outdir, logger):
    logger.info("    Plotting high-res data: time series ...")

    df.replace(-9999, np.nan, inplace=True)

    # One plot for all vars
    num_plots = len(df.columns)
    cols = df.columns

    # Gridspec and axes
    gs = gridspec.GridSpec(num_plots, 1)  # rows, cols
    gs.update(wspace=0.1, hspace=0, left=0.03, right=0.99, top=0.99, bottom=0.01)
    fig = plt.Figure(facecolor='white', figsize=(16, 9))
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    gsrow = 1
    for add_ax in range(1, num_plots):
        axes.append(fig.add_subplot(gs[gsrow, 0], sharex=axes[0]))
        gsrow += 1

    text_args = dict(verticalalignment='top',
                     size=6, color='black', backgroundcolor='none', zorder=100)

    # Plot columns
    col_idx = -1
    for col in df.columns:
        # if dblock == '[HS50-A]':
        #     print(col)
        col_idx += 1
        ax = axes[col_idx]

        dataok = check_plot_data(ax=ax, df=df, col=col)
        if dataok:
            # Numeric data, values available
            ax.plot(df[col].index, df[col], 'r,', alpha=0.5, c='#5f87ae')
            txt_info = f"values: {df[col].count():.0f}\n" \
                       f"median: {df[col].median():.3f} | mean: {df[col].mean():.3f}\n" \
                       f"min: {df[col].min():.3f} | max:{df[col].max():.3f}"
            ax.text(0.99, 0.96, txt_info, transform=ax.transAxes, horizontalalignment='right', **text_args)

        _default_format(ax=ax, width=1, length=2)
        ax.text(0.01, 0.96, f"{col[0]} {col[1]}", transform=ax.transAxes, horizontalalignment='left',
                **text_args)

        # Show bottom labels for last axis in column
        if col_idx == len(df.columns) - 1:
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)

    dblock_outfile = outdir / f"{outfile}_hires.png"
    fig.savefig(f"{dblock_outfile}", format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)
    logger.info(f"    --> saved high-resolution time series plot: {dblock_outfile}")


def check_plot_data(ax, df, col):
    dataok = False
    isnumeric = False if df[col].dtypes == object else True
    ishex = False if 'hexadecimal_value' not in col[1] else True
    # isoctal = False if 'status_code_irga' not in col[1] else True
    isemtpy = False if not df[col].dropna().empty else True

    txt_warning = "-NOT-FOUND-"
    facecolor = None

    if not isemtpy and not isnumeric and ishex:
        # Expected non-numeric hexadecimal data, values available
        facecolor = 'xkcd:green'
        txt_warning = f"NOTE: Variable {col} is a HEXADECIMAL (non-numeric) value " \
                      f"and was therefore not plotted."

    # if not isemtpy and not isnumeric and not ishex and isoctal:
    #     # Expected non-numeric octal data, values available
    #     facecolor = 'xkcd:green'
    #     txt_warning = f"NOTE: Variable {col} is a OCTAL (non-numeric) value " \
    #                   f"and was therefore not plotted."

    if not isemtpy and not isnumeric and not ishex:
        # General non-numeric data, values available
        facecolor = 'xkcd:salmon'
        txt_warning = f"(!)WARNING: variable {col} is NON-NUMERIC and was therefore not plotted."

    if isemtpy:
        # No values available
        facecolor = 'xkcd:aqua'
        txt_warning = f"(!)WARNING: data for variable {col} is EMPTY and was therefore not plotted."

    if not isemtpy and not ishex and isnumeric:
        # Data are OK
        dataok = True

    if not dataok:
        ax.text(0.5, 0.5, txt_warning,
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                size=14, color='white', backgroundcolor=facecolor)
    return dataok


def _default_format(ax, fontsize=12, label_color='black',
                    txt_xlabel='', txt_ylabel='', txt_ylabel_units='',
                    width=1, length=5, direction='in', colors='black', facecolor='white'):
    """Apply default format to plot."""
    ax.set_facecolor(facecolor)
    ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize,
                   top=True)
    ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize,
                   right=True)
    _format_spines(ax=ax, color=colors, lw=0.5)
    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')


def _format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)
