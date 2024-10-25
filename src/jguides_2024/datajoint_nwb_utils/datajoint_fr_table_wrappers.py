from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import format_epochs, \
    get_sort_group_unit_id, plot_junction_fractions, plot_well_events
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import (abbreviate_path_name,
                                                                          format_nwb_file_name)
from src.jguides_2024.firing_rate_map.jguidera_ppt_firing_rate_map import STFrmapPuptSm, FrmapPuptSm
from src.jguides_2024.firing_rate_map.jguidera_well_arrival_departure_firing_rate_map import FrmapWADSmWT
from src.jguides_2024.firing_rate_map.jguidera_well_arrival_firing_rate_map import STFrmapWellArrivalSm, \
    FrmapUniqueWellArrivalSm
from src.jguides_2024.metadata.jguidera_brain_region import BrainRegionColor, BrainRegionCohort, \
    SortGroupTargetedLocation
from src.jguides_2024.metadata.jguidera_metadata import TaskIdentification
from src.jguides_2024.position_and_maze.jguidera_maze import EnvironmentColor, MazePathWell, RewardWellPath
from src.jguides_2024.spikes.jguidera_unit import BrainRegionUnits, BrainRegionUnitsParams
from src.jguides_2024.task_event.jguidera_dio_trials import DioWellArrivalTrialsParams
from src.jguides_2024.task_event.jguidera_task_performance import PerformanceOutcomeColors
from src.jguides_2024.utils.df_helpers import df_filter_columns, df_from_data_list, zip_df_columns
from src.jguides_2024.utils.plot_helpers import (get_gridspec_ax_maps, plot_heatmap, format_ax, save_figure,
                                                 plot_spanning_line, plot_ave_conf)
from src.jguides_2024.utils.set_helpers import check_membership
from src.jguides_2024.utils.stats_helpers import average_confidence_interval
from src.jguides_2024.utils.vector_helpers import expand_interval, unpack_single_vector


class PlotSTFRMap:
    
    def __init__(self, nwb_file_name, epochs, restrict_time_periods, rewarded_paths=True, rewarded_wells=True,
                 path_order="left_right", well_order="left_right"):
        self.nwb_file_name = nwb_file_name
        self.epochs = epochs
        self.restrict_time_periods = restrict_time_periods
        self.rewarded_paths = rewarded_paths
        self.rewarded_wells = rewarded_wells
        self.path_order = path_order
        self.well_order = well_order
        self.restrictions = self._get_restrictions()

    def _get_restrictions(self):
        # Get trial restriction information

        # Define maps from restrict_time_period to other information
        # ...fr table name
        fr_table_name_map = {"path": STFrmapPuptSm, "well_arrival": STFrmapWellArrivalSm}
        # ...type of maze restriction
        restrict_maze_name_type_map = {"path": "path_name", "well_arrival": "well_name"}
        # ...instances of maze restriction
        restrict_maze_names_map = {
            "path": MazePathWell.get_same_turn_path_names_across_epochs(
                self.nwb_file_name, self.epochs, rewarded_paths=self.rewarded_paths, collapse=True,
                collapsed_path_order=self.path_order),
            "well_arrival": MazePathWell.get_well_names_across_epochs(
                self.nwb_file_name, self.epochs, rewarded_wells=self.rewarded_wells, well_order=self.well_order)}
        # ...performance outcome name
        performance_outcome_name_map = {"path": "trial_end_performance_outcomes", "well_arrival": "performance_outcome"}
        # ...xlabel
        xlabel_map = {"path": "Path fraction", "well_arrival": "Time from well arrival (s)"}
        Restriction = namedtuple(
            'Restriction', 'restrict_time_period restrict_maze_name_type restrict_maze_name '
                           'performance_outcome_name fr_table xlabel')

        return [Restriction(
            restrict_time_period, restrict_maze_name_type_map[restrict_time_period], restrict_maze_name,
            performance_outcome_name_map[restrict_time_period], fr_table_name_map[restrict_time_period],
            xlabel_map[restrict_time_period])
            for restrict_time_period in self.restrict_time_periods
            for restrict_maze_name in restrict_maze_names_map[restrict_time_period]]

    def _get_fr_table_subset(self, key, epoch, restriction, populate_tables=False):
        # Get subset of firing rate table

        # Update key
        key.update({"epoch": epoch, restriction.restrict_maze_name_type: restriction.restrict_maze_name})

        if populate_tables:
            restriction.fr_table.populate(key)

        table_subset = (restriction.fr_table & key)

        if len(table_subset) == 0:
            print(f"Warning! No entries found for fr_arr in table {restriction.fr_table} using key {key}. "
                  f"Returning None")
            return None

        return table_subset

    def _get_fr_df(self, key, epoch, restriction, populate_tables):
        # Get firing rates from table

        table_subset = self._get_fr_table_subset(key, epoch, restriction, populate_tables)

        if table_subset is None:
            return None

        fr_df = table_subset.fetch1_dataframe()

        # Filter fr df if well arrival firing rate maps (in this case, this is the stage where
        # restrict_maze_name comes into play)
        if restriction.restrict_time_period == "well_arrival":
            fr_df = df_filter_columns(fr_df, {restriction.restrict_maze_name_type: restriction.restrict_maze_name})

        return fr_df

    def plot_single_trial_firing_rate_map(self,
                                          brain_regions=None,
                                          unit_names=None,  # use to restrict to a subset of units
                                          firing_rate_bounds_type="IQR",
                                          curation_set_name="runs_analysis_v1",
                                          min_epoch_mean_firing_rate=.1,
                                          plot_performance_outcomes=True,
                                          plot_environment_marker=True,
                                          fr_ppt_smoothed_param_name="0.05",
                                          fr_wa_smoothed_param_name="0.1",
                                          dio_well_arrival_trials_param_name=None,
                                          average_function=np.nanmedian,
                                          max_units=None,
                                          populate_tables=True,
                                          fig=None,
                                          subplot_width=3,
                                          subplot_height=2.5,
                                          height_ratios=np.asarray([1, 1]),
                                          sharey="columns",
                                          num_mega_rows_top=0,
                                          num_mega_rows_bottom=0,
                                          num_mega_columns_left=0,
                                          num_mega_columns_right=0,
                                          suppress_labels=False,
                                          plot_color_bar=False,
                                          save_fig=False):

        def _get_max_y(sharey, ax, max_ylim_map, row_iterable):

            # If each subplot has its own y axis, use this to get max height
            if sharey is None:
                return ax.get_ylim()[1]

            # Otherwise use max y across subplots
            else:
                return max_ylim_map[row_iterable] * 1.02

        # Check inputs
        valid_firing_rate_bounds_type = ["mean_CI", "IQR"]
        if firing_rate_bounds_type not in valid_firing_rate_bounds_type:
            raise Exception(f"firing_rate_bounds_type must be in {firing_rate_bounds_type} but is"
                            f" {firing_rate_bounds_type}")

        # Get inputs if not passed
        if dio_well_arrival_trials_param_name is None:
            dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_param_name([-2, 4])

        # Define brain region cohort name
        brain_region_cohort_name = "all_targeted"

        # Get brain regions if not passed
        if brain_regions is None:
            brain_regions = (BrainRegionCohort() & {
                    "brain_region_cohort_name": brain_region_cohort_name, "nwb_file_name": self.nwb_file_name}).fetch1(
                "brain_regions")

        # Define key for querying tables
        key = {"nwb_file_name": self.nwb_file_name,
               "dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name}
        if fr_ppt_smoothed_param_name is not None:
            key["fr_ppt_smoothed_param_name"] = fr_ppt_smoothed_param_name
        if fr_wa_smoothed_param_name is not None:
            key["fr_wa_smoothed_param_name"] = fr_wa_smoothed_param_name
        
        # Get brain region colors
        brain_region_color_map = BrainRegionColor().return_brain_region_color_map(brain_regions)

        # Define epoch colors based on environment
        environment_color_map = EnvironmentColor().get_environment_color_map()
        epoch_colors = {epoch: environment_color_map[TaskIdentification().get_environment(self.nwb_file_name, epoch)]
                        for epoch in self.epochs}

        # Plot parameters
        # ...gap between plots
        wspace = .3
        hspace = None
        row_iterables = ["trial_fr", "average_fr"]
        # ...location where performance outcomes are plotted on x axis
        performance_outcome_x_val = -1
        # ...clim
        scale_clim = 1
        # ...fontsizes
        fontsize = 14
        ticklabels_fontsize = 12

        # Define units to iterate over
        brain_region_units_param_name = BrainRegionUnitsParams().lookup_epochs_param_name(
                self.nwb_file_name, self.epochs, min_epoch_mean_firing_rate)
        units_df = BrainRegionUnits().get_unit_name_df(
            self.nwb_file_name, brain_region_units_param_name, curation_set_name, brain_region_cohort_name,
            brain_regions)
        # If unit names not passed, iterate over all units in brain_regions
        if unit_names is None:
            unit_names = units_df.index

        unit_counter = 0  # initialize counter for units
        for unit_name in unit_names:
            sort_group_id, unit_id = get_sort_group_unit_id(unit_name)
            key["sort_group_id"] = sort_group_id
            brain_region = units_df.loc[unit_name].brain_region

            # Continue if reached max units that want to plot
            if max_units is not None:
                unit_counter += 1
                if unit_counter > max_units:
                    continue

            # Initialize figure
            _, ax_map, fig_ = get_gridspec_ax_maps(
                mega_row_iterables=self.epochs, mega_column_iterables=np.asarray([0]), row_iterables=row_iterables,
                column_iterables=np.arange(0, len(self.restrictions)), num_mega_rows_top=num_mega_rows_top,
                num_mega_rows_bottom=num_mega_rows_bottom, num_mega_columns_left=num_mega_columns_left,
                num_mega_columns_right=num_mega_columns_right, fig=fig, sharex="columns", sharey=sharey,
                subplot_width=subplot_width, subplot_height=subplot_height,
                wspace=wspace, hspace=hspace, height_ratios=height_ratios)

            # Find min and max firing rate for this unit across all epochs and restrict_maze_names to use as clim
            # ...pool firing rates across conditions
            fr_list = []
            for restriction_idx, restriction in enumerate(self.restrictions):
                for epoch_idx, epoch in enumerate(self.epochs):
                    fr_df = self._get_fr_df(key, epoch, restriction, populate_tables)
                    if fr_df is None:
                        continue
                    fr_list += list(np.concatenate(np.vstack(df_filter_columns(
                        fr_df, {"unit_id": unit_id})["smoothed_rate_map"])))
            # ...get min and max frs
            min_fr = np.nanmin(fr_list + [0])
            max_fr = np.nanmax(fr_list + [0])

            # Plot

            # Initialize map that will use to enforce common y lim across plots
            max_ylim_map = {k: 0 for k in row_iterables}

            # # Initialize map for xlims
            # xlims_map = dict()

            # Loop through restrictions
            for restriction_idx, restriction in enumerate(self.restrictions):

                # Loop through epochs
                for epoch_idx, epoch in enumerate(self.epochs):

                    # Get firing rate. Set populate_tables to False since already would have populated tables above
                    fr_df = self._get_fr_df(key, epoch, restriction, populate_tables=False)
                    if fr_df is None:
                        for row_iterable in row_iterables:
                            ax = ax_map[(epoch, 0, row_iterable, restriction_idx)]
                            ax.axis("off")
                        continue
                    # # Store x lims (do here to avoid loading table multiple times)
                    # xlims_map[(epoch, restriction)] = fr_df._get_xlims()
                    fr_df_subset = df_filter_columns(fr_df, {"unit_id": unit_id})
                    fr_arr = np.vstack(fr_df_subset["smoothed_rate_map"])

                    # FIRST ROW OF SUBPLOTS: Plot firing rate on individual trials
                    row_iterable = "trial_fr"
                    ax = ax_map[(epoch, 0, row_iterable, restriction_idx)]
                    plot_heatmap(fr_arr, fig_ax_list=(fig_, ax), clim=[min_fr, max_fr], scale_clim=scale_clim,
                                 plot_color_bar=plot_color_bar)

                    # ...plot performance outcome if indicated
                    if plot_performance_outcomes:
                        performance_outcome_colors = [PerformanceOutcomeColors().get_performance_outcome_color(x)
                                                     for x in fr_df_subset[restriction.performance_outcome_name]]
                        ax.scatter([performance_outcome_x_val]*len(fr_arr), np.arange(0, len(fr_arr)) + .5,
                                   color=performance_outcome_colors, alpha=1, s=5)

                    # ...y label
                    ylabel = None
                    if restriction_idx == 0 and not suppress_labels:
                        ylabel = "Trial"

                    # ...Title
                    title = f"{abbreviate_path_name(restriction.restrict_maze_name)}"
                    if restriction_idx == 0:
                        title = f"Unit: {unit_name}\n{title}"

                    # suppress path name if suppressing labels (but allow unit name)
                    if suppress_labels:
                        title = ""
                        if restriction_idx == 0:
                            title = f"Unit: {unit_name}"

                    # ...Axis
                    xticks = xticklabels = []
                    yticks = [ax.get_ylim()[1]]
                    yticklabels = []
                    if restriction_idx == 0:
                        yticklabels = [int(x) for x in yticks]  # show trial number as int
                    ylim = ax.get_ylim()
                    format_ax(
                        ax=ax, xticks=xticks, xticklabels=xticklabels, yticks=yticks, yticklabels=yticklabels,
                        ylabel=ylabel, title=title, fontsize=fontsize, ticklabels_fontsize=ticklabels_fontsize)
                    ax.spines["left"].set_bounds(ylim)  # set left spine to same range as data

                    # ...update maximum y lim
                    max_ylim_map[row_iterable] = np.max([max_ylim_map[row_iterable], ax.get_ylim()[1]])

                    # SECOND ROW OF SUBPLOTS: Plot average firing rate with bounds
                    row_iterable = "average_fr"
                    ax = ax_map[(epoch, 0, row_iterable, restriction_idx)]
                    color = brain_region_color_map[brain_region]

                    # ...get bounds on mean firing rate (confidence interval around mean, or IQR of firing rates),
                    # excluding nans
                    if firing_rate_bounds_type == "mean_CI":
                        fr_bounds = [average_confidence_interval(
                        x, exclude_nan=True, average_function=average_function) for x in fr_arr.T]
                    elif firing_rate_bounds_type == "IQR":
                        fr_bounds = [np.percentile(x[np.invert(np.isnan(x))], [25, 75]) for x in fr_arr.T]

                    # ...plot average and bounds
                    mean = average_function(fr_arr, axis=0)
                    plot_params = {"color": color}
                    table_subset = self._get_fr_table_subset(key, epoch, restriction)
                    bin_centers_name = table_subset.get_bin_centers_name()
                    x_vals = unpack_single_vector(fr_df_subset[bin_centers_name].values)
                    plot_ave_conf(ax, x_vals, mean, fr_bounds, **plot_params)

                    # ...get x limits for this firing rate table
                    xlims = table_subset._get_xlims()

                    # ...update max y lims across units
                    max_ylim_map[row_iterable] = np.max([max_ylim_map[row_iterable], ax.get_ylim()[1]])

                    # ...x and y labels
                    xlabel, ylabel = None, None
                    if restriction_idx == 0 and not suppress_labels:
                        xlabel = restriction.xlabel
                        ylabel = "FR (Hz)"

                    # ...axis
                    xticks = xticklabels = xlims
                    yticklabels = []
                    if restriction_idx == 0:
                        yticklabels = None
                    format_ax(
                        ax=ax, xlim=xlims, xticks=xticks, xticklabels=xticklabels, yticklabels=yticklabels,
                        xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, ticklabels_fontsize=ticklabels_fontsize)

            # Plot vertical lines to denote task events (do after making all plots so that have common y lim max)
            for (epoch, _, row_iterable, restriction_idx), ax in ax_map.items():
                restriction = self.restrictions[restriction_idx]

                # Maze corners if path firing rate maps
                if restriction.restrict_time_period == "path" and restriction.restrict_maze_name_type == "path_name":
                    valid_path_names = RewardWellPath().fetch1("path_names")
                    if restriction.restrict_maze_name not in valid_path_names:
                        raise Exception(f"restrict_maze_name must be in {valid_path_names} to plot maze corners but is "
                                        f"{restriction.restrict_maze_name}")
                    fn_name = plot_junction_fractions
                    x_scale_factor = 1
                    if row_iterable == "trial_fr":
                        x_scale_factor = 1/float(fr_ppt_smoothed_param_name)
                    kwargs = {"x_scale_factor": x_scale_factor}

                # Well arrival and reward delivery if well arrival firing rate maps
                elif restriction.restrict_time_period == "well_arrival":
                    fn_name = plot_well_events
                    x_scale_factor = 1
                    shift_x = 0
                    if row_iterable == "trial_fr":
                        x_scale_factor = 1 / float(fr_wa_smoothed_param_name)
                        # For trial firing rate, must shift x depending on well arrival period since bins are
                        # sample number
                        fr_table_subset = self._get_fr_table_subset(key, epoch, restriction, populate_tables=False)
                        if len(fr_table_subset) == 0:
                            continue
                        time_period_start = fr_table_subset._get_xlims()[0]
                        shift_x = -time_period_start * x_scale_factor
                    kwargs = {"x_scale_factor": x_scale_factor, "shift_x": shift_x}

                else:
                    raise Exception(f"case not accounted for")

                # Get max y value of lines. For first subplot with single trial firing rates, set
                # always to max y value in each plot (to avoid bars that go up past where trials end).
                # For second subplot with average firing rate, set depending on whether using
                # common y axis or not
                max_y = ax.get_ylim()[1]  # default
                if row_iterable == "average_fr":
                    max_y = _get_max_y(sharey, ax, max_ylim_map, row_iterable)
                y_vals = expand_interval([0, max_y], expand_factor=0)  # set y value a bit larger than y lims
                fn_name(ax, y_vals, **kwargs)

            # Plot bar to indicate environment (do after making all plots so that have common y lim max)
            if plot_environment_marker:
                for (epoch, _, row_iterable, _), ax in ax_map.items():
                    max_y = _get_max_y(sharey, ax, max_ylim_map, row_iterable) * 1.02
                    ax.plot(ax.get_xlim(), [max_y]*2, color=epoch_colors[epoch], linewidth=5, alpha=1)

            # Set y lims if indicated (shared axis keyword seems to not work with GridSpec...)
            for (epoch, _, row_iterable, _), ax in ax_map.items():
                if sharey is not None:
                    ylims = [0, max_ylim_map[row_iterable]*1.1]
                    ax.set_ylim(ylims)

            if save_fig:
                file_name = f"fr_{format_nwb_file_name(self.nwb_file_name)}_{format_epochs(self.epochs)}" \
                    f"_{unit_name}"
                save_figure(fig_, file_name, save_fig=save_fig)


def order_units_by_trial_segments_fr(nwb_file_name, epoch, df_names=None, well_name=None, path_name=None,
                                     dio_well_arrival_trials_param_name=None, verbose=False):

    # Define valid inputs
    valid_df_names = ["FrmapPuptSm", "FrmapUniqueWellArrivalSm", "FrmapWADSmWT"]

    # Get defaults if not passed
    if dio_well_arrival_trials_param_name is None:
        dio_well_arrival_trials_param_name = DioWellArrivalTrialsParams().lookup_param_name([0, 2])
    if df_names is None:
        df_names = valid_df_names

    # Check inputs
    # ...Check df names valid
    check_membership(df_names, valid_df_names, "df_names", "valid_df_names")
    # ...Check that required params passed for passed df_names
    if any([x in df_names for x in ["FrmapWADSmWT", "FrmapUniqueWellArrivalSm"]]) \
        and well_name is None:
        raise Exception(f"well_name should have been passed")
    if "FrmapPuptSm" in df_names and path_name is None:
        raise Exception(f"path_name should have been passed")

    # Define key for querying tables
    key = {"nwb_file_name": nwb_file_name, "epoch": epoch,
           "dio_well_arrival_trials_param_name": dio_well_arrival_trials_param_name, "well_name": well_name,
           "path_name": path_name}

    # Get map from target region to sort group ID
    # Here, exclude sort group IDs that have no entry in EpochSpikeTimes (occurs if no units)
    targeted_location_sort_group_map = SortGroupTargetedLocation().return_targeted_location_sort_group_map(
        nwb_file_name, exclude_no_unit_sort_group_ids=True)

    # Loop through target regions
    data_list = []
    for target_region, sort_group_ids in targeted_location_sort_group_map.items():
        for sort_group_id in sort_group_ids:
            key.update({"sort_group_id": sort_group_id})

            # Get indicated dfs
            df_list = []
            for df_name in df_names:
                if df_name == "FrmapPuptSm":
                    df_list.append((FrmapPuptSm & key).fetch1_dataframe())
                elif df_name == "FrmapUniqueWellArrivalSm":
                    df_list.append((FrmapUniqueWellArrivalSm & key).fetch1_dataframe())
                elif df_name == "FrmapWADSmWT":
                    # Different well names are in same df for FrmapWADSmWT, so must
                    # filter for well name here
                    df_list.append(df_filter_columns((FrmapWADSmWT & key).fetch1_dataframe(),
                                                     {"well_name": well_name}).sort_index())
            # Unpack shared df index
            index = unpack_single_vector([x.index for x in df_list])
            rate_maps_list = [np.vstack(getattr(df, "smoothed_rate_map")) for df in df_list]  # concatenate rate maps
            concat_rate_maps = np.hstack(rate_maps_list)
            # Store concatenated frs in df
            concat_fr_df = pd.DataFrame(concat_rate_maps, index=index)
            # Store peak idx for each unit
            peak_idxs = concat_fr_df.idxmax(axis=1)
            # Store in list
            data_list += list(zip([sort_group_id] * len(peak_idxs), peak_idxs.index, peak_idxs.values))

            # Plot concatenated fr if indicated
            if verbose:
                rate_map_bounds = np.cumsum([np.shape(x)[1] for x in rate_maps_list])
                for x in concat_fr_df.values:
                    fig, ax = plt.subplots()
                    ax.plot(x, color="black")
                    span_data = ax.get_ylim()
                    for x in rate_map_bounds:
                        plot_spanning_line(ax=ax, constant_val=x, span_data=span_data, span_axis="y", color="red")

    peak_idxs_df = df_from_data_list(data_list, ["sort_group_id", "unit_id", "peak_idx"])

    return _sorted_units(peak_idxs_df)


def _sort_peak_idxs_df(peak_idxs_df):
    return peak_idxs_df.sort_values(
        by=["peak_idx", "sort_group_id", "unit_id"], ascending=[True, True, True], inplace=False)  # sort df


def _sorted_units(peak_idxs_df):

    peak_idxs_df = _sort_peak_idxs_df(peak_idxs_df)  # sort by peak idx

    return list(
        zip_df_columns(peak_idxs_df, ["sort_group_id", "unit_id"]))  # [(sg_1, unit_1), (sg_1, unit_2), ...]
