import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import spyglass as nd

from src.jguides_2024.datajoint_nwb_utils.datajoint_analysis_helpers import plot_horizontal_lines
from src.jguides_2024.datajoint_nwb_utils.datajoint_cross_validation_table_helpers import \
    insert_cross_validation_table
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_base import SelBase, ComputedBase
from src.jguides_2024.datajoint_nwb_utils.schema_helpers import populate_schema
from src.jguides_2024.time_and_trials.jguidera_condition_trials import ConditionTrials, \
    populate_jguidera_condition_trials
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import ResTimeBinsPoolCohort

# Needed for table definitions
nd
ConditionTrials

schema_name = "jguidera_leave_one_out_condition_trials_cross_validation"
schema = dj.schema(schema_name)


@schema
class LOOCTTrainTestSplitSel(SelBase):
    definition = """
    # Selection from upstream tables for LOOCTTrainTestSplit
    -> ConditionTrials
    """


# Test each trial for a given condition value using remaining trials
@schema
class LOOCTTrainTestSplit(ComputedBase):
    definition = """
    # Train and test indices for leave one out cross validation on condition trials
    -> LOOCTTrainTestSplitSel
    ---
    -> nd.common.AnalysisNwbfile
    train_set_df_object_id = "none" : varchar(40)
    test_set_df_object_id = "none" : varchar(40)
    train_test_set_df_object_id : varchar(40)
    time_bins_df_object_id : varchar(40)
    """

    def make(self, key):
        # Get time bin information across cohort
        time_bins_df = (ResTimeBinsPoolCohort & key).fetch_dataframes()

        # Only param is map from condition trial indices to condition values
        condition_trials_map = (ConditionTrials & key).fetch1("condition_trials_map")
        params = {"condition_trials_map": condition_trials_map}

        # Insert into table
        insert_cross_validation_table(self, key, "leave_one_out_condition_trials", params, time_bins_df)

    def fetch1_dataframe(self, object_id_name=None, restore_empty_nwb_object=True, df_index_name=None):
        df_index_name_map = {"train_set_df": "train_set_id", "test_set_df": "test_set_id"}
        df_index_name = self.get_default_df_index_name(df_index_name, object_id_name, df_index_name_map)
        return super().fetch1_dataframe(object_id_name, restore_empty_nwb_object, df_index_name)

    def plot_results(self, ax=None):

        # Get results
        dfs = self.fetch1_dataframes()

        # Initialize figure if not passed
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 10))

        # Plot results
        time_bin_centers = dfs.time_bins_df.time_bin_centers
        for _, df_row in dfs.test_set_df.iterrows():
            plot_x = time_bin_centers.iloc[df_row.test_idxs]
            ax.plot(plot_x, [df_row.name] * len(plot_x))

        # Plot horizontal lines to help visualize y positions
        xlims = [np.min(time_bin_centers), np.max(time_bin_centers)]
        plot_horizontal_lines(xlims, dfs.test_set_df.index, ax)


def populate_jguidera_leave_one_out_condition_trials_cross_validation(
        key=None, tolerate_error=False, populate_upstream_limit=None, populate_upstream_num=None):
    schema_name = "jguidera_leave_one_out_condition_trials_cross_validation"
    upstream_schema_populate_fn_list = [populate_jguidera_condition_trials]
    populate_schema(schema_name, key, tolerate_error, upstream_schema_populate_fn_list,
                    populate_upstream_limit, populate_upstream_num)


def drop_jguidera_leave_one_out_condition_trials_cross_validation():
    from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import drop_jguidera_cross_validation_pool
    drop_jguidera_cross_validation_pool()
    schema.drop()
