from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import insert_analysis_table_entry
from src.jguides_2024.utils.cross_validation_helpers import CrossValidate


def insert_cross_validation_table(table, key, cross_validation_method, params, time_bins_df):

    # Get train/test indices
    tt = CrossValidate(cross_validation_method=cross_validation_method,
                       cross_validation_params=params).get_train_test_sets()

    # Insert into table. Note that must reset df index because nwb file does not store properly
    insert_analysis_table_entry(
        table, [tt.train_set_df.reset_index(), tt.test_set_df.reset_index(), tt.train_test_set_df,
                time_bins_df[["time_bin_centers", "time_bin_edges"]]], key, [
            "train_set_df_object_id", "test_set_df_object_id", "train_test_set_df_object_id", "time_bins_df_object_id"])
