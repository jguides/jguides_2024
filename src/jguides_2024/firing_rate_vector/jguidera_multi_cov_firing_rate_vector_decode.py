import datajoint as dj

schema = dj.schema("jguidera_path_firing_rate_vector_decode")


# These imports are called with eval or used in table definitions (do not remove):


@schema
class DecodeMultiCovFRVecParams(dj.Manual):  # use when initially generating table; if not cannot update table later
# class DecodeMultiCovFRVecParams(DecodeCovFRVecParamsBase):
    definition = """
    # Parameters for DecodeMultiCovFRVec
    decode_multi_cov_fr_vec_param_name : varchar(100)
    ---
    decode_multi_cov_fr_vec_params : blob
    """

    def get_valid_bin_nums(self, **kwargs):
        key = kwargs.pop("key")
        raise Exception(f"Must write this")

    def insert_defaults(self, **kwargs):

        # Decode path progression on correct trials where rat stayed for full delay period at destination well
        # Decode training on one path, and testing on another path
        decode_multi_cov_fr_vec_param_name = "LDA_path_progression_loocv_correct_stay_trials"
        decode_multi_cov_fr_vec_params = {
            "classifier_name": "linear_discriminant_analysis", "decode_var": "path_progression",
            "multi_cov_fr_vec_param_name": "correct_incorrect_stay_trials", "cross_validation_method": "loocv"}
        self.insert1(
            {"decode_multi_cov_fr_vec_param_name": decode_multi_cov_fr_vec_param_name,
             "decode_multi_cov_fr_vec_params": decode_multi_cov_fr_vec_params}, skip_duplicates=True)

