import numpy as np
import pandas as pd

from src.jguides_2024.utils.interval_helpers import apply_merge_close_intervals
from src.jguides_2024.utils.point_process_helpers import event_times_in_intervals_bool
from src.jguides_2024.utils.tuple_helpers import index_list_of_tuples, tuples_index_list
from src.jguides_2024.utils.vector_helpers import find_spans_increasing_list, overlap


def mean_vector_around_event(time_series,
                             event_times,
                             forward_shift,
                             backward_shift,
                             t_step):
    peri_event_times_list = [np.arange(event_time + backward_shift,
                                  event_time + forward_shift + t_step, t_step)
                        for event_time in event_times]  # times around each event
    vector_peri_events = [np.interp(peri_event_times,
                                      time_series.index,
                                      time_series.values)
                          for peri_event_times in peri_event_times_list]
    return np.mean(np.vstack(vector_peri_events), axis=0)


def sliding_overlap(vector,
                    profile):
    """
    Calculate overlap (shared area) between a vector and profile in shifting windows
    :param vector: vector along which to slide profile and calculate overlap
    :param profile: profile with which to calculate overlap with vector in sliding window
    :return: overlap between vector and profile in sliding window
    """
    if len(profile) // 2 == 0:
        raise Exception(f"profile must have odd number of values")
    num_samples_before = int(np.floor(len(profile) / 2))  # number of samples to look backwards from center value
    num_samples_after = len(profile) - num_samples_before  # number of samples to look forwards from center value
    profile_overlap = np.asarray([np.nan] * len(vector))  # initialize
    for idx in np.arange(num_samples_before,
                         len(vector) - num_samples_after):
        vector_subset = vector[idx - num_samples_before:
                               idx + num_samples_after]
        if all(np.isfinite(vector_subset)):  # can only calculate overlap with finite vectors
            profile_overlap[idx] = overlap(profile, vector_subset)
    return profile_overlap


class SimilarOverlapPeriods:
    def __init__(self,
                 time_series,
                 event_times,
                 forward_shift,
                 backward_shift,
                 overlap_threshold,
                 valid_intervals=None,
                 external_time_series=None,
                 external_time_series_lower_threshold=None,
                 external_time_series_upper_threshold=None,
                 combine_close_spans_threshold=None,
                 ignore_external_vector_nan=False):
        self.time_series = time_series
        self.event_times = event_times
        self.forward_shift = forward_shift
        self.backward_shift = backward_shift
        self.overlap_threshold = overlap_threshold
        self.valid_intervals = valid_intervals
        self.external_time_series = external_time_series
        self.external_time_series_lower_threshold = external_time_series_lower_threshold
        self.external_time_series_upper_threshold = external_time_series_upper_threshold
        self.combine_close_spans_threshold = combine_close_spans_threshold
        self.ignore_external_vector_nan = ignore_external_vector_nan
        self._check_inputs()
        self.average_profile = self._get_average_profile()
        self.ts_profile_overlap = self._get_ts_profile_overlap()
        self.above_overlap_thresh_idxs = self._get_above_overlap_thresh_idxs()
        self.above_overlap_thresh_spans = self._get_above_overlap_thresh_spans()
        self.valid_above_overlap_thresh_spans = self._get_valid_above_overlap_thresh_spans()
        self.max_overlap_times = self._get_max_overlap_times()


    def _check_inputs(self):
        if self.external_time_series is not None:
            if self.external_time_series_lower_threshold is None:
                raise Exception(f"external_time_series_lower_threshold must not be None if external_time_series passed")
            if self.external_time_series_upper_threshold is None:
                raise Exception(f"external_time_series_upper_threshold must not be None if external_time_series passed")

    def _get_average_profile(self):
        # Find mean vector profile in a window around events
        return mean_vector_around_event(time_series=self.time_series,
                                         event_times=self.event_times,
                                         forward_shift=self.forward_shift,
                                         backward_shift=self.backward_shift,
                                         t_step=np.median(np.diff(self.time_series.index)))

    def _get_ts_profile_overlap(self):
        # Calculate overlap between mean vector profile and time series
        return pd.Series(sliding_overlap(vector=self.time_series.values,
                                         profile=self.average_profile),
                                         index=self.time_series.index)

    def _get_above_overlap_thresh_idxs(self):
        # Find spans of overlap exceeding threshold
        return np.where(self.ts_profile_overlap >
                        self.overlap_threshold)[0]  # where overlap exceeds threshold

    def _get_above_overlap_thresh_spans(self):
        return find_spans_increasing_list(self.above_overlap_thresh_idxs)[0]  # spans where overlap exceeds threshold

    def _get_valid_above_overlap_thresh_spans(self):
        # Exclude spans with start/end outside valid intervals
        if self.valid_intervals is not None:  # if valid intervals passed
            spans_start, spans_end = list(zip(*self.above_overlap_thresh_spans))
            spans_start_time = self.ts_profile_overlap.index[list(spans_start)]
            spans_end_time = self.ts_profile_overlap.index[list(spans_end)]
            valid_spans_bool = np.logical_and(event_times_in_intervals_bool(spans_start_time, self.valid_intervals),
                                              event_times_in_intervals_bool(spans_end_time, self.valid_intervals))
            valid_above_overlap_thresh_spans = index_list_of_tuples(self.above_overlap_thresh_spans, valid_spans_bool)
        # Exclude spans with values outside valid range if indicated
        if self.external_time_series is not None:
            valid_above_overlap_thresh_spans_time = tuples_index_list(valid_above_overlap_thresh_spans,
                                                                      list(self.ts_profile_overlap.index))  # times of spans where overlap threshold passed
            valid_above_overlap_thresh_spans_ext = [self.external_time_series.iloc[event_times_in_intervals_bool(self.external_time_series.index,
                                                                                           [span])].values
                                                    for span in valid_above_overlap_thresh_spans_time]  # external time series during spans where overlap threshold passed
            valid_spans_bool = []
            for span_ext in valid_above_overlap_thresh_spans_ext:
                if self.ignore_external_vector_nan:  # if only want to consider finite values from external vector (e.g. only finite values of ppt)
                    span_ext = span_ext[np.isfinite(span_ext)]
                valid_spans_bool.append(all(np.logical_and(span_ext >= self.external_time_series_lower_threshold,
                                                           span_ext <= self.external_time_series_upper_threshold)))
            valid_above_overlap_thresh_spans = index_list_of_tuples(valid_above_overlap_thresh_spans, valid_spans_bool)
        return valid_above_overlap_thresh_spans

    def _combine_close_spans(self):
        # Combine spans that are close in time if indicated
        if self.combine_close_spans_threshold is not None:
            valid_above_overlap_thresh_spans_time = tuples_index_list(self.valid_above_overlap_thresh_spans,
                                                                      self.ts_profile_overlap.index)
            if abs(len(valid_above_overlap_thresh_spans_time) * len(self.valid_above_overlap_thresh_spans)) > 0:
                self.valid_above_overlap_thresh_spans = apply_merge_close_intervals(
                    intervals_1=valid_above_overlap_thresh_spans_time,
                    merge_threshold=self.combine_close_spans_threshold,
                    intervals_2=self.valid_above_overlap_thresh_spans)

    def _get_max_overlap_times(self):
        # Define max overlap times as times within each span when overlap measure is greatest
        max_overlap_times = []
        for span_start, span_end in self.valid_above_overlap_thresh_spans:
            span_overlap = self.ts_profile_overlap.iloc[span_start:span_end + 1]
            max_overlap_times.append(span_overlap.index[np.argmax(span_overlap)])
        return max_overlap_times