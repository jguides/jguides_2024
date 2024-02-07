from development.jguidera_head_speed import drop_jguidera_head_speed
from development.jguidera_lfp import drop_jguidera_lfp
from development.jguidera_position_stop import drop_jguidera_position_stop
from development.jguidera_position_stop_firing_rate_map import drop_jguidera_position_stop_firing_rate_map
from development.jguidera_position_stop_trials import drop_jguidera_position_trials
from development.jguidera_well_departure_firing_rate_map import drop_jguidera_well_departure_firing_rate_map
from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_virtual_module
from src.jguides_2024.edeno_decoder.jguidera_edeno_decoder_helpers import drop_jguidera_edeno_decoder_helpers
from src.jguides_2024.firing_rate_map.jguidera_ppt_firing_rate_map import drop_jguidera_ppt_firing_rate_map
from src.jguides_2024.firing_rate_map.jguidera_well_arrival_departure_firing_rate_map import \
    drop_jguidera_well_arrival_departure_firing_rate_map
from src.jguides_2024.firing_rate_map.jguidera_well_arrival_firing_rate_map import \
    drop_jguidera_well_arrival_firing_rate_map
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector import \
    drop_jguidera_firing_rate_difference_vector
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity import \
    drop_jguidera_firing_rate_difference_vector_similarity
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_difference_vector_similarity_ave import \
    drop_jguidera_firing_rate_difference_vector_similarity_ave
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector import drop_jguidera_firing_rate_vector
from src.jguides_2024.firing_rate_vector.jguidera_firing_rate_vector_euclidean_distance import \
    drop_jguidera_firing_rate_vector_euclidean_distance
from src.jguides_2024.glm.jguidera_basis_function import drop_jguidera_basis_function
from src.jguides_2024.glm.jguidera_el_net import drop_jguidera_el_net
from src.jguides_2024.glm.jguidera_measurements_interp_pool import drop_jguidera_measurements_interp_pool
from src.jguides_2024.metadata.jguidera_brain_region import drop_jguidera_brain_region
from src.jguides_2024.metadata.jguidera_epoch import drop_jguidera_epoch
from src.jguides_2024.position_and_maze.jguidera_maze import drop_jguidera_maze
from src.jguides_2024.position_and_maze.jguidera_position import drop_jguidera_position
from src.jguides_2024.position_and_maze.jguidera_ppt import drop_jguidera_ppt
from src.jguides_2024.position_and_maze.jguidera_ppt_interp import drop_jguidera_ppt_interp
from src.jguides_2024.spikes.jguidera_res_spikes import drop_jguidera_res_spikes
from src.jguides_2024.spikes.jguidera_spikes import drop_jguidera_spikes
from src.jguides_2024.spikes.jguidera_unit import drop_jguidera_unit
from src.jguides_2024.task_event.jguidera_dio_event import drop_jguidera_dio_event
from src.jguides_2024.task_event.jguidera_dio_trials import drop_jguidera_dio_trials
from src.jguides_2024.task_event.jguidera_statescript_event import drop_jguidera_statescript_event
from src.jguides_2024.task_event.jguidera_task_event import drop_jguidera_task_event
from src.jguides_2024.task_event.jguidera_task_performance import drop_jguidera_task_performance
from src.jguides_2024.time_and_trials.jguidera_condition_trials import drop_jguidera_condition_trials
from src.jguides_2024.time_and_trials.jguidera_cross_validation_pool import drop_jguidera_cross_validation_pool
from src.jguides_2024.time_and_trials.jguidera_interval import drop_jguidera_interval
from src.jguides_2024.time_and_trials.jguidera_kfold_cross_validation import drop_jguidera_kfold_cross_validation
from src.jguides_2024.time_and_trials.jguidera_leave_one_out_condition_trials_cross_validation import \
    drop_jguidera_leave_one_out_condition_trials_cross_validation
from src.jguides_2024.time_and_trials.jguidera_ppt_trials import drop_jguidera_ppt_trials
from src.jguides_2024.time_and_trials.jguidera_res_set import drop_jguidera_res_set
from src.jguides_2024.time_and_trials.jguidera_res_time_bins import drop_jguidera_res_time_bins
from src.jguides_2024.time_and_trials.jguidera_res_time_bins_pool import drop_jguidera_res_time_bins_pool
from src.jguides_2024.time_and_trials.jguidera_time_bins import drop_jguidera_time_bins
from src.jguides_2024.time_and_trials.jguidera_trials_pool import drop_jguidera_trials_pool
from src.jguides_2024.time_and_trials.jguidera_warped_axis_bins import drop_jguidera_warped_axis_bins
from src.jguides_2024.utils.for_loop_helpers import stoppable_function_loop, stoppable_function_outer_loop


def clear_tables(tables, drop=True, tolerate_error=True):
    for table in tables:
        # Drop table
        if drop:
            if tolerate_error:
                try:
                    table.drop()
                except:
                    print(f"Could not drop {table}")
            else:
                table.drop()
        # Otherwise, delete all entries
        else:
            if tolerate_error:
                try:
                    table.delete()
                except:
                    print(f"Could not delete entries from {table}")
            else:
                table.drop()


def drop_dgramling_lfp():
    try:
        get_virtual_module("dgramling_lfp").schema.drop()
    except:
        print(f"could not drop dgramling lfp, likely because does not exist")


def drop_schema(stop_drop_after_schema=None):
    # Note we do NOT want we to drop the following schema:
    # jguidera_reference_electrode (spikesorting depends on this)
    # jguidera_task_identification (timestamps table depends on this)
    # jguidera_timestamps (takes a long time to populate)

    # Must perform all the following drops to drop trials_pool
    stoppable_function_outer_loop(
        [drop_firing_rate_vector_schema,
         drop_unit_schema,
         drop_basis_function_schema,
         drop_trials_pool_derivative_schema,
         drop_decoding_schema,
         drop_firing_rate_map_schema,  # depends on spikes
         drop_spikes_schema,
         drop_time_bins_schema,  # depends on dio trials
         drop_position_schema,  # depends on dio trials
         drop_event_schema,
         drop_metadata_schema], stop_drop_after_schema)


def drop_unit_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_unit": drop_jguidera_unit}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_basis_function_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_basis_function": drop_jguidera_basis_function}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_trials_pool_derivative_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_glm": drop_jguidera_el_net,
                "jguidera_measurements_interp_pool": drop_jguidera_measurements_interp_pool,
                "jguidera_ppt_interp": drop_jguidera_ppt_interp,
                "jguidera_cross_validation_pool": drop_jguidera_cross_validation_pool,
                "jguidera_leave_one_out_condition_trials_cross_validation": drop_jguidera_leave_one_out_condition_trials_cross_validation,
                "jguidera_kfold_cross_validation": drop_jguidera_kfold_cross_validation,
                "jguidera_res_spikes": drop_jguidera_res_spikes,
                "jguidera_condition_trials": drop_jguidera_condition_trials,
                "jguidera_res_time_bins_pool": drop_jguidera_res_time_bins_pool,
                "jguidera_res_time_bins": drop_jguidera_res_time_bins,
                "jguidera_res_set": drop_jguidera_res_set,
                "jguidera_trials_pool": drop_jguidera_trials_pool,}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_position_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_head_speed": drop_jguidera_head_speed,
                "jguidera_position_trials": drop_jguidera_position_trials,
                "jguidera_position_stop": drop_jguidera_position_stop,
                "jguidera_ppt_trials": drop_jguidera_ppt_trials,
                "jguidera_ppt": drop_jguidera_ppt,
                "jguidera_position": drop_jguidera_position}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_event_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_dio_trials": drop_jguidera_dio_trials,
                "jguidera_task_performance": drop_jguidera_task_performance,  # depends on statescript events
                "jguidera_task_event": drop_jguidera_task_event,
                "jguidera_statescript_event": drop_jguidera_statescript_event,
                "jguidera_dio_event": drop_jguidera_dio_event}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_time_bins_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_warped_axis_bins": drop_jguidera_warped_axis_bins,
                "jguidera_time_bins": drop_jguidera_time_bins}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_res_spikes_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_res_spikes": drop_jguidera_res_spikes,
                }
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_spikes_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_lfp": drop_jguidera_lfp,  # depends on jguidera_brain_region
                "dgramling_lfp": drop_dgramling_lfp,
                "jguidera_brain_region": drop_jguidera_brain_region,  # depends on jguidera_spikes
                "jguidera_spikes": drop_jguidera_spikes,
                }
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_firing_rate_map_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_well_arrival_departure_firing_rate_map": drop_jguidera_well_arrival_departure_firing_rate_map,
                "jguidera_well_arrival_firing_rate_map": drop_jguidera_well_arrival_firing_rate_map,
                "jguidera_well_departure_firing_rate_map": drop_jguidera_well_departure_firing_rate_map,
                "jguidera_position_stop_firing_rate_map": drop_jguidera_position_stop_firing_rate_map,
                "jguidera_ppt_firing_rate_map": drop_jguidera_ppt_firing_rate_map}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_metadata_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_maze": drop_jguidera_maze,
                "jguidera_epoch": drop_jguidera_epoch,
                "jguidera_interval": drop_jguidera_interval}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_decoding_schema(stop_drop_after_schema=None):
    drop_map = {"jguidera_edeno_decoder_helpers": drop_jguidera_edeno_decoder_helpers}
    return stoppable_function_loop(drop_map, stop_drop_after_schema)


def drop_firing_rate_vector_schema(stop_drop_after_schema=None):
    drop_map = {
        "jguidera_firing_rate_difference_vector_similarity_ave":
            drop_jguidera_firing_rate_difference_vector_similarity_ave,
        "jguidera_firing_rate_difference_vector_similarity": drop_jguidera_firing_rate_difference_vector_similarity,
        "jguidera_firing_rate_vector_euclidean_distance": drop_jguidera_firing_rate_vector_euclidean_distance,
        "jguidera_firing_rate_difference_vector": drop_jguidera_firing_rate_difference_vector,
        "jguidera_firing_rate_vector": drop_jguidera_firing_rate_vector,
    }
    return stoppable_function_loop(drop_map, stop_drop_after_schema)
