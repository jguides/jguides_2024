import copy

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_schema_table_names_from_file, \
    populate_insert
from src.jguides_2024.datajoint_nwb_utils.get_datajoint_table import get_table


def populate_schema(schema_name, key=None, tolerate_error=False, upstream_schema_populate_fn_list=None,
                    populate_upstream_limit=None, populate_upstream_num=None):
    # Populate all tables within a schema

    print(f"populating {schema_name}...")
    print(f"populate_upstream_limit: {populate_upstream_limit}, populate_upstream_num: {populate_upstream_num}\n")

    if populate_upstream_limit is None:
        populate_upstream_limit = 1  # go at most one level up when populating

    if populate_upstream_num is None:
        populate_upstream_num = 0

    if populate_upstream_num < populate_upstream_limit and upstream_schema_populate_fn_list is not None:
        # Copy population level counter so that populating multiple upstream schema at a single
        # level contributes one count
        upstream_num = copy.deepcopy(populate_upstream_num)
        upstream_num += 1
        for populate_fn in upstream_schema_populate_fn_list:
            populate_fn(key, tolerate_error, populate_upstream_limit, upstream_num)

    for table_name in get_schema_table_names_from_file(schema_name):
        table = get_table(table_name)
        populate_insert(table, key=key, key_filter=key, tolerate_error=tolerate_error)

    print(f"\n")


def check_schema_populated(schema_name, key):
    # Return True if all tables have at least one entry for the passed key

    for table_name in get_schema_table_names_from_file(schema_name):
        table = get_table(table_name)
        # Check that at least one entry in table for key
        if len(table & key) == 0:
            print(f"no entry in {table.table_name} for {key}. Returning False from for schema {schema_name} ")
            return False

    return True
