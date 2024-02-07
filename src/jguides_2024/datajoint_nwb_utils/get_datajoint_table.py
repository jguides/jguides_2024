from src.jguides_2024.datajoint_nwb_utils.write_get_datajoint_table import write_get_datajoint_table


def get_table(table):
   # Return datajoint table

   write_get_datajoint_table()

   # Import after writing so can import newly made script
   from src.jguides_2024.datajoint_nwb_utils._get_datajoint_table import _get_table  #

   return _get_table(table)
