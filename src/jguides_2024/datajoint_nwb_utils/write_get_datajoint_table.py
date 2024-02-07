# Write import statements to a file
import os

from src.jguides_2024.datajoint_nwb_utils.datajoint_table_helpers import get_import_statements
from src.jguides_2024.utils.save_load_helpers import get_file_contents


def write_get_datajoint_table(verbose=False):

    # Define file name
    file_name = "_get_datajoint_table.py"

    # Define lines in file
    opening_lines = ["def _get_table(table_name):",
                     "   # Avoid circular import", "   import os",
                     "   os.chdir('/home/jguidera/Src/jguides_2024')"]
    middle_lines = [f"   {x}" for x in get_import_statements()]
    closing_lines = ["   # Return table", "   return eval(table_name)"]
    lines = opening_lines + middle_lines + closing_lines

    # Change to directory where we want to write file
    os.chdir("/home/jguidera/Src/jguides_2024/src/jguides_2024")

    # Write file if different from existing (reduces amount of writing, which may be helpful since seems can get errors
    # when writing/reading from multiple processes)
    current_file = get_file_contents(
        f"_get_datajoint_table.py", "/home/jguidera/Src/jguides_2024/src/jguides_2024/datajoint_nwb_utils")
    if current_file != "\n".join(lines) + "\n":
        if verbose:
            print(f"Overwriting {file_name}...")
        with open(file_name, "w") as f:
            for line in lines:
                f.write(line)
                f.write(f"\n")
        f.close()
