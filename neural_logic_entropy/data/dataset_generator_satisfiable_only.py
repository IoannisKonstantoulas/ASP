import csv
import time

import run_clingo
import test_data_asp_program
from neural_logic_entropy.util.atom_enum import atoms
from neural_logic_entropy.util.atom_mapper import make_atom_string
from neural_logic_entropy.util.excel_header import excel_header
from neural_logic_entropy.util.loop_utils import time_print_loop

filename = "dataset_satisfiable_only.csv"
log_level = "INFO"
start_time = time.time()

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(excel_header)
    for i in range(1, 2 ** 20 + 1):
        time_print_loop(start_time, i, 1000)
        output_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        output_set_only_mask_flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        mask_all = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        mask_only_mask_flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        clingo_input_atoms, input_set = make_atom_string(i)
        clingo_output = run_clingo.run(clingo_input_atoms + test_data_asp_program.asp_program, log_level)

        if clingo_output is not None:
            output_parts = clingo_output.split(" ")
            for output_part in output_parts:
                for j in range(20):
                    if atoms[j] == output_part:
                        output_set[j] = 1
            writer.writerow(input_set + output_set + mask_all)
