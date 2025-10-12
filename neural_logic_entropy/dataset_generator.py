import run_clingo
import test_data_asp_program

from neural_logic_entropy.atom_mapper import make_atom_string

for i in range(1, 2 ** 20 + 1):
    run_clingo.run(make_atom_string(i) + test_data_asp_program.asp_program)
