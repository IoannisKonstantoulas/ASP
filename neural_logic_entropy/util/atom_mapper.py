from neural_logic_entropy.util.atom_enum import atoms


def make_atom_string(number):
    input_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    atom_string = ""
    for i in range(20):
        if (number >> i) & 1:
            atom_string += atoms[i]+"."
            input_set[i] = 1

    return atom_string, input_set