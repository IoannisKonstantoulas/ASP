from neural_logic_entropy.atom_enum import atoms


def make_atom_string(number):
    atom_string = ""
    for i in range(20):
        if (number >> i) & 1:
            atom_string += atoms[i]
    return atom_string