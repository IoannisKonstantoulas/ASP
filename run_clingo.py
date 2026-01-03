from textwrap import dedent

import clingo


def run(string, log_level="DEBUG"):
    control = clingo.Control()

    control.add("base", [], dedent(string))

    control.ground([("base", [])])

    control.configuration.solve.models = 0

    with control.solve(yield_=True) as handle:
        # loop over all models and print them
        for model in handle:
            if log_level == "DEBUG":
                print(model)
            return(str(model))
