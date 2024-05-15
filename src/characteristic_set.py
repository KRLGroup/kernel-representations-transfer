
def characteristic_set(tran_func):
    # Initialize the characteristic set with an empty set
    char_set = set()

    # Define a recursive function to explore all possible traces
    def explore_trace(state, trace, trace_sym):
        # Add the current trace to the characteristic set
        char_set.add(tuple(trace_sym))

        # Explore all possible transitions from the current state
        for symbol, next_state in tran_func[state].items():
            # If the next state is not in the trace, explore it
            if next_state not in trace:
                explore_trace(next_state, trace + [next_state], trace_sym + [symbol])

    # Start exploring traces from the initial state
    explore_trace(0, [0], [])

    # Return the characteristic set of traces
    return char_set


def characteristic_set_no_subtraces(tran_func, state=None, trace=None, trace_syms=None, char_set=None):

    if any(x is None for x in [state, trace, trace_syms, char_set]):
        assert all(x is None for x in [state, trace, trace_syms, char_set])
        state = 0
        trace = [0]
        trace_syms = []
        char_set = set()

    leaf = True
    for symbol, next_state in tran_func[state].items():
        # If this conditional is entered at least once, then there will
        # be a trace longer than the current one that also contains the
        # current trace. This means we can skip adding the current trace
        # to the characteristic set.
        if next_state not in trace:
            leaf = False
            characteristic_set_no_subtraces(tran_func, next_state, trace + [next_state], trace_syms + [symbol], char_set=char_set)

    if leaf:
        char_set.add(tuple(trace_syms))

    return char_set


if __name__ == "__main__":
    # Example usage:
    transition_function = {0: {0: 2, 1: 1}, 1: {0: 0, 1: 3}, 2: {0: 2, 1: 0}, 3:{0:3, 1:3}}

    char_set_no_subtraces = characteristic_set_no_subtraces(transition_function)
    print("Characteristic Set of Traces (no_subtraces):", char_set_no_subtraces)

    char_set = characteristic_set(transition_function)
    print("Characteristic Set of Traces:", char_set)
