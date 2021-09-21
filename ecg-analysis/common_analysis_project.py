import numpy as np  # type: ignore


def find_list_fraction(input_list, fraction=0.5, interpolate=True):
    """
    Find index corresponding to certain fractional length within a list, e.g. halfway along, a third along

    If only looking for an interval halfway along the list, uses a simpler method that is computationally faster

    Input parameters (required):
    ----------------------------

    input_list      List to find the fractional value of

    Input parameters (optional):
    ----------------------------

    fraction        0.5     Fraction of length of list to return the value of
    interpolate     True    If fraction does not precisely specify a particular entry in the list, whether to return the
                            values on either side, or whether to interpolate between the two values (with weighting
                            given to how close the fraction is to one value or the other)
    """

    if fraction == 0.5:
        middle = float(len(input_list)) / 2
        if middle % 2 != 0:
            return input_list[int(middle - .5)]
        else:
            if interpolate:
                return np.mean((input_list[int(middle)], input_list[int(middle - 1)]), axis=0)
            else:
                return input_list[int(middle - 1)], input_list[int(middle)]

    fraction_list = np.linspace(0, 1, len(input_list))
    fraction_bounds = 0.1
    fraction_idx = np.where((fraction_list >= fraction-fraction_bounds) &
                            (fraction_list <= fraction+fraction_bounds))[0]
    while len(fraction_idx) > 2:
        fraction_bounds /= 10
        fraction_idx = np.where((fraction_list >= fraction - fraction_bounds) &
                                (fraction_list <= fraction + fraction_bounds))[0]
        if len(fraction_idx) < 1:
            fraction_bounds *= 2
            fraction_idx = np.where((fraction_list >= fraction - fraction_bounds) &
                                    (fraction_list <= fraction + fraction_bounds))[0]

    if len(fraction_idx) == 1:
        return input_list[fraction_idx[0]]
    else:
        if interpolate:
            # Interpolate between two values, based on:
            # l = l_a * f_a + l_b * f_b
            # where l_a is value of list at fraction_idx[0] and l_b is value of list at fraction_idx[1]
            # f_a = (-1/(b-a))*(x-a) + 1
            # f_b = (1/(b-a))*(x-a)
            # where a=fraction_idx[0], b=fraction_idx[1], and x is the actually desired fraction
            a = fraction_list[fraction_idx[0]]
            b = fraction_list[fraction_idx[1]]
            gradient = 1/(b-a)
            f_a = -gradient*(fraction-a)+1
            f_b = gradient*(fraction-a)

            # Return different answers, depending on whether the input list is a list of lists or not
            if isinstance(input_list[0], list):
                return [i*f_a+j*f_b for i, j in zip(input_list[fraction_idx[0]], input_list[fraction_idx[1]])]
            else:
                return input_list[fraction_idx[0]]*f_a + input_list[fraction_idx[1]]*f_b
        else:
            return tuple(np.array(input_list)[fraction_idx])
