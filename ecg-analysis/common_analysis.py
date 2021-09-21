def check_list_depth(input_list, depth_count=1, max_depth=0, n_args=0):
    """ Function to calculate the depth of nested loops

    TODO: Finish this damn code

    Parameters
    ----------
    input_list : list
        Input argument to check
    depth_count : int, optional
        Depth of nested loops thus far
    max_depth : int, optional
        Maximum expected depth of list, default=0 (not checked)
    n_args : int, optional
        Required length of 'base' list, default=0 (not checked)

    Returns
    -------
    depth_count : int
        Depth of nested loops

    Notes
    -----
    A list of form [a1, a2, a3, ...] has depth 1.
    A list of form [[a1, a2, a3, ...], [b1, b2, b3, ...], ...] has depth 2.
    And so forth...

    If n_args is set to an integer greater than 0, it will check that the lowest level of lists (for all entries)
    will be of the required length
        if depth=1 as above, len([a1, a2, a3, ...]) == n_args
        if depth=2 as above, len([a1, a2, a3, ...]) == n_args && len([b1, b2, b3, ...]) == n_args
    """

    for input_list_inner in input_list:
        if isinstance(input_list_inner, list):
            depth_count += 1

    if not isinstance(input_list[0], list):
        assert all([not isinstance(input_list_inner, list) for input_list_inner in input_list])
        if n_args > 0:
            for input_list_inner in input_list:
                assert len(input_list_inner) == n_args, "Incorrect list lengths"
    else:
        depth_count += 1
        if max_depth > 0:
            assert depth_count <= max_depth, "Maximum depth exceeded"
        for input_list_inner in input_list:
            check_list_depth(input_list_inner, depth_count=depth_count)
    return depth_count
