def validate_kwargs(**kwargs):
    # Handle keyword arguments
    if kwargs is None:
        options = {}
    else:
        options = kwargs

    return options