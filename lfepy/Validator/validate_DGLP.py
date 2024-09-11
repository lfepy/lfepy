def validate_DGLP(options):
    # Extract the DGLP
    if 'DGLP' not in options:
        options.update({'DGLP': 0})

    # Validate the DGLP
    valid_DGLP = [0, 1]
    if options['DGLP'] not in valid_DGLP:
        raise ValueError(f"Invalid DGLP '{options['DGLP']}'. Valid DGLP are {valid_DGLP}.")

    return options