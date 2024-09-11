def validate_mbcMode(options):
    # Extract MBC mode or set default
    if 'mbcMode' not in options:
        options['mbcMode'] = 'A'

    # Validate the MBC mode
    valid_MBC = ['A', 'O', 'P']
    if options['mbcMode'] not in valid_MBC:
        raise ValueError(f"Invalid mbc Mode '{options['mbcMode']}'. Valid mbc Modes are {valid_MBC}.")

    return options