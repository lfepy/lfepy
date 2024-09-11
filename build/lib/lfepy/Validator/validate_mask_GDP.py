def validate_mask_GDP(options):
    # Extract mask method
    if 'mask' not in options:
        options['mask'] = 'sobel'
        t = 22.5
    elif options['mask'] == 'sobel' and 't' not in options:
        t = 22.5
    elif options['mask'] == 'prewitt' and 't' not in options:
        t = 330
    else:
        if 't' in options:
            t = options['t']
        else:
            t = 22.5

    # Validate the mask
    valid_masks = ['sobel', 'prewitt']
    if options['mask'] not in valid_masks and 'mask' in options:
        raise ValueError(f"Invalid mask '{options['mask']}'. Valid masks are {valid_masks}.")

    return options, t