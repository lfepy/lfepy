def validate_mask_LDN(options):
    # Extract mask method
    if 'mask' in options:
        mask = options.get('mask', 'kirsch')
    else:
        mask = options.get('mask', 'kirsch')

    options['mask'] = mask

    # Validate the mask
    valid_masks = ['gaussian', 'kirsch', 'sobel', 'prewitt']
    if options['mask'] not in valid_masks and 'mask' in options:
        raise ValueError(f"Invalid mask '{options['mask']}'. Valid masks are {valid_masks}.")

    return mask