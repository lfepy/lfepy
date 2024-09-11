def validate_radius(options):
    # Extract radius and compute number of neighbors or use defaults
    if 'radius' in options and options['radius'] >= 1:
        radius = options['radius']
        neighbors = 8 * radius
    else:
        radius = 1
        neighbors = 8

    return options, radius, neighbors