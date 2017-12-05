def partial_diff_test(pointnum, point, matrix, pointdict, mass_matrix=None,
                      multiple_dofs_on_point=True, scale_factor=1):
    zero_der = 0
    x_der = 0
    xx_der = 0
    yy_der = 0
    y_der = 0
    xy_der = 0

    if not multiple_dofs_on_point:
        for k, v in pointdict.items():
            dx = point[0] - k[0]
            dy = point[1] - k[1]

            zero_der += matrix[pointnum, v]

            x_der += dx * matrix[pointnum, v]
            xx_der += matrix[pointnum, v] * dx ** 2 / 2
            y_der += dy * matrix[pointnum, v]
            yy_der += dy ** 2 / 2 * matrix[pointnum, v]
            xy_der += dx * dy * matrix[pointnum, v]
    else:
        for k, vs in pointdict.items():
            for v in vs:
                dx = point[0] - k[0]
                dy = point[1] - k[1]
                zero_der += matrix[pointnum, v]
                x_der += dx * matrix[pointnum, v]
                xx_der += matrix[pointnum, v] * dx ** 2 / 2
                y_der += dy * matrix[pointnum, v]
                yy_der += dy ** 2 / 2 * matrix[pointnum, v]
                xy_der += dx * dy * matrix[pointnum, v]

    if mass_matrix is not None:
        print('Mass coef: {}'.format(mass_matrix[pointnum].sum()))

    print('zero derivative: {}'.format(zero_der * scale_factor))
    print('X derivative: {}'.format(x_der * scale_factor))
    print('Y derivative: {}'.format(y_der * scale_factor))
    print('XX derivative: {}'.format(xx_der * scale_factor))
    print('YY derivative: {}'.format(yy_der * scale_factor))
    print('XY_derivative: {}'.format(xy_der * scale_factor), end='\n\n')
    return {
        'zero': zero_der * scale_factor,
        'XX': xx_der * scale_factor,
        'YY': yy_der * scale_factor,
        'X': x_der * scale_factor,
        'Y': y_der * scale_factor,
            }
