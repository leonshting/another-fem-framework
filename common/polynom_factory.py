# coding: utf-8

import numpy as np
import itertools
import sympy
from sympy.core import S, Dummy, pi
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.rootoftools import RootOf

def gauss_lobatto(n, n_digits=10):
    x = Dummy("x")
    p = legendre_poly(n - 1, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in pd.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S(1) / 10 ** (n_digits + 2))
        xi.append(r.n(n_digits))
        w.append((2 / (n * (n - 1) * p.subs(x, r) ** 2)).n(n_digits))

    xi.insert(0, -1)
    xi.append(1)
    w.insert(0, (S(2) / (n * (n - 1))).n(n_digits))
    w.append((S(2) / (n * (n - 1))).n(n_digits))
    return xi, w

def populate_size(dim, size):
    shape = np.array(size).shape
    if (len(shape) == 1):
        size = (min(size), max(size))
        return tuple(dim * [size])
    elif (len(shape) == dim):
        size = tuple([(min(s), max(s)) for s in size])
        return size
    else:
        raise BaseException("invalid dimensions")

def gl_points(order, size):
    to_mult = (size[1] - size[0]) / 2.
    return np.array([float(size[0])] + (
    list(((np.polynomial.legendre.leggauss(order - 1)[0]) + 1) * to_mult + size[0]) if order > 1 else []) + [
                        float(size[1])])

def gc_points(order, size):
    to_mult = (size[1] - size[0]) / 2.
    roots_arr = np.sort(np.polynomial.chebyshev.chebgauss(order - 1)[0])
    return np.array(
        [float(size[0])] + (list((roots_arr + 1) * to_mult + size[0]) if order > 1 else []) + [float(size[1])])

def glob_points(order, size):
    roots_arr = np.array(gauss_lobatto(n=order + 1)[0], dtype=np.float32)
    to_mult = (size[1] - size[0]) / 2.
    return np.array(list((roots_arr + 1) * to_mult + size[0]))

def glob_weights(order, size=(0,1)):
    return (max(size) - min(size)) * np.array(gauss_lobatto(n=order + 1)[1], dtype=np.float32)

def gen_nodes(order, dim, size, distribution='uniform'):
    # order - polynomial power
    # dim - number of dimensions
    nodes = []
    sizes = populate_size(dim=dim, size=size)
    ls = []
    if distribution == 'uniform':
        ls = [np.linspace(size[0], size[1], num=order + 1) for size in sizes]
    elif distribution == 'gl':
        ls = [gl_points(order, size) for size in sizes]
    elif distribution == 'gc':
        ls = [gc_points(order, size) for size in sizes]
    elif distribution == 'globatto':
        ls = [glob_points(order, size) for size in sizes]
    for i in itertools.product(*ls):
        nodes.append(i)
    return nodes

def iterate_powers(order, dim):
    ## ditributes total order b/w dimensions
    for i in itertools.product(range(order + 1), repeat=dim):
        yield i

def polynom_factory(order=2, dim=2, size=(0., 1.), distribution='uniform', ret_argmatrix=False):
    # providing list of lambdas defined on unit square - [0,1]^2 or another customizable size
    nodes = gen_nodes(order=order, dim=dim, size=size, distribution=distribution)
    M = []
    for node in nodes:
        M.append(
            [np.multiply.reduce([node_point ** power for node_point, power in zip(node, power_comb)]) for power_comb in
             iterate_powers(order=order, dim=dim)])
    M = np.array(M)
    argss = []
    for num, node in enumerate(nodes):
        rhs = np.zeros((order + 1) ** dim)
        rhs[num] = 1.
        argss.append(np.linalg.solve(M, rhs))
    funcs = []
    for args in argss:
        funcs.append(polynom_lambda(order=order, dim=dim, coef=args))

    if ret_argmatrix:
        return np.array(argss), nodes
    else:
        return funcs, nodes

def polynom_lambda(order, dim, **kwargs):
    # makes symbolic function from set of coeffs
    symbols = [sympy.Symbol('x_{}'.format(i)) for i in range(1, dim + 1)]
    func = sympy.Function('f')
    func = 0
    for arg, power in zip(kwargs['coef'], iterate_powers(order=order, dim=dim)):
        tmp_func = sympy.Function('t')
        tmp_func = 1
        for sym, power in zip(symbols, power):
            tmp_func *= sym ** power
        func += arg * tmp_func
    return func

def gradient(symfunc, symbols):
    return [sympy.diff(symfunc, symbol) for symbol in symbols]

def local_gradgrad_matrix(order, dim, size=(0., 1.), distribution='uniform'):
    # gradgrad matrix for set of basis functions
    sizes = populate_size(dim=dim, size=size)
    symbols = [sympy.Symbol('x_{}'.format(i)) for i in range(1, dim + 1)]
    shape = (order + 1) ** dim
    ret_M = np.zeros((shape, shape))
    ret_indices = {}
    funcs, nodes = polynom_factory(order=order, dim=dim, size=size, distribution=distribution)
    for num, node in enumerate(nodes):
        ret_indices[node] = num
    gradients = [gradient(func, symbols) for func in funcs]
    for bundle in itertools.product(enumerate(gradients), repeat=2):
        ret_M[bundle[0][0], bundle[1][0]] = sympy.integrate(sum([i * j for i, j in zip(bundle[0][1], bundle[1][1])]),
                                                            *[(symbol, size[0], size[1]) for symbol, size in
                                                              zip(symbols, sizes)])
    return (ret_M, ret_indices)

def local_index(order, dim, size=(0., 1.), distribution='uniform'):
    ret_indices = {}
    sizes = populate_size(dim=dim, size=size)
    funcs, nodes = polynom_factory(order=order, dim=dim, size=sizes, distribution=distribution)
    for num, node in enumerate(nodes):
        ret_indices[node] = num
    return ret_indices

def local_gradients(order, dim, size=(0., 1.), distribution='uniform'):
    # gradgrad matrix for set of basis functions
    sizes = populate_size(dim=dim, size=size)
    symbols = [sympy.Symbol('x_{}'.format(i)) for i in range(1, dim + 1)]
    ret_indices = {}
    funcs, nodes = polynom_factory(order=order, dim=dim, size=sizes, distribution=distribution)
    for num, node in enumerate(nodes):
        ret_indices[node] = num
    gradients = [gradient(func, symbols) for func in funcs]
    return gradients

def local_gradgrad_functions(order, dim, size=(0., 1.), distribution='uniform'):
    sizes = populate_size(dim=dim, size=size)
    symbols = [sympy.Symbol('x_{}'.format(i)) for i in range(1, dim + 1)]
    ret_indices = {}
    funcs, nodes = polynom_factory(order=order, dim=dim, size=sizes, distribution=distribution)
    for num, node in enumerate(nodes):
        ret_indices[node] = num
    gradients = [gradient(func, symbols) for func in funcs]
    ret_list = {}
    for bundle in itertools.product(enumerate(gradients), repeat=2):
        ret_list[(bundle[0][0], bundle[1][0])] = (sum([i * j for i, j in zip(bundle[0][1], bundle[1][1])]))
    return ret_list, symbols


def local_funcfunc_matrix(order, dim, size=(0., 1.), distribution='uniform'):
    # funcfunc matrix for set of basis functions
    sizes = populate_size(dim=dim, size=size)
    symbols = [sympy.Symbol('x_{}'.format(i)) for i in range(1, dim + 1)]
    shape = (order + 1) ** dim
    ret_M = np.zeros((shape, shape))
    funcs, nodes = polynom_factory(order=order, dim=dim, size=size, distribution=distribution)
    for bundle in itertools.product(enumerate(funcs), repeat=2):
        ret_M[bundle[0][0], bundle[1][0]] = sympy.integrate \
            (bundle[0][1] * bundle[1][1], *[(symbol, size[0], size[1]) \
                                            for symbol, size in zip(symbols, sizes)])
    return ret_M

def norm_vector(x, y):
    return np.array([x, y]) / np.sqrt(x ** 2 + y ** 2)


def normed_normal(x1, x2, y1, y2):
    return norm_vector(y1 - y2, x2 - x1)


def normed_colin(x1, x2, y1, y2):
    return norm_vector(x2 - x1, y2 - y1)

def gen_borders(dim, size=(0., 1.)):
    # return border nodes in a clockwise order for 2D
    # else do not know
    sizes = populate_size(dim=dim, size=size)
    nodes = []
    if (dim == 2):
        clockwise = [(0, 0), (0, 1), (1, 1), (1, 0)]
        for node in clockwise:
            nodes.append((sizes[0][node[0]], sizes[1][node[1]]))
    else:
        raise BaseException("not implemented yet")
    return nodes


def local_gradfunc_matrix(order, dim, size=(0., 1.), distribution='globatto'):
    # fluxes matrix for set of basis functions
    # returns ij-matrix for complete border and set (border,matrix)
    sizes = populate_size(dim=dim, size=size)
    symbols = [sympy.Symbol('x_{}'.format(i)) for i in range(1, dim + 1)]
    shape = (order + 1) ** dim
    borderwise = []
    result = np.zeros((shape, shape))
    if (dim == 2):
        functions, nodes = polynom_factory(order=order, dim=dim, size=size, distribution=distribution)
        gradients = [gradient(func, symbols) for func in functions]
        borders = gen_borders(dim=dim, size=sizes)
        for num, border in enumerate(zip(borders, borders[1:] + [borders[0]])):
            result_for_border = np.zeros((shape, shape))
            normal = normed_normal(border[0][0], border[0][1], border[1][0], border[1][1])
            # is working surely only for rectangles with OX OY sides
            for (numfunc, func), (numgrad, grad) in itertools.product(enumerate(functions), enumerate(gradients)):
                grad_normal = sum([i * j for i, j in zip(normal, grad)])
                to_integrate = (grad_normal * func).subs(symbols[num % 2], border[num % 2][0])
                integral = sympy.integrate(to_integrate, (
                symbols[1 - (num % 2)], min(border[1 - (num % 2)]), max(border[1 - (num % 2)])))
                result_for_border[numfunc, numgrad] += integral
                result[numfunc, numgrad] += integral
            borderwise.append(result_for_border)
        return (result, borderwise)
    else:
        raise BaseException("not implemented yet")


def get_href_constraint_matrix(order, inversed=False, distribution='globatto'):
    glob_func_on_loc_mesh = np.zeros((order+1, order+1))
    funcs_prim, glob_grid = polynom_factory(dim=1, order=order,size=(0,2), distribution=distribution)
    funcs, local_grid = polynom_factory(dim=1, order=order,size=(0,1), distribution=distribution)
    for num1, i in enumerate(local_grid):
        for num2, prim_f in enumerate(funcs_prim):
            glob_func_on_loc_mesh[num1, num2] = prim_f.subs({'x_1':i[0]})
    if(inversed):
        return np.linalg.inv(glob_func_on_loc_mesh)
    else:
        return glob_func_on_loc_mesh