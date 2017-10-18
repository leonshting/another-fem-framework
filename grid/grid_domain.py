#coding - utf8
import numpy as np

from common.custom_types_for_typechecking import *


class GridDomain:

    def __init__(self, integer_size: size_nD_type, domain_size: domain_size_nD_type):
        if len(integer_size) == len(domain_size):
            self.dim = len(integer_size)
        else:
            raise Exception('Dimensions of size tuples must match.')

        self.integer_size = integer_size
        self.domain_size = domain_size

    def map_domain2grid(self, domain_point: domain_vertex_nD_type):
        """:return integer multi index of closest vertex"""
        return (np.array(self.integer_size) * (np.array(domain_point[0]) - np.array(self.domain_size[0])) /
                (np.array(self.domain_size[1]) - np.array(self.domain_size[0]))).astype(np.int)

