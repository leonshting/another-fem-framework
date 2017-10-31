from typing import TypeVar, Tuple, List

numeric_type = TypeVar('numeric', int, float)

vertex_2D_type = Tuple[int, int]
vertex_3D_type = Tuple[int, int, int]
vertex_nD_type = TypeVar('vertex_nd', vertex_2D_type, vertex_3D_type)

edge_2D_type = Tuple[vertex_2D_type, vertex_2D_type]
edge_3D_type = Tuple[vertex_3D_type, vertex_3D_type]
edge_nD_type = TypeVar('edge_nd', edge_2D_type, edge_3D_type)

size_2D_type = vertex_2D_type
size_3D_type = vertex_3D_type
size_nD_type = TypeVar('size_nd', size_2D_type, size_3D_type)

domain_vertex_2D_type = Tuple[numeric_type, numeric_type]
domain_vertex_3D_type = Tuple[numeric_type, numeric_type, numeric_type]
domain_vertex_nD_type = TypeVar('domain_vertex_nd', domain_vertex_2D_type, domain_vertex_3D_type)

domain_edge_2D_type = Tuple[domain_vertex_2D_type, domain_vertex_2D_type]
domain_edge_3D_type = Tuple[domain_vertex_3D_type, domain_vertex_3D_type]
domain_edge_nD_type = TypeVar('domain_edge_nd', domain_edge_2D_type, domain_edge_3D_type)

domain_size_2D_type = domain_vertex_2D_type
domain_size_3D_type = domain_vertex_3D_type
domain_size_nD_type = TypeVar('domain_size_nd', domain_size_2D_type, domain_size_3D_type)

interpolator_orders = List[List[int]]
interpolator_sizes = List[List[Tuple[numeric_type, numeric_type]]]

