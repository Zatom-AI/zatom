"""Copyright (c) Meta Platforms, Inc. and affiliates."""

# wrap around the manifolds from geoopt
from geoopt import Euclidean, ProductManifold

from .hyperbolic import PoincareBall
from .mesh import Mesh
from .spd import SPD
from .sphere import Sphere
from .torus import FlatTorus
from .utils import geodesic
