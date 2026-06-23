from __future__ import annotations

from typing import Dict, Iterable, Tuple

from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


def clean_geom(geom, tolerance: float = 0.0):
    geom = make_valid(geom)

    if geom.is_empty:
        return geom

    geom = geom.buffer(0)
    geom = make_valid(geom)

    # Removes tiny slivers caused by rotation/difference.
    # Keep this very small: 0.001 to 0.01 meters.
    if tolerance > 0:
        geom = geom.buffer(tolerance).buffer(-tolerance)
        geom = make_valid(geom.buffer(0))

    return geom


def largest_polygon(geom) -> Polygon:
    geom = clean_geom(geom)

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)

    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polys:
            return max(polys, key=lambda g: g.area)

    raise RuntimeError(f"Cannot convert {geom.geom_type} to Polygon.")


def load_layer(env: dict, key: str):
    return clean_geom(wkt.loads(env["geometry_wkt"][key]))


def geom_parts(geom):
    geom = clean_geom(geom)

    if geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon)]

    return []


def get_movable_barriers(env: dict) -> Dict[int, Polygon]:
    ids = [int(x) for x in env["metadata"]["movable_barrier_ids"]]

    movable_geom = load_layer(env, "movable_barriers")
    parts = geom_parts(movable_geom)

    if len(ids) != len(parts):
        raise RuntimeError(
            f"Movable barrier IDs count ({len(ids)}) does not match "
            f"movable barrier polygons count ({len(parts)})."
        )

    return {bid: clean_geom(poly) for bid, poly in zip(ids, parts)}


def rotate_barriers(
    env: dict,
    barrier_actions: Dict[int, float],
    *,
    origin: str | Tuple[float, float] = "centroid",
    clean_tolerance: float = 0.002,
) -> Dict[int, Polygon]:
    barriers = get_movable_barriers(env)
    actions = {int(k): float(v) for k, v in barrier_actions.items()}

    unknown_ids = sorted(set(actions) - set(barriers))
    if unknown_ids:
        raise KeyError(f"Unknown movable barrier IDs: {unknown_ids}")

    rotated = {}

    for bid, poly in barriers.items():
        angle = 15 #actions.get(bid, 0.0)

        if abs(angle) < 1e-9:
            rotated_poly = poly
        else:
            rotated_poly = rotate(
                poly,
                angle,
                origin=origin,
                use_radians=False,
            )

        rotated[bid] = clean_geom(rotated_poly, tolerance=clean_tolerance)

    return rotated


def apply_barrier_actions(
    env: dict,
    barrier_actions: Dict[int, float],
    *,
    origin: str | Tuple[float, float] = "centroid",
    clean_tolerance: float = 0.002,
    keep_largest: bool = True,
) -> Polygon:
    """
    Safer method:
    1. Start from original connected_geometry.
    2. Add back the original movable barrier spaces.
    3. Subtract the rotated movable barriers.

    This preserves your original scenario zones better.
    """

    connected = load_layer(env, "connected_geometry")
    original_movable = load_layer(env, "movable_barriers")

    rotated_barriers = unary_union(
        list(
            rotate_barriers(
                env,
                barrier_actions,
                origin=origin,
                clean_tolerance=clean_tolerance,
            ).values()
        )
    )

    # Restore old movable-barrier area first
    geometry = connected.union(original_movable)
    geometry = clean_geom(geometry, tolerance=clean_tolerance)

    # Then subtract the new rotated movable barriers
    geometry = geometry.difference(rotated_barriers)
    geometry = clean_geom(geometry, tolerance=clean_tolerance)

    if keep_largest:
        geometry = largest_polygon(geometry)

    if not isinstance(geometry, Polygon):
        raise RuntimeError(
            f"Final geometry is {geometry.geom_type}, but JuPedSim needs one Polygon."
        )

    return geometry
def barrier_action_vector_to_dict(
    action_vector: Iterable[float],
    barrier_ids: Iterable[int],
    *,
    max_angle_degrees: float = 20.0,
    normalized: bool = True,
) -> Dict[int, float]:
    ids = [int(x) for x in barrier_ids]
    values = list(action_vector)

    if len(ids) != len(values):
        raise ValueError(f"Expected {len(ids)} actions, got {len(values)}.")

    result = {}

    for bid, value in zip(ids, values):
        value = float(value)

        if normalized:
            value = max(-1.0, min(1.0, value)) * max_angle_degrees

        result[bid] = value

    return result