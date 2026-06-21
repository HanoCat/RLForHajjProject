from shapely import Polygon

def create_geometry_scenario2():
    geometry = Polygon([(-3, -2), (16, -2), (16, 2), (-3, 2)])
    return geometry


def define_positions_scenario2():
    """Define initial positions and desired speeds."""
    positions = [
        (-2, 0),
        (2, 0),
    ]
    speeds = [
        1.0,
        0.0,
    ]
    return positions, speeds


def define_goals_scenario2():
    """Define goal polygons."""
    goals = [
        Polygon([(12, -1), (15, -1), (15, 1), (12, 1)]),
        Polygon([(12, -1), (15, -1), (15, 1), (12, 1)]),
    ]
    return goals
