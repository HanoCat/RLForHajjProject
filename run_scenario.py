
import matplotlib.pyplot as plt

from scenario_config import SCENARIO
from barrier_control import apply_barrier_pair_states
from shapely.geometry import Point
from simulation_utils import (
    load_environment,
    make_zone_from_fraction,
    make_convex_goal_from_zone,
    random_points,
    load_p2pnet_points,
    create_simulation,
    add_agents,
    run_simulation,
    save_animation,
)


geometry, env = load_environment(SCENARIO["env_json"])

# Apply dynamic movable-barrier orientation before creating zones and simulation.
# For RL integration later, replace SCENARIO["barrier_actions"] with the action output.

print("Barrier pair states:", SCENARIO["barrier_pair_states"])

geometry = apply_barrier_pair_states(
    env,
    SCENARIO["barrier_pair_states"],
    SCENARIO["barrier_pairs"],
    SCENARIO["barrier_pose_config"],
)

agent_groups = []
total_agents = 0

p2pnet_positions = []

if "p2pnet_points_file" in SCENARIO:
        p2pnet_positions = load_p2pnet_points(
            SCENARIO["p2pnet_points_file"],
            geometry,
            min_score=SCENARIO.get("p2pnet_min_score", 0.0),
            max_agents=SCENARIO.get("p2pnet_max_agents"),
        )
        print("Has p2pnet key:", "p2pnet_points_file" in SCENARIO)
        print("P2PNet loaded before grouping:", len(p2pnet_positions))



for group in SCENARIO["agent_groups"]:
    start_zone = make_zone_from_fraction(
        geometry,
        group["start_box_frac"],
        safe_distance=SCENARIO["safe_distance"],
    )

    goal_zone = make_zone_from_fraction(
        geometry,
        group["goal_box_frac"],
        safe_distance=SCENARIO["safe_distance"],
    )

    goal_area = make_convex_goal_from_zone(
        goal_zone,
        size=0.4,
        safe_distance=SCENARIO["safe_distance"],
    )

    random_positions = random_points(
        start_zone,
        group["count"],
        min_distance=SCENARIO["min_agent_distance"],
    )

    p2pnet_group_positions = [
        pos for pos in p2pnet_positions
        if start_zone.covers(Point(pos))
    ]
    print(group["group_id"], "matched p2pnet:", len(p2pnet_group_positions))

    positions = random_positions + p2pnet_group_positions


    ''' 
    mid_zone = None
    mid_points = None

    if "mid_box_frac" in group:
        mid_zone = make_zone_from_fraction(
            geometry,
            group["mid_box_frac"],
            safe_distance=SCENARIO["safe_distance"],
        )

        mid_points = random_points(
            mid_zone,
            len(positions),
            min_distance=0.0,
        )
    '''

    agent_groups.append({
        "group_id": group["group_id"],
        "positions": positions,
        "start_zone": start_zone,
        "goal_zone": goal_zone,
        "goal_area": goal_area,
    })

    total_agents += len(positions)

print("Geometry bounds:", geometry.bounds)
print("First p2pnet point:", p2pnet_positions[0] if p2pnet_positions else None)

print("Scenario:", SCENARIO["name"])
print("Geometry area:", round(geometry.area, 2))
print("Groups:", len(agent_groups))
print("Total agents:", total_agents)

for group in agent_groups:
    print(
        group["group_id"],
        "| agents:", len(group["positions"]),
        "| start area:", round(group["start_zone"].area, 2),
        "| goal area:", round(group["goal_area"].area, 2),
    )


fig, ax = plt.subplots(figsize=(14, 7))

x, y = geometry.exterior.xy
ax.fill(x, y, alpha=0.20, label="environment")

for group in agent_groups:
    sx, sy = group["start_zone"].exterior.xy
    ax.plot(sx, sy, linewidth=2, label=f"{group['group_id']} start")

    gzx, gzy = group["goal_zone"].exterior.xy
    ax.plot(
        gzx,
        gzy,
        linewidth=2,
        linestyle="--",
        label=f"{group['group_id']} goal zone",
    )

    gx, gy = group["goal_area"].exterior.xy
    ax.fill(gx, gy, alpha=0.5)

    positions = group["positions"]
    ax.scatter(
        [p[0] for p in positions],
        [p[1] for p in positions],
        s=18,
        label=f"{group['group_id']} agents",
    )

    ''' 
    if group["mid_zone"] is not None:
        mx, my = group["mid_zone"].exterior.xy
        ax.plot(
            mx,
            my,
            linewidth=2,
            linestyle=":",
            label=f"{group['group_id']} mid zone",
        )

        mid_points = group["mid_points"]
        ax.scatter(
            [p[0] for p in mid_points],
            [p[1] for p in mid_points],
            s=12,
            marker="x",
            label=f"{group['group_id']} mid points",
        )
    '''
ax.set_aspect("equal")
#ax.legend(fontsize=8)
ax.set_title(SCENARIO["name"])
plt.show()


simulation = create_simulation(
    geometry,
    SCENARIO["trajectory_file"],
    dt=0.05,
)

total_added = 0

for group in agent_groups:
    added = add_agents(
        simulation,
        group["positions"],
        group["goal_area"],
        speed_min=SCENARIO["speed_min"],
        speed_max=SCENARIO["speed_max"],
    )

    print(f"Added agents for {group['group_id']}:", added)
    total_added += added

print("Total added agents:", total_added)

result = run_simulation(
    simulation,
    SCENARIO["max_iterations"],
)

print("Result:", result)

save_animation(
    SCENARIO["every_nth_frame_n"],
    SCENARIO["trajectory_file"],
    SCENARIO["html_file"],
    title=SCENARIO["name"],
)

print("Animation saved:", SCENARIO["html_file"])