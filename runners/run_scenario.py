# runners/run_scenario.py

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon

from config.scenario_config import TRAINING_CONFIG
from utils.barrier_control import apply_barrier_pair_states
from utils.simulation_utils import (
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


def build_geometry(config):
    _, env = load_environment(config["env_json"])

    geometry = apply_barrier_pair_states(
        env,
        config["barrier_pair_states"],
        config["barrier_pairs"],
        config["barrier_pose_config"],
    )

    return geometry


def load_detection_points(config, geometry):
    if "p2pnet_points_file" not in config:
        return []

    return load_p2pnet_points(
        config["p2pnet_points_file"],
        geometry,
        min_score=config.get("p2pnet_min_score", 0.0),
        max_agents=config.get("p2pnet_max_agents"),
    )


def build_agent_groups(config, geometry, p2pnet_positions):
    agent_groups = []
    total_agents = 0
    random_positions = []
    p2pnet_group_positions = []

    for group in config["agent_groups"]:
        start_zone = make_zone_from_fraction(
            geometry,
            group["start_box_frac"],
            safe_distance=config["safe_distance"],
        )

        goal_zone = make_zone_from_fraction(
            geometry,
            group["goal_box_frac"],
            safe_distance=config["safe_distance"],
        )

        goal_area = make_convex_goal_from_zone(
            goal_zone,
            size=0.4,
            safe_distance=config["safe_distance"],
        )

        if config["random_agents_load"]:
            random_positions = random_points(
                start_zone,
                group["count"],
                min_distance=config["min_agent_distance"],
            )
            print(f"{len(random_positions)} random agents loaded")

        if config["p2pnet_load"]:
            p2pnet_group_positions = [
                pos
                for pos in p2pnet_positions
                if start_zone.covers(Point(pos))
            ]
            print(f"{len(p2pnet_group_positions)} P2PNet agents loaded")

        if not config["random_agents_load"] and not config["p2pnet_load"]:
            raise ValueError(
                "At least one agent source must be enabled. "
                "Set 'random_agents_load=True' or 'p2pnet_load=True' in scenario_config."
            )

        positions = random_positions + p2pnet_group_positions


        agent_groups.append({
            "group_id": group["group_id"],
            "positions": positions,
            "start_zone": start_zone,
            "goal_zone": goal_zone,
            "goal_area": goal_area,
        })

        total_agents += len(positions)

    return agent_groups, total_agents


def _plot_polygon(ax, geometry, **kwargs):
    """
    Plot a Shapely Polygon or MultiPolygon.
    """
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    elif hasattr(geometry, "geoms"):
        polygons = [
            geom
            for geom in geometry.geoms
            if isinstance(geom, Polygon)
        ]
    else:
        return

    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.fill(x, y, **kwargs)

        # Plot holes inside the polygon.
        for interior in polygon.interiors:
            hole_x, hole_y = interior.xy
            ax.fill(
                hole_x,
                hole_y,
                facecolor="white",
                edgecolor="black",
                linewidth=0.6,
            )


def save_initial_positions_plot(
    geometry,
    agent_groups,
    output_file,
    title="Initial agent positions",
):
    """
    Save a plot showing:
    - the simulation geometry,
    - start zones,
    - goal zones,
    - initial agent positions.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Main environment.
    _plot_polygon(
        ax,
        geometry,
        facecolor="#eeeeee",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9,
    )

    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for index, group in enumerate(agent_groups):
        group_id = group["group_id"]
        positions = group["positions"]

        # Start zone.
        _plot_polygon(
            ax,
            group["start_zone"],
            facecolor="tab:blue",
            edgecolor="tab:blue",
            linewidth=1.0,
            alpha=0.10,
        )

        # Goal zone.
        _plot_polygon(
            ax,
            group["goal_zone"],
            facecolor="tab:green",
            edgecolor="tab:green",
            linewidth=1.0,
            alpha=0.10,
        )

        # Actual goal area used by the simulation.
        _plot_polygon(
            ax,
            group["goal_area"],
            facecolor="tab:green",
            edgecolor="tab:green",
            linewidth=1.2,
            alpha=0.35,
        )

        if positions:
            x_positions = [position[0] for position in positions]
            y_positions = [position[1] for position in positions]

            ax.scatter(
                x_positions,
                y_positions,
                s=18,
                marker=markers[index % len(markers)],
                label=f"{group_id}: {len(positions)} agents",
                alpha=0.85,
            )

    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _find_result_value(result, possible_keys, default=None):
    """
    Find a metric while allowing slightly different result-key names.
    """
    if not isinstance(result, dict):
        return default

    for key in possible_keys:
        if key in result and result[key] is not None:
            return result[key]

    return default


def _format_metric(value, decimals=3):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.{decimals}f}"

    return str(value)


def add_results_to_html(
    html_file,
    result,
    config,
    total_added,
    max_iterations,
    simulation_config,
    initial_plot_file=None,
):


    """
    Add a simulation summary to the generated animation HTML.
    """
    html_file = Path(html_file)

    if not html_file.exists():
        print("Could not add results to HTML. File not found:", html_file)
        return

    mean_velocity = _find_result_value(
        result,
        [
            "mean_velocity",
            "average_velocity",
            "avg_velocity",
            "velocity",
        ],
    )

    stop_ratio = _find_result_value(
        result,
        [
            "stop_ratio",
            "stopped_ratio",
            "stopping_ratio",
            "mean_stop_ratio",
            "average_stop_ratio",
        ],
    )

    iterations = _find_result_value(
        result,
        [
            "iterations",
            "iteration",
            "num_iterations",
            "steps",
        ],
    )



    if iterations is not None and iterations >= max_iterations:
        run_status = "Maximum iteration limit reached"
    else:
        run_status = "Completed before the iteration limit"

    # Show percentage when stop_ratio is stored between 0 and 1.
    if isinstance(stop_ratio, (int, float)):
        if 0 <= stop_ratio <= 1:
            formatted_stop_ratio = f"{stop_ratio * 100:.2f}%"
        else:
            formatted_stop_ratio = f"{stop_ratio:.2f}%"
    else:
        formatted_stop_ratio = _format_metric(stop_ratio)

    # extract the configuration values
    dt = simulation_config.get("dt")
    training_num_agents = simulation_config.get("training_num_agents")
    every_nth_frame = simulation_config.get("every_nth_frame")
    write_trajectory = simulation_config.get("write_trajectory")
    save_animation_enabled = simulation_config.get("save_animation")

    def format_boolean(value):
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        return None

    remaining_agents = result.get("remaining_agents")
    elapsed_time = result.get("elapsed_time")

    completed_agents = None

    if remaining_agents is not None:
        completed_agents = total_added - remaining_agents

    if config["random_agents_load"] and config["p2pnet_load"]:
        agent_type = "Random + P2PNet"

    elif config["random_agents_load"]:
        agent_type = "Random"

    elif config["p2pnet_load"]:
        agent_type = "P2PNet"

    else:
        raise ValueError(
            "At least one agent source must be enabled."
        )

    metrics = [
        ("Simulation time", elapsed_time),
        ("Iterations executed", iterations),
        ("Maximum iterations", max_iterations),
        ("Render every nth frame", every_nth_frame),

        ("Time step (dt)", dt),
        ("Render every nth frame", every_nth_frame),
        ("Write trajectory", format_boolean(write_trajectory)),
        ("Save animation", format_boolean(save_animation_enabled)),

        ("Configured training agents", training_num_agents),
        ("Number of agents", total_added),
        ("Added agents", agent_type),
        ("Remaining agents", remaining_agents),
        ("Completed agents", completed_agents),

        ("Mean velocity", mean_velocity),
        (
            "Stop ratio",
            formatted_stop_ratio if stop_ratio is not None else None,
        ),
        ("Run status", run_status),
    ]




    # Do not display unavailable metrics.
    metrics = [
        (name, _format_metric(value))
        for name, value in metrics
        if value is not None
    ]

    metrics_rows = "\n".join(
        f"""
        <tr>
            <td>{escape(str(name))}</td>
            <td>{escape(str(value))}</td>
        </tr>
        """
        for name, value in metrics
    )

    initial_plot_html = ""

    if initial_plot_file is not None:
        initial_plot_file = Path(initial_plot_file)

        if initial_plot_file.exists():
            # The PNG and HTML are in the same output folder.
            relative_plot_path = initial_plot_file.name

            initial_plot_html = f"""
            <h3>Initial agent positions</h3>
            <img
                src="{escape(relative_plot_path)}"
                alt="Initial agent positions"
                style="
                    display: block;
                    width: 100%;
                    max-width: 1000px;
                    margin: 12px auto;
                    border: 1px solid #dddddd;
                    border-radius: 8px;
                "
            >
            """

    summary_html = f"""
    <section
        id="simulation-results"
        style="
            max-width: 1100px;
            margin: 30px auto;
            padding: 24px;
            font-family: Arial, sans-serif;
            background: #fafafa;
            border: 1px solid #dddddd;
            border-radius: 10px;
        "
    >
        <h2 style="margin-top: 0;">Simulation results</h2>

        <table
            style="
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 24px;
            "
        >
            <thead>
                <tr>
                    <th
                        style="
                            text-align: left;
                            padding: 10px;
                            border-bottom: 2px solid #cccccc;
                        "
                    >
                        Metric
                    </th>
                    <th
                        style="
                            text-align: left;
                            padding: 10px;
                            border-bottom: 2px solid #cccccc;
                        "
                    >
                        Value
                    </th>
                </tr>
            </thead>

            <tbody>
                {metrics_rows}
            </tbody>
        </table>

        {initial_plot_html}
    </section>
    """

    html_content = html_file.read_text(encoding="utf-8")

    # Avoid adding the section twice.
    if 'id="simulation-results"' in html_content:
        print("Simulation results already exist in HTML:", html_file)
        return

    if "</body>" in html_content:
        html_content = html_content.replace(
            "</body>",
            f"{summary_html}\n</body>",
            1,
        )
    else:
        html_content += summary_html

    html_file.write_text(html_content, encoding="utf-8")


def run_scenario(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = dict(TRAINING_CONFIG)

    config["output_dir"] = str(log_dir)
    config["trajectory_file"] = str(log_dir / "trajectory.sqlite")
    config["html_file"] = str(log_dir / "animation.html")
    config["zones_plot_file"] = str(log_dir / "initial_agents.png")

    print(f"Running scenario: {config['name']}")

    geometry = build_geometry(config)

    p2pnet_positions = load_detection_points(config, geometry)

    agent_groups, total_agents = build_agent_groups(
        config,
        geometry,
        p2pnet_positions,
    )

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

    # Save the plot before the simulation starts.
    save_initial_positions_plot(
        geometry=geometry,
        agent_groups=agent_groups,
        output_file=config["zones_plot_file"],
        title=f"{config['name']} — initial agent positions",
    )

    print("Initial positions plot saved:", config["zones_plot_file"])

    simulation = create_simulation(
        geometry,
        config["trajectory_file"],
        config["simulation"],
    )

    total_added = 0

    for group in agent_groups:
        added = add_agents(
            simulation,
            group["positions"],
            group["goal_area"],
            speed_min=config["speed_min"],
            speed_max=config["speed_max"],
        )

        print(f"Added agents for {group['group_id']}:", added)
        total_added += added

    print("Total added agents:", total_added)

    result = run_simulation(
        simulation,
        config["max_iterations"],
    )

    print("Result:", result)

    save_animation(
        config["simulation"].get("every_nth_frame"),
        config["trajectory_file"],
        config["html_file"],
        title=config["name"],
    )

    # Add results and the initial-agents image to animation.html.
    add_results_to_html(
        html_file=config["html_file"],
        result=result,
        config=config,
        total_added=total_added,
        max_iterations=config["max_iterations"],
        simulation_config=config["simulation"],
        initial_plot_file=config["zones_plot_file"],
    )

    print("Scenario outputs saved in:", config["output_dir"])
    print("Animation saved:", config["html_file"])

    return {
        "result": result,
        "geometry": geometry,
        "agent_groups": agent_groups,
        "total_agents": total_agents,
        "total_added": total_added,
        "output_dir": config["output_dir"],
        "trajectory_file": config["trajectory_file"],
        "html_file": config["html_file"],
        "initial_positions_plot": config["zones_plot_file"],
    }