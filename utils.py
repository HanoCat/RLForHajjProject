import pathlib

import jupedsim as jps
import matplotlib.pyplot as plt
import numpy as np
import pedpy
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
from shapely import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union


def initialize_simulation(
    model, agent_parameters, geometry, goals, positions, speeds, trajectory_file
):
    simulation = jps.Simulation(
        model=model,
        geometry=geometry,
        dt=0.01,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=pathlib.Path(trajectory_file), every_nth_frame=5
        ),
    )

    exit_ids = [simulation.add_exit_stage(goal) for goal in goals]
    journey = jps.JourneyDescription(exit_ids)
    journey_id = simulation.add_journey(journey)
    centroids = [polygon.centroid.coords[0] for polygon in goals]
    orientations = [
        np.array(centroid) - np.array(position)
        for centroid, position in zip(centroids, positions)
    ]
    for pos, v0, exit_id, orientation in zip(
        positions, speeds, exit_ids, orientations
    ):
        if agent_parameters == jps.AnticipationVelocityModelAgentParameters:
            simulation.add_agent(
                agent_parameters(
                    journey_id=journey_id,
                    stage_id=exit_id,
                    desired_speed=v0,
                    position=pos,
                    anticipation_time=1,
                    reaction_time=0.3,
                )
            )
        elif agent_parameters == jps.SocialForceModelAgentParameters:
            simulation.add_agent(
                agent_parameters(
                    journey_id=journey_id,
                    stage_id=exit_id,
                    desiredSpeed=v0,
                    position=pos,
                    orientation=orientation,
                )
            )
        elif (
            agent_parameters
            == jps.GeneralizedCentrifugalForceModelAgentParameters
        ):
            simulation.add_agent(
                agent_parameters(
                    journey_id=journey_id,
                    stage_id=exit_id,
                    desired_speed=v0,
                    position=pos,
                    orientation=orientation,
                )
            )
        else:
            simulation.add_agent(
                agent_parameters(
                    journey_id=journey_id,
                    stage_id=exit_id,
                    position=pos,
                    desired_speed=v0,
                )
            )

    return simulation


def plot_simulation_configuration(geometry, starting_positions, exit_areas):
    """Plot setup for visual inspection."""
    walkable_area = pedpy.WalkableArea(geometry)
    axes = pedpy.plot_walkable_area(walkable_area=walkable_area)
    for exit_area in exit_areas:
        axes.fill(*exit_area.exterior.xy, color="indianred")

    axes.scatter(*zip(*starting_positions), label="Starting Position")
    axes.set_xlabel("x/m")
    axes.set_ylabel("y/m")
    axes.set_aspect("equal")
    axes.grid(True, alpha=0.3)


def plot_evacuation_times(times_dict, figsize=(10, 6)):
    """
    Plot evacuation times for different pedestrian models.
    """
    fig = plt.figure(figsize=figsize)

    bars = plt.bar(list(times_dict.keys()), list(times_dict.values()))

    plt.title("Evacuation Times by Model")
    plt.ylabel("Time (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    return fig


def run_simulation(simulation, max_iterations=4000):
    while (
        simulation.agent_count() > 0
        and simulation.iteration_count() < max_iterations
    ):
        simulation.iterate()
    simulation._writer.close()
    print(f"Evacuation time: {simulation.elapsed_time():.2f} s")
    return simulation.elapsed_time()


width = 600
height = 600