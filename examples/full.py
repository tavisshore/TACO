"""Full pipeline example using TACO for KITTI pose graph optimization.

This example demonstrates:
1. Loading KITTI data with IMU and image sequences
2. Integrating IMU measurements using TACO's IMUIntegrator
3. Creating a 2D pose graph with GTSAM Pose2 factors
4. Adding CVGL localization measurements
5. Optimization with Levenberg-Marquardt
6. Visualizing results

The pose graph uses Pose2 (x, y, yaw) for 2D vehicle trajectory estimation.
"""

from pathlib import Path

import numpy as np
import pypose as pp
import torch
from PIL import Image
from tqdm import tqdm

from taco import parse_args
from taco.data.kitti import Kitti, narrow_candidates_from_turns
from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.sensors.cvgl.model import ImageRetrievalModel, ImageRetrievalModelConfig
from taco.sensors.imu import detect_corners_from_gyro
from taco.utils.conversions import numpy_pose_to_gtsam, quaternion_to_yaw
from taco.visualization import plot_trajectory


def initialize_cvgl_model(checkpoint_path: Path) -> tuple[ImageRetrievalModel, torch.device]:
    """Initialize CVGL model and load checkpoint if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained weights if available
    if checkpoint_path.exists():
        print(f"   Loading CVGL model from {checkpoint_path}")
        # Use the load_from_checkpoint class method (auto-detects encoder)
        cvgl_model = ImageRetrievalModel.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            img_size=384,
            map_location=device,
        )
    else:
        print(
            "   Warning: No trained CVGL checkpoint found, creating model with pretrained encoder"
        )
        # Create config
        cvgl_config = ImageRetrievalModelConfig(
            embedding_dim=512,
            learning_rate=1e-4,
            temperature=0.07,
            loss_type="ntxent",
            weights_path="weights/sample4geo/cvusa/convnext_base.fb_in22k_ft_in1k_384/weights_e40_98.6830.pth",
        )
        # Create model with Sample4Geo encoder (default recommended)
        cvgl_model = ImageRetrievalModel.from_sample4geo(
            config=cvgl_config,
            model_name="convnext_base.fb_in22k_ft_in1k_384",
            pretrained=True,
            img_size=384,
        )

    cvgl_model.eval()
    cvgl_model = cvgl_model.to(device)

    return cvgl_model, device


def handle_cvgl_measurement(
    cvgl_enabled: bool,
    candidate_nodes: list,
    cvgl_model: ImageRetrievalModel,
    data: Kitti,
    idx: int,
    timestamp: float,
    device: torch.device,
    next_pose_gtsam,
    graph: PoseGraph,
    next_pose_id: int,
    gt_pos_meters: np.ndarray,
    num_turns: int,
) -> None:
    """Handle CVGL measurement or GPS fallback."""
    if not (cvgl_enabled and len(candidate_nodes) > 0):
        if len(candidate_nodes) > 0:
            # GPS Ground Truth fallback (when CVGL disabled)
            gt_pos_2d = np.array([gt_pos_meters[0], gt_pos_meters[1]])
            gt_yaw = next_pose_gtsam.theta()
            gt_pose_gtsam = numpy_pose_to_gtsam(gt_pos_2d, gt_yaw)
            pose_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.05]))
            graph.add_pose_factor(next_pose_id, gt_pose_gtsam, pose_noise)
        return

    # Get current frame image
    query_img = data.get_colour_img(idx)
    query_img_np = np.array(query_img)
    current_yaw = next_pose_gtsam.theta()

    # Query the CVGL database for position
    try:
        cvgl_measurement = cvgl_model.query_database_as_measurement(
            query_image=query_img_np,
            timestamp=timestamp,
            top_k=1,
            device=device,
            use_weighted_average=True,
            base_position_std=5.0,
            current_yaw=current_yaw,
        )

        # Add CVGL position measurement to pose graph
        cvgl_pose = cvgl_measurement.to_gtsam_pose(yaw=current_yaw)
        cvgl_noise = cvgl_measurement.get_gtsam_noise_model()
        print(f"   CVGL position noise stddev: {cvgl_measurement.position_std:.2f} m")
        # print("    Check changing top_k to 1")

        graph.add_pose_factor(next_pose_id, cvgl_pose, cvgl_noise)

        print(
            f"   CVGL match at turn {num_turns}: "
            f"pos=({cvgl_measurement.coordinates[0]:.7f}, {cvgl_measurement.coordinates[1]:.7f}), "
        )

    except RuntimeError as e:
        print(f"   CVGL query failed: {e}")
        # Fall back to GPS ground truth if CVGL fails
        gt_pos_2d = np.array([gt_pos_meters[0], gt_pos_meters[1]])
        gt_yaw = next_pose_gtsam.theta()
        gt_pose_gtsam = numpy_pose_to_gtsam(gt_pos_2d, gt_yaw)
        pose_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.05]))
        graph.add_pose_factor(next_pose_id, gt_pose_gtsam, pose_noise)


def main() -> None:
    """Run the full TACO pipeline on KITTI data."""
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    print("TACO Full Pipeline - KITTI Pose Graph Optimisation (Pose2)")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KITTI data
    print(f"\n1. Loading KITTI sequence {args.sequence}...")
    data = Kitti(args)
    print(f"   Loaded {len(data)} frames")

    # Initialize pose graph
    print("\n2. Initialising pose graph...")
    graph = PoseGraph()

    # Initialize CVGL model
    print("\n2.5. Initialising CVGL localization model...")
    checkpoint_path = Path(
        "weights/sample4geo/cvusa/convnext_base.fb_in22k_ft_in1k_384/weights_e40_98.6830.pth"
    )
    cvgl_model, device = initialize_cvgl_model(checkpoint_path)

    # Build reference database from graph nodes
    # Load all satellite images from the Kitti graph nodes
    print("   Building reference database from graph nodes...")
    reference_coords = []
    reference_images = []

    # Use all graph nodes as reference database
    all_nodes = list(data.graph.nodes())
    for node_id in tqdm(all_nodes, desc="   Processing reference nodes"):
        node_data = data.graph.nodes[node_id]
        lat, lon = node_data["y"], node_data["x"]

        # Get satellite image path from node data
        sat_image_path = node_data.get("sat_image")

        if sat_image_path is None:
            print(f"   Warning: Node {node_id} has no satellite image, skipping")
            continue

        # Load satellite image
        try:
            sat_img = Image.open(sat_image_path)
            sat_img_np = np.array(sat_img)  # Returns (H, W, 3) RGB [0, 255]

            # Store coordinates and image
            reference_coords.append([lat, lon])
            reference_images.append(sat_img_np)
        except Exception as e:
            print(f"   Warning: Failed to load satellite image for node {node_id}: {e}")
            continue

    reference_coords_array = np.array(reference_coords)
    cvgl_model.build_reference_database(
        images=reference_images,
        coordinates=reference_coords_array,
        use_utm=True,  # IMPORTANT: Use UTM for consistent global frame
        device=device,
        batch_size=32,
    )
    print(f"   Reference database built with {len(reference_images)} images")
    print(f"   UTM Zone: {cvgl_model.utm_zone}{cvgl_model.utm_letter}")

    cvgl_enabled = True  # Set to True when reference database is built
    use_utm_frame = True  # Use UTM coordinates for absolute measurements
    print(f"   CVGL localization: {'enabled' if cvgl_enabled else 'disabled (using GPS GT)'}")
    print(f"   Coordinate frame: {'UTM (absolute)' if use_utm_frame else 'Local tangent plane'}")

    # Get initial pose from KITTI ground truth (already in meters)
    init_values = data.get_init_value()
    init_pos = init_values["pos"][0].numpy()
    init_yaw = init_values["yaw"]

    # Get 2D pose (x, y, yaw) - positions are already in meters
    init_pos_2d = np.array([init_pos[0], init_pos[1]])

    initial_pose_gtsam = numpy_pose_to_gtsam(init_pos_2d, init_yaw)

    # Add first pose with strong prior (3 DOF for Pose2: x, y, theta)
    pose_id_0 = graph.add_pose_estimate(initial_pose_gtsam, timestamp=0.0)
    prior_noise = create_noise_model_diagonal(
        np.array([0.01, 0.01, 0.005])  # Very tight prior for x, y, theta
    )
    graph.add_prior_factor(pose_id_0, initial_pose_gtsam, prior_noise)
    print(f"   Added initial pose (ID: {pose_id_0}) with prior")

    # Initialize IMU integrator
    integrator = pp.module.IMUPreintegrator(
        init_values["pos"], init_values["rot"], init_values["vel"], reset=False
    ).to("cpu")
    imu_state = {"pos": init_values["pos"], "rot": init_values["rot"], "vel": init_values["vel"]}

    # Storage for trajectories (store as 2D in meters)
    trajectories = {
        "gt": [[init_pos[0], init_pos[1]]],
        "imu": [[init_pos[0], init_pos[1]]],
        "optimised": [[init_pos[0], init_pos[1]]],
    }

    # Process frames
    print("\n3. Processing frames...")
    current_pose = initial_pose_gtsam
    current_pose_id = pose_id_0
    num_turns = 0

    # Use length of gt_pos_meters to avoid index out of bounds
    num_frames = len(data.gt_pos_meters)
    for idx in tqdm(range(1, num_frames), desc=f"Sequence {args.sequence}"):
        timestamp = data.get_timestamp(idx)

        # Get ground truth position in meters
        gt_pos_meters = data.get_pos_meters(idx)
        trajectories["gt"].append([gt_pos_meters[0], gt_pos_meters[1]])

        # Get IMU data and integrate
        imu_data = data.get_imu(idx)
        # Get corner detection first
        turns = detect_corners_from_gyro(
            imu_data["gyro_full"][:, 2],
            imu_data["dt_full"],
            initial_heading=init_yaw,
            min_turn_angle=np.radians(25),
        )

        imu_predict = integrator(
            dt=imu_data["dt"],
            gyro=imu_data["gyro"],
            acc=imu_data["acc"],
            rot=imu_data["init_rot"],
            init_state=imu_state,
        )
        imu_state = {
            "pos": imu_predict["pos"][..., -2, :],
            "rot": imu_predict["rot"][..., -2, :],
            "vel": imu_predict["vel"][..., -2, :],
        }
        # Extract 3D position and rotation from IMU prediction
        next_pos_3d = imu_predict["pos"][..., -1, :].cpu().numpy()[0]
        next_rot_quat = (
            imu_predict["rot"][..., -1, :].cpu().numpy().flatten()
        )  # pypose quaternion [x,y,z,w]
        # Convert pypose quaternion [x,y,z,w] to [w,x,y,z] for our conversion function
        quat_wxyz = np.array(
            [next_rot_quat[3], next_rot_quat[0], next_rot_quat[1], next_rot_quat[2]]
        )
        next_yaw = quaternion_to_yaw(quat_wxyz)
        # Create gtsam.Pose2 from x, y, yaw
        next_pose_gtsam = numpy_pose_to_gtsam(next_pos_3d[:2], next_yaw)
        # Store IMU trajectory (x, y from Pose2)
        trajectories["imu"].append([next_pose_gtsam.x(), next_pose_gtsam.y()])

        # Add new pose to graph
        next_pose_id = graph.add_pose_estimate(next_pose_gtsam, timestamp=timestamp)

        # Add between factor (odometry constraint from IMU)
        # Use high uncertainty since IMU integration drifts significantly
        relative_pose = current_pose.between(next_pose_gtsam)
        odometry_noise = create_noise_model_diagonal(
            np.array([1.0, 1.0, 0.25])  # High uncertainty for x, y, theta
        )
        graph.add_between_factor(current_pose_id, next_pose_id, relative_pose, odometry_noise)

        # GT from GPS
        # NOTE: Replace with CVGL
        if len(turns.entry_angles) > num_turns:
            num_turns += 1

            # Auto-estimate distance bounds from IMU velocity and turn timing
            # (can also specify manually with min_distance_meters/max_distance_meters)
            candidate_nodes = narrow_candidates_from_turns(
                data=data,
                turns=turns,
                angle_tolerance=np.radians(20),  # radians
                verbose=False,
                # output_path=output_dir / f"frame_{len(turns.entry_angles)}.jpg",
                frame_idx=idx,
            )

            # CVGL Localization (Position-only measurement)
            handle_cvgl_measurement(
                cvgl_enabled=cvgl_enabled,
                candidate_nodes=candidate_nodes,
                cvgl_model=cvgl_model,
                data=data,
                idx=idx,
                timestamp=timestamp,
                device=device,
                next_pose_gtsam=next_pose_gtsam,
                graph=graph,
                next_pose_id=next_pose_id,
                gt_pos_meters=gt_pos_meters,
                num_turns=num_turns,
            )

        # Update current pose for next iteration
        current_pose = next_pose_gtsam
        current_pose_id = next_pose_id

        # Increment frame counter
        data.frame_id += 1

    # Optimise graph
    print("\n4. Optimising pose graph...")
    print(f"   Number of factors: {graph.size()}")
    print(f"   Initial error: {graph.get_error():.4f}")

    graph.optimize(optimizer_type="LevenbergMarquardt", max_iterations=100)
    print(f"   Final error: {graph.get_error():.4f}")

    # Extract optimised trajectory
    print("\n5. Extracting optimised trajectory...")
    optimized_poses = graph.get_all_poses()
    for pose_id in sorted(optimized_poses.keys()):
        pose = optimized_poses[pose_id]
        trajectories["optimised"].append([pose.x(), pose.y()])

    # Visualize results
    print("\n6. Visualising results...")

    # Plot ground truth trajectory (positions are already in meters)
    gt_positions = np.array(trajectories["gt"])
    fig1 = plot_trajectory(
        gt_positions,
        title=f"KITTI Sequence {args.sequence} - Ground Truth",
        show=False,
        convert_latlon=False,  # Data is already in meters
        turns=turns,
    )
    fig1.savefig(output_dir / f"gt_trajectory_seq{args.sequence}.png", dpi=150)
    print(f"   Saved gt_trajectory_seq{args.sequence}.png")

    # Plot IMU trajectory
    if len(trajectories["imu"]) > 1:
        imu_positions = np.array(trajectories["imu"])
        fig_imu = plot_trajectory(
            imu_positions,
            title=f"KITTI Sequence {args.sequence} - IMU Integrated",
            show=False,
            convert_latlon=False,  # Data is already in meters
            turns=turns,
        )
        fig_imu.savefig(output_dir / f"imu_trajectory_seq{args.sequence}.png", dpi=150)
        print(f"   Saved imu_trajectory_seq{args.sequence}.png")

    # Plot optimised trajectory
    if len(trajectories["optimised"]) > 1:
        opt_positions = np.array(trajectories["optimised"])
        fig2 = plot_trajectory(
            opt_positions,
            title=f"KITTI Sequence {args.sequence} - Optimised",
            show=False,
            convert_latlon=False,  # Data is already in meters
            turns=turns,
        )
        fig2.savefig(output_dir / f"optimised_trajectory_seq{args.sequence}.png", dpi=150)
        print(f"   Saved optimised_trajectory_seq{args.sequence}.png")

    # Save trajectories
    output_file = output_dir / f"trajectory_seq{args.sequence}.npz"
    np.savez(
        output_file,
        gt=np.array(trajectories["gt"]),
        imu=np.array(trajectories["imu"]) if trajectories["imu"] else np.array([]),
        optimised=np.array(trajectories["optimised"]),
    )
    print(f"   Saved trajectories to {output_file}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
