import cv2
import numpy as np


def panorama_horizontal_crop(
    equirect_img: np.ndarray,
    heading_deg: float = 0.0,
    fov_deg: float = 90.0,
    output_shape: tuple = (224, 224),
) -> np.ndarray:
    """
    Extract a horizontal slice from an equirectangular panoramic image.

    Takes a simple horizontal crop centered at the heading angle, without
    perspective transformation. The crop width is determined by the FOV.

    Args:
        equirect_img: Input equirectangular image (H, W, C)
        heading_deg: Horizontal center direction in degrees (0° = start of image)
        fov_deg: Horizontal field of view in degrees (determines crop width)
        output_shape: Output image size (height, width)

    Returns:
        Cropped and resized image of shape (output_shape[0], output_shape[1], C)
    """
    h_equirect, w_equirect = equirect_img.shape[:2]
    h_out, w_out = output_shape

    # Calculate crop width based on FOV
    crop_width = int((fov_deg / 360.0) * w_equirect)

    # Calculate center position based on heading
    center_x = int((heading_deg / 360.0) * w_equirect)

    # Calculate start and end positions
    start_x = center_x - crop_width // 2
    end_x = start_x + crop_width

    # Handle wrapping for panoramic images
    if start_x < 0 or end_x > w_equirect:
        # Need to wrap around
        # Normalize start_x to be within [0, w_equirect)
        start_x = start_x % w_equirect
        end_x = start_x + crop_width

        if end_x <= w_equirect:
            # Fits after wrapping
            crop = equirect_img[:, start_x:end_x]
        else:
            # Spans across the wrap boundary
            right_part = equirect_img[:, start_x:]
            left_part = equirect_img[:, : end_x - w_equirect]
            crop = np.concatenate([right_part, left_part], axis=1)
    else:
        # Simple crop without wrapping
        crop = equirect_img[:, start_x:end_x]

    # Resize to output shape
    output_img = cv2.resize(crop, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

    return output_img


def gnomonic_projection(
    equirect_img: np.ndarray,
    heading_deg: float = 0.0,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    output_shape: tuple = (224, 224),
) -> np.ndarray:
    """
    Extract a perspective crop from an equirectangular panoramic image using gnomonic projection.

    This simulates a pinhole camera extracting a limited-FOV perspective view from a 360° panorama.

    Args:
        equirect_img: Input equirectangular image (H, W, C) where:
                     - W spans 360° horizontally (longitude: -180° to +180°)
                     - H spans 180° vertically (latitude: -90° to +90°)
        heading_deg: Horizontal viewing direction in degrees (0° = forward, 90° = right, etc.)
        pitch_deg: Vertical viewing angle in degrees (0° = horizon, +90° = up, -90° = down)
        fov_deg: Field of view in degrees (e.g., 90° for typical camera)
        output_shape: Output image size (height, width)

    Returns:
        Perspective-projected image of shape (output_shape[0], output_shape[1], C)
    """
    h_equirect, w_equirect = equirect_img.shape[:2]
    h_out, w_out = output_shape

    # Convert angles to radians
    heading_rad = np.deg2rad(heading_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    fov_rad = np.deg2rad(fov_deg)

    # Create output pixel coordinates (centered at 0)
    x_coords = np.linspace(-1, 1, w_out)
    y_coords = np.linspace(-1, 1, h_out)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Convert to 3D ray directions in camera space (gnomonic projection)
    focal_length = 1.0 / np.tan(fov_rad / 2.0)
    x_cam = x_grid
    y_cam = y_grid
    z_cam = focal_length * np.ones_like(x_grid)

    # Normalize to unit vectors
    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    # Rotation matrix for pitch (rotation around X-axis)
    rot_pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ]
    )

    # Rotation matrix for heading (rotation around Y-axis)
    rot_heading = np.array(
        [
            [np.cos(heading_rad), 0, np.sin(heading_rad)],
            [0, 1, 0],
            [-np.sin(heading_rad), 0, np.cos(heading_rad)],
        ]
    )

    # Combine rotations: first pitch, then heading
    rot_matrix = rot_heading @ rot_pitch

    # Apply rotation to ray directions
    rays = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (h_out, w_out, 3)
    rays_reshaped = rays.reshape(-1, 3)  # (h_out * w_out, 3)
    rays_rotated = (rot_matrix @ rays_reshaped.T).T  # (h_out * w_out, 3)
    rays_rotated = rays_rotated.reshape(h_out, w_out, 3)

    # Convert 3D rays to spherical coordinates
    x_rot, y_rot, z_rot = rays_rotated[..., 0], rays_rotated[..., 1], rays_rotated[..., 2]

    # Longitude (azimuth): -180° to +180°
    lon_rad = np.arctan2(x_rot, z_rot)

    # Latitude (elevation): -90° to +90°
    lat_rad = np.arcsin(np.clip(y_rot, -1.0, 1.0))

    # Map spherical coordinates to equirectangular pixel coordinates
    # Longitude: -π to π → 0 to w_equirect
    u = ((lon_rad + np.pi) / (2 * np.pi)) * w_equirect

    # Latitude: -π/2 to π/2 → h_equirect to 0 (equirectangular images have north at top)
    v = ((np.pi / 2 - lat_rad) / np.pi) * h_equirect

    # Sample from equirectangular image using bilinear interpolation
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Use OpenCV remap for efficient bilinear interpolation
    perspective_img = cv2.remap(
        equirect_img,
        u,
        v,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,  # Wrap horizontally for 360° continuity
    )

    return perspective_img
