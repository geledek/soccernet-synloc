"""
Pitch visualization utilities.

Draw soccer pitch and visualize BEV predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle
from typing import Optional, Tuple, List
import torch


def draw_pitch(
    ax: Optional[plt.Axes] = None,
    pitch_color: str = 'green',
    line_color: str = 'white',
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Axes:
    """Draw a soccer pitch.

    Pitch dimensions: 105m x 68m (standard FIFA)
    Origin at center of pitch.

    Args:
        ax: Matplotlib axes (creates new if None).
        pitch_color: Pitch background color.
        line_color: Line color.
        figsize: Figure size if creating new.

    Returns:
        Matplotlib axes with pitch drawn.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Pitch dimensions (half-values from center)
    pitch_length = 105  # meters
    pitch_width = 68    # meters
    half_length = pitch_length / 2
    half_width = pitch_width / 2

    # Penalty area dimensions
    penalty_area_length = 16.5
    penalty_area_width = 40.3
    goal_area_length = 5.5
    goal_area_width = 18.3

    # Set background
    ax.set_facecolor(pitch_color)

    # Pitch outline
    ax.plot([-half_length, half_length], [-half_width, -half_width], line_color, lw=2)
    ax.plot([-half_length, half_length], [half_width, half_width], line_color, lw=2)
    ax.plot([-half_length, -half_length], [-half_width, half_width], line_color, lw=2)
    ax.plot([half_length, half_length], [-half_width, half_width], line_color, lw=2)

    # Center line
    ax.plot([0, 0], [-half_width, half_width], line_color, lw=2)

    # Center circle
    center_circle = Circle((0, 0), 9.15, fill=False, color=line_color, lw=2)
    ax.add_patch(center_circle)

    # Center spot
    ax.plot(0, 0, 'o', color=line_color, markersize=3)

    # Left penalty area
    ax.plot([-half_length, -half_length + penalty_area_length],
            [-penalty_area_width/2, -penalty_area_width/2], line_color, lw=2)
    ax.plot([-half_length, -half_length + penalty_area_length],
            [penalty_area_width/2, penalty_area_width/2], line_color, lw=2)
    ax.plot([-half_length + penalty_area_length, -half_length + penalty_area_length],
            [-penalty_area_width/2, penalty_area_width/2], line_color, lw=2)

    # Right penalty area
    ax.plot([half_length, half_length - penalty_area_length],
            [-penalty_area_width/2, -penalty_area_width/2], line_color, lw=2)
    ax.plot([half_length, half_length - penalty_area_length],
            [penalty_area_width/2, penalty_area_width/2], line_color, lw=2)
    ax.plot([half_length - penalty_area_length, half_length - penalty_area_length],
            [-penalty_area_width/2, penalty_area_width/2], line_color, lw=2)

    # Left goal area
    ax.plot([-half_length, -half_length + goal_area_length],
            [-goal_area_width/2, -goal_area_width/2], line_color, lw=2)
    ax.plot([-half_length, -half_length + goal_area_length],
            [goal_area_width/2, goal_area_width/2], line_color, lw=2)
    ax.plot([-half_length + goal_area_length, -half_length + goal_area_length],
            [-goal_area_width/2, goal_area_width/2], line_color, lw=2)

    # Right goal area
    ax.plot([half_length, half_length - goal_area_length],
            [-goal_area_width/2, -goal_area_width/2], line_color, lw=2)
    ax.plot([half_length, half_length - goal_area_length],
            [goal_area_width/2, goal_area_width/2], line_color, lw=2)
    ax.plot([half_length - goal_area_length, half_length - goal_area_length],
            [-goal_area_width/2, goal_area_width/2], line_color, lw=2)

    # Penalty spots
    ax.plot(-half_length + 11, 0, 'o', color=line_color, markersize=3)
    ax.plot(half_length - 11, 0, 'o', color=line_color, markersize=3)

    # Penalty arcs
    left_arc = Arc((-half_length + 11, 0), 18.3, 18.3, angle=0,
                   theta1=-53, theta2=53, color=line_color, lw=2)
    right_arc = Arc((half_length - 11, 0), 18.3, 18.3, angle=180,
                    theta1=-53, theta2=53, color=line_color, lw=2)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # Set axis properties
    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')

    return ax


def visualize_bev_predictions(
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    pred_color: str = 'red',
    gt_color: str = 'blue',
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> plt.Axes:
    """Visualize BEV predictions on pitch.

    Args:
        predictions: Predicted positions (N, 2) in world coordinates.
        ground_truth: Ground truth positions (M, 2) in world coordinates.
        ax: Matplotlib axes.
        pred_color: Color for predictions.
        gt_color: Color for ground truth.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    ax = draw_pitch(ax=ax, figsize=figsize)

    # Plot predictions
    if len(predictions) > 0:
        ax.scatter(
            predictions[:, 0], predictions[:, 1],
            c=pred_color, s=100, marker='o',
            edgecolors='white', linewidths=2,
            label='Predictions', zorder=10
        )

    # Plot ground truth
    if ground_truth is not None and len(ground_truth) > 0:
        ax.scatter(
            ground_truth[:, 0], ground_truth[:, 1],
            c=gt_color, s=100, marker='x',
            linewidths=3,
            label='Ground Truth', zorder=9
        )

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(loc='upper right')

    return ax


def visualize_frame_comparison(
    image: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    camera_matrix: torch.Tensor,
    undist_poly: torch.Tensor,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """Visualize image alongside BEV comparison.

    Args:
        image: Input image (H, W, 3).
        predictions: Predicted BEV positions (N, 2).
        ground_truth: Ground truth BEV positions (M, 2).
        camera_matrix: Camera projection matrix.
        undist_poly: Undistortion polynomial.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Image view
    axes[0].imshow(image)
    axes[0].set_title('Camera View')
    axes[0].axis('off')

    # BEV view
    visualize_bev_predictions(
        predictions, ground_truth,
        ax=axes[1],
        title='Bird\'s Eye View'
    )

    plt.tight_layout()
    return fig
