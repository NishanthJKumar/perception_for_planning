import numpy as np
from PIL import Image
import supervision as sv
from matplotlib import pyplot as plt, patches


def visualize_detections(
    image: Image.Image, results: list[dict], output_path: str | None = None, show_plot: bool = True
) -> tuple:
    if not results:
        print("No results to visualize")
        return None, None

    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)

    # Get image dimensions for coordinate conversion
    img_height, img_width = img_array.shape[:2]

    # Generate colors for different objects (using a more vibrant colormap)
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for i, item in enumerate(results):
        box = item.get("box_2d", [])
        label = item.get("label", "Unknown")

        if len(box) == 4:
            # Convert normalized coordinates (0-1000) to pixel coordinates
            ymin, xmin, ymax, xmax = box
            ymin = (ymin / 1000.0) * img_height
            xmin = (xmin / 1000.0) * img_width
            ymax = (ymax / 1000.0) * img_height
            xmax = (xmax / 1000.0) * img_width

            # Create rectangle patch
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin), width, height, linewidth=2, edgecolor=colors[i], facecolor="none", alpha=0.8
            )
            ax.add_patch(rect)

            # Add label with background
            ax.text(
                xmin,
                ymin - 5,
                f"{i + 1}. {label}",
                fontsize=10,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8),
            )

    ax.set_title("Bounding Box Detection Results", fontsize=14, weight="bold")
    ax.axis("off")  # Hide axes

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    # Show the plot
    if show_plot:
        plt.show()

    return fig, ax


def visualize_masks(rgb_pil: Image.Image, masks: np.ndarray, bboxes: list[dict]) -> np.ndarray:
    masks_sv = masks.squeeze(1).astype(bool)  # (num_objects, H, W)
    xyxy = []
    for mask in masks_sv:
        coords = np.column_stack(np.where(mask))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            xyxy.append([x_min, y_min, x_max, y_max])
        else:
            xyxy.append([0, 0, 0, 0])
    xyxy = np.array(xyxy)

    detections = sv.Detections(xyxy=xyxy, mask=masks_sv, class_id=np.arange(len(bboxes)))
    labels = [bbox["label"] for bbox in bboxes]

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    rgb_np = np.array(rgb_pil)
    annotated_image = mask_annotator.annotate(scene=rgb_np.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image