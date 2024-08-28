import torch

def calculate_iou(preds, labels, num_classes, ignore_index=None):
    iou_list = []
    gt_classes = set(torch.unique(labels).tolist())

    with torch.no_grad():
        for cls in range(num_classes):
            if ignore_index is not None and cls == ignore_index:
                continue  # Skip the ignored class

            # Calculate intersection and union
            intersection = torch.sum((preds == cls) & (labels == cls)).item()
            union = torch.sum((preds == cls) | (labels == cls)).item()

            # Compute IoU for the class
            iou = intersection / union if union > 0 else 0
            iou_list.append((cls, iou))  # Store class ID with IoU value
            print(f"Class {cls}: Intersection = {intersection}, Union = {union}, IoU = {iou}")

    # Filter IoU list to only include classes present in ground truth and not ignored
    valid_iou_list = [(cls, iou) for cls, iou in iou_list if cls in gt_classes]

    # Compute mean IoU, excluding classes that are not present in ground truth
    if len(valid_iou_list) > 0:
        mean_iou = sum(iou for _, iou in valid_iou_list) / len(valid_iou_list)
    else:
        mean_iou = 0.0

    return mean_iou, iou_list
