import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """
    Metrics:
    - mIoU: Mean Intersection over Union
    - mAcc: Mean Accuracy
    - BF1: Boundary F1 Score
    - FPs: Frames Per second (inference speed)
    - P(M): Model parameters
    """

    def __init__(self, num_classes=5, boundary_tolerance=2):
        """
        Args:
            num_classes: number of classes (including background)
            boundary_tolerance: tolerance in pixels for BF1 (default: 2)
        """
        self.num_classes = num_classes
        self.boundary_tolerance = boundary_tolerance

    def compute_iou(self, pred, target):
        """
        Args:
            pred: (H, W) prediction mask
            target: (H, W) ground truth mask

        Returns:
            iou_per_class: (num_classes,) IoU for each class
        """
        iou_list = []

        for cls in range(self.num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union == 0:
                iou = float('nan')
            else:
                iou = intersection / union

            iou_list.append(iou)

        return np.array(iou_list)

    def compute_miou(self, pred, target, ignore_nan=True):
        """
        Args:
            pred: (H, W) prediction mask
            target: (H, W) ground truth mask

        Returns:
            miou: float
        """
        iou_per_class = self.compute_iou(pred, target)

        if ignore_nan:
            iou_per_class = iou_per_class[~np.isnan(iou_per_class)]

        return np.mean(iou_per_class)

    def compute_accuracy(self, pred, target):
        """
            acc: float
        """
        correct = (pred == target).sum()
        total = pred.size
        return correct / total

    def compute_macc(self, pred, target):
        """
        Args:
            pred: (H, W) prediction mask
            target: (H, W) ground truth mask

        Returns:
            macc: float
        """
        acc_list = []

        for cls in range(self.num_classes):
            target_mask = (target == cls)
            if target_mask.sum() == 0:
                continue

            pred_mask = (pred == cls)
            correct = np.logical_and(pred_mask, target_mask).sum()
            acc = correct / target_mask.sum()
            acc_list.append(acc)

        return np.mean(acc_list)

    def compute_confusion_matrix(self, pred, target):
        """
            cm: (num_classes, num_classes) confusion matrix
        """
        return confusion_matrix(
            target.flatten(),
            pred.flatten(),
            labels=range(self.num_classes)
        )

    def extract_boundary(self, mask):
        """
        Args:
            mask: (H, W) binary mask

        Returns:
            boundary: (H, W) bool mask of boundary pixels
        """
        from scipy.ndimage import binary_erosion, generate_binary_structure

        struct = generate_binary_structure(2, 1)
        eroded = binary_erosion(mask, structure=struct)
        boundary = mask & ~eroded

        return boundary

    def compute_bf1(self, pred, target, tolerance=None):
        """
        Args:
            pred: (H, W) prediction mask
            target: (H, W) ground truth mask
            tolerance: tolerance in pixels (default: self.boundary_tolerance)

        Returns:
            bf1: float
        """
        if tolerance is None:
            tolerance = self.boundary_tolerance

        from scipy.ndimage import distance_transform_edt

        # Extract boundaries for each class
        pred_boundaries = []
        target_boundaries = []

        for cls in range(self.num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            if pred_cls.sum() > 0:
                pred_bound = self.extract_boundary(pred_cls)
            else:
                pred_bound = np.zeros_like(pred_cls, dtype=bool)

            if target_cls.sum() > 0:
                target_bound = self.extract_boundary(target_cls)
            else:
                target_bound = np.zeros_like(target_cls, dtype=bool)

            pred_boundaries.append(pred_bound)
            target_boundaries.append(target_bound)

        # Combine all boundaries
        pred_boundary = np.any(pred_boundaries, axis=0)
        target_boundary = np.any(target_boundaries, axis=0)

        # Compute distance transform for tolerance
        if target_boundary.any():
            distance = distance_transform_edt(~target_boundary)
            matched = (distance <= tolerance) & pred_boundary

            tp = matched.sum()
            fp = pred_boundary.sum() - tp
            fn = target_boundary.sum() - (matched & target_boundary).sum()
        else:
            # No target boundary
            if pred_boundary.any():
                return 0.0
            else:
                return 1.0

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        bf1 = 2 * precision * recall / (precision + recall + 1e-8)

        return bf1

    def evaluate_all(self, pred, target):
        """
        Args:
            pred: (H, W) prediction mask
            target: (H, W) ground truth mask

        Returns:
            dict: all metric values
        """
        return {
            'mIoU': self.compute_miou(pred, target),
            'mAcc': self.compute_macc(pred, target),
            'BF1': self.compute_bf1(pred, target),
            'Accuracy': self.compute_accuracy(pred, target)
        }

    @staticmethod
    def compute_parameters(model):
        """
        Args:
            model: PyTorch model

        Returns:
            params: number of parameters (in millions)
        """
        total_params = sum(p.numel() for p in model.parameters())
        return total_params / 1e6  # in millions

    @staticmethod
    def compute_fps(model, input_size=(512, 512), device='cuda', num_runs=100):
        """
        Args:
            model: PyTorch model
            input_size: (H, W) input image size
            device: device to run on
            num_runs: number of runs for averaging

        Returns:
            fps: float
        """
        import torch
        import time

        # Warm-up
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        fps = num_runs / elapsed

        return fps