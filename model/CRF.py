import numpy as np
from scipy.ndimage import gaussian_filter


class ProgressiveCRF:

    def __init__(self, config):
        self.config = config


        self.fast_crf_weight = config.get('fast_weight', 3.0)
        self.fast_color_std = config.get('fast_color_std', 5.0)
        self.fast_space_std = config.get('fast_space_std', 3.0)
        self.fast_iterations = config.get('fast_iterations', 5)


        self.adapt_weight = config.get('adapt_weight', 1.0)
        self.adapt_iterations = config.get('adapt_iterations', 5)
        self.superpixel_consistency = config.get('superpixel_consistency', 0.5)


        self.entropy_threshold = config.get('entropy_threshold', 0.5)
        self.gradient_threshold = config.get('gradient_threshold', 0.3)

    def fast_optimize(self, initial_mask, image, iterations=None):

        if iterations is None:
            iterations = self.fast_iterations

        # Convert to probabilistic map
        prob_map = self._to_probability(initial_mask)

        # Apply gaussian smoothing (simplified CRF)
        for _ in range(iterations):
            # Spatial smoothing
            smoothed = gaussian_filter(prob_map, sigma=self.fast_space_std)
            # Weighted combination
            prob_map = self.fast_crf_weight * smoothed + (1 - self.fast_crf_weight) * prob_map

        # Convert back to discrete labels
        smoothed_mask = np.argmax(prob_map, axis=2)

        return smoothed_mask

    def identify_uncertain_regions(self, mask):

        # Compute entropy
        prob_map = self._to_probability(mask)
        entropy = -np.sum(prob_map * np.log(prob_map + 1e-8), axis=2)

        # Compute gradient magnitude
        from scipy.ndimage import sobel
        grad_x = sobel(mask.astype(float), axis=0)
        grad_y = sobel(mask.astype(float), axis=1)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Combine via thresholds
        entropy_mask = entropy > self.entropy_threshold
        grad_mask = gradient_mag > self.gradient_threshold

        uncertainty_mask = entropy_mask | grad_mask

        return uncertainty_mask

    def adaptive_optimize(self, mask, image, uncertainty_mask, iterations=None):

        if iterations is None:
            iterations = self.adapt_iterations

        refined = mask.copy()

        # Only refine uncertain regions
        uncertain_indices = np.where(uncertainty_mask)

        if len(uncertain_indices[0]) == 0:
            return refined

        # Apply adaptive smoothing to uncertain regions
        prob_map = self._to_probability(refined)

        for _ in range(iterations):
            # Local spatial smoothing (simplified)
            smoothed = gaussian_filter(prob_map, sigma=1.0)
            # Combine with original
            prob_map[uncertainty_mask] = (
                    self.adapt_weight * smoothed[uncertainty_mask] +
                    (1 - self.adapt_weight) * prob_map[uncertainty_mask]
            )

        # Convert back
        refined = np.argmax(prob_map, axis=2)

        # Merge with deterministic regions
        final_mask = mask.copy()
        final_mask[uncertainty_mask] = refined[uncertainty_mask]

        return final_mask

    def _to_probability(self, mask):

        n_classes = max(mask.max() + 1, 2)
        H, W = mask.shape
        prob = np.zeros((H, W, n_classes))

        for c in range(n_classes):
            prob[:, :, c] = (mask == c).astype(float)

        # Add small epsilon for numerical stability
        prob = prob + 1e-6
        prob = prob / prob.sum(axis=2, keepdims=True)

        return prob