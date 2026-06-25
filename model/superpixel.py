import numpy as np
import torch
from skimage.segmentation import slic
from skimage import color


class AdaptiveSuperpixelSegmentation:

    def __init__(self, config):
        self.config = config
        self.ndvi_soil = config.get('ndvi_soil', 0.1)  # NDVI_soil
        self.ndvi_veg = config.get('ndvi_veg', 0.8)  # NDVI_veg
        self.base_superpixels_per_pixel = config.get('base_per_pixel', 4000)
        self.min_superpixels = config.get('min_count', 30)
        self.max_superpixels = config.get('max_count', 100)
        self.base_threshold = config.get('base_threshold', 0.25)
        self.noise_threshold = config.get('noise_threshold', 0.1)

    def compute_fvc(self, image: np.ndarray) -> np.ndarray:
        # Convert RGB to float and get R and NIR approximations
        # For RGB-only images, use approximation or color-space based methods
        image_float = image.astype(np.float32) / 255.0
        red = image_float[:, :, 0]
        green = image_float[:, :, 1]

        # Simulate NIR from RGB (for demonstration)
        # In actual implementation, NIR band should be provided
        nir = np.clip(0.5 * red + 0.4 * green + 0.1, 0, 1)

        # Compute NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)

        # Compute FVC using pixel dichotomy model
        fvc = (ndvi - self.ndvi_soil) / (self.ndvi_veg - self.ndvi_soil + 1e-8)
        fvc = np.clip(fvc, 0, 1)

        return fvc

    def adaptive_count(self, fvc_map: np.ndarray) -> int:

        H, W = fvc_map.shape
        total_pixels = H * W

        # Global average FVC
        mean_fvc = np.mean(fvc_map)

        # Adaptive count
        base = total_pixels / self.base_superpixels_per_pixel
        n_seg = int(base * (1 + mean_fvc))
        n_seg = max(self.min_superpixels, min(self.max_superpixels, n_seg))

        return n_seg

    def segment(self, image: np.ndarray, n_seg: int) -> np.ndarray:

        compactness = self.config.get('compactness', 10)
        sigma = self.config.get('sigma', 1)

        # Use SLIC superpixel segmentation
        segments = slic(
            image,
            n_segments=n_seg,
            compactness=compactness,
            sigma=sigma,
            start_label=1
        )
        return segments

    def extract_region_features(self, image: np.ndarray, superpixel_map: np.ndarray):

        regions = {}
        n_regions = superpixel_map.max() + 1

        for idx in range(1, n_regions):
            mask = (superpixel_map == idx)
            if not mask.any():
                continue

            # Get region pixels and bounding box
            coords = np.where(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Extract region image
            region_patch = image[y_min:y_max + 1, x_min:x_max + 1]

            # For demonstration, use color histogram features
            features = self._extract_features(region_patch)

            regions[idx] = {
                'features': features,
                'mask': mask,
                'bbox': (x_min, y_min, x_max, y_max),
                'size': mask.sum()
            }

        return regions

    def _extract_features(self, patch):
        """
        Extract feature vector from region patch.
        In practice, uses CLIP visual encoder.
        """
        # Simplified feature extraction for demonstration
        # Actual implementation uses CLIP visual encoder
        hist_r = np.histogram(patch[:, :, 0], bins=16, range=(0, 255))[0]
        hist_g = np.histogram(patch[:, :, 1], bins=16, range=(0, 255))[0]
        hist_b = np.histogram(patch[:, :, 2], bins=16, range=(0, 255))[0]
        features = np.concatenate([hist_r, hist_g, hist_b])
        features = features / (features.sum() + 1e-8)
        return features

    def region_semantic_matching(self, region_features, semantic_centers,
                                 fvc_map, superpixel_map):

        H, W = fvc_map.shape
        mask = np.zeros((H, W), dtype=np.int32)

        categories = list(semantic_centers.keys())
        centers = list(semantic_centers.values())

        for region_id, data in region_features.items():
            # Compute regional vegetation density
            region_mask = data['mask']
            region_fvc = fvc_map[region_mask].mean()
            rho_k = region_fvc

            # Compute similarity with each category
            feats = data['features']
            similarities = []
            for center in centers:
                # Cosine similarity (simplified)
                sim = np.dot(feats, center) / (np.linalg.norm(feats) * np.linalg.norm(center) + 1e-8)
                similarities.append(sim)

            similarities = np.array(similarities)
            max_sim = similarities.max()
            best_idx = np.argmax(similarities)

            # Dynamic threshold
            gamma_k = self.base_threshold + 0.15 * (0.5 - rho_k)


            if max_sim > gamma_k and rho_k > self.noise_threshold:
                assigned_class = best_idx + 1  # 1-indexed classes
                mask[region_mask] = assigned_class
            else:
                mask[region_mask] = 0  # non-vegetation

        return mask