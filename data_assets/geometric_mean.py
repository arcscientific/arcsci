"""
Python Implementation of Analytic Framework for Evaluating Precision of Ancient Egyptian Artifacts

This software implements the analytic framework developed by Karoly Poka, for
assessing the geometric precision and craftsmanship quality of ancient Egyptian stone vessels
through 3D scanning analysis. The framework evaluates fashioning quality by analyzing
cross-sectional circularity and concentricity along the vessel height.

Geometric Mean Score - Defined calculation steps:
1) Slice artifact in z-plane (thickness not fixed, custom input variable)
   For each slice:
   - Fit circle using least squares
   - Calculate circularity error (RMSD of fit)
   - Compute concentricity ΔC = √(x_c² + y_c²)
2) Determine median circularity and median concentricity
3) Calculate composite score: √(circularity_median × concentricity_median)

Extra filters have been added to only compute slices with more than 50 data-points, 
this requirement was not defined by Karoly but was added to ensure fit quality.

Link to video where Karoly Poka presents his data and findings using this metric:
https://www.youtube.com/watch?v=8d752WFDL24

Link to article by Stine Gerdes evaluating and rejecting this metric as valid for metrology analysis:
https://arcsci.org/articles/comparative_metrological_analysis.html

Copyright (c) 2025 Stine Gerdes (arcsci.org)

License: CC BY-NC-SA 4.0

"""

import numpy as np
from skimage.measure import CircleModel, ransac
from stl import mesh
import os
import warnings


class CircleFit:
    def __init__(self):
        self.r = None
        self.xc = None
        self.yc = None
        self.inliers = None
        self._model = None

    def fit(self, sample_coords, use_ransac=False, residuals_threshold=None):
        """
        Fit a circle to 2D points.

        Parameters:
        - sample_coords: ndarray of shape (N, 2) - the points to fit.
        - use_ransac: bool - whether to use RANSAC for robust fitting.
        - residuals_threshold: float or None - threshold for inliers in RANSAC.

        Returns:
        - r: radius of the fitted circle
        - xc: x-coordinate of the center
        - yc: y-coordinate of the center
        - inliers: boolean array indicating inliers (None if use_ransac=False)
        """
        if use_ransac:
            result = self._fit_ransac(sample_coords, residuals_threshold)
        else:
            result = self._fit_standard(sample_coords)
        
        # Store results on the class
        if result is not None:
            if len(result) == 4:  # RANSAC result
                self.r, self.xc, self.yc, self.inliers = result
            else:  # Standard result (3 elements)
                self.r, self.xc, self.yc = result
                self.inliers = None
        else:
            self.r, self.xc, self.yc, self.inliers = None, None, None, None
        
        return self.r, self.xc, self.yc, self.inliers

    def _fit_standard(self, sample_coords):
        """Fit a circle without RANSAC."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CircleModel()
            success = model.estimate(sample_coords)
            if not success:
                return None, None, None
            xc, yc, r = model.params

            # Store model for residuals calculation
            self._model = model
            return r, xc, yc

    def _fit_ransac(self, sample_coords, residuals_threshold=None):
        """Fit a circle using RANSAC algorithm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Determine default threshold if not provided
            if residuals_threshold is None:
                model = CircleModel()
                success = model.estimate(sample_coords)
                if not success:
                    return None, None, None, None

                residuals = model.residuals(sample_coords)
                q75, q25 = np.percentile(residuals, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                iqr_outliers = residuals[(residuals < lower_bound) | (residuals > upper_bound)]

                if len(iqr_outliers) > 0:
                    residuals_threshold = max(abs(iqr_outliers.min()), abs(iqr_outliers.max())) * 1.1
                else:
                    residuals_threshold = np.median(np.abs(residuals - np.median(residuals))) * 1.4826 * 3

            # RANSAC fitting
            min_inliers = int(sample_coords.shape[0] * 0.9)
            try:
                ransac_model, inliers = ransac(
                    sample_coords, CircleModel, min_inliers, residuals_threshold, max_trials=200
                )
                if ransac_model is None:
                    return None, None, None, None

                xc, yc, r = ransac_model.params

                # Store model for residuals calculation
                self._model = ransac_model
                return r, xc, yc, inliers
            except Exception:
                return None, None, None, None

    def residuals(self, coords2D):
        """
        Calculate residuals for given 2D coordinates using the fitted circle model.

        Parameters:
        - coords2D: ndarray of shape (N, 2) - the points to calculate residuals for.

        Returns:
        - residuals: ndarray of shape (N,) - residuals for each point.
        """
        if self._model is None:
            raise ValueError("No circle model has been fitted yet. Call fit() first.")
        return self._model.residuals(coords2D)


def extract_vertices_from_stl(stl_file_path):
    """
    Extract all vertex data from an STL file including duplicates.
    
    Parameters:
    stl_file_path (str): Path to the STL file
    
    Returns:
    numpy.ndarray: Array of all vertex coordinates with shape (n_vertices, 3)
                   where each row contains [x, y, z] coordinates
    """
    try:
        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(stl_file_path)
        
        # Get all vertices from the mesh (including duplicates)
        all_vertices = stl_mesh.vectors.reshape(-1, 3)
        
        return all_vertices
    
    except FileNotFoundError:
        raise FileNotFoundError(f"STL file not found: {stl_file_path}")
    except Exception as e:
        raise Exception(f"Error reading STL file: {str(e)}")


def slice_mesh_coords(points, slice_height, margin=None):
    """
    Slice a 3D point cloud into horizontal cross-sections.
    
    Parameters:
        points: numpy array of shape (n, 3) with [X, Y, Z] coordinates
        slice_height: height of each slice in mm
        margin: tuple (top_margin, bottom_margin) in mm to discard from top and bottom
    
    Returns:
        List of tuples: (slice_points, slice_z_center)
    """
    if margin is not None:
        if isinstance(margin, (int, float)):
            top_margin, bottom_margin = margin, margin
        else:
            top_margin, bottom_margin = margin[0], margin[1]
        
        z_coords = points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_min_filtered = z_min + bottom_margin
        z_max_filtered = z_max - top_margin
        mask = (z_coords >= z_min_filtered) & (z_coords <= z_max_filtered)
        filtered_points = points[mask]
    else:
        filtered_points = points
    
    if len(filtered_points) == 0:
        return []
    
    z_coords = filtered_points[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    # Calculate number of full slices
    total_height = z_max - z_min
    num_slices = int(total_height / slice_height)
    
    if num_slices == 0:
        return []
    
    # Sort points by z-coordinate for efficient slicing
    sorted_indices = np.argsort(z_coords)
    sorted_points = filtered_points[sorted_indices]
    sorted_z = z_coords[sorted_indices]
    
    sliced_mesh_coords = []
    for i in range(num_slices):
        z_min_slice = z_min + i * slice_height
        z_max_slice = z_min + (i + 1) * slice_height
        
        # Find indices within slice using searchsorted for efficiency
        idx_min = np.searchsorted(sorted_z, z_min_slice, side='left')
        idx_max = np.searchsorted(sorted_z, z_max_slice, side='left')
        
        if idx_max > idx_min:
            slice_points = sorted_points[idx_min:idx_max]
            slice_z_center = z_min_slice + slice_height / 2
            sliced_mesh_coords.append((slice_points, slice_z_center))
    
    return sliced_mesh_coords


def calculate_stats(values):
    """Calculate statistical measures for a set of values."""
    if len(values) == 0:
        return {
            "min": 0, "max": 0, "mean": 0, "median": 0,
            "std": 0, "range": 0
        }
    
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "range": float(np.max(values) - np.min(values))
    }


class AnalyzerGeometricMedianComposite:
    """
    Analyzer implementing Karoly Poka's quality metric for ancient Egyptian stone vessels.
    """
    
    MIN_SLICE_POINTS = 50  # Class variable for minimum points per slice
    
    def __init__(self, source, slice_height=0.03, margin=4.0, ransac=False, min_slice_points=None):
        """
        Initialize the analyzer with 3D point cloud data.
        
        Parameters:
            source: numpy array of shape (n, 3) with [X, Y, Z] coordinates OR path to stl
            slice_height: height of each slice in mm (default: 0.03)
            margin: margin to discard from top and bottom in mm (default: 4.0)
            use_ransac: whether to use RANSAC for circle fitting (default: True)
            min_slice_points: minimum number of points required per slice (default: None uses class variable)
        """

       # Validate source input
        if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith('.stl'):
            self.points = extract_vertices_from_stl(source)
        elif isinstance(source, np.ndarray) and source.ndim == 2 and source.shape[1] == 3:
            self.points = source
        else:
            print("Invalid source. Please provide either a path to an STL file or a numpy array with shape (n, 3) containing [X, Y, Z] coordinates.")
            return

        self.slice_height = slice_height
        self.margin = margin
        self.ransac = ransac
        self.min_slice_points = min_slice_points if min_slice_points is not None else self.MIN_SLICE_POINTS
        self.results = None
        self.detailed_results = None
    
    def analyze(self):
        """Perform the complete analysis."""
        # Slice the mesh
        slices = slice_mesh_coords(self.points, self.slice_height, self.margin)

        if len(slices) == 0:
            self.results = 0.0
            self.detailed_results = {
                "quality_score": 0.0,
                "total_RMSD_median": 0.0,
                "total_C_median": 0.0,
                "num_slices_analyzed": 0,
                "slice_data": []
            }
            return
        
        slice_results = []
        
        # Process each slice
        for slice_points, slice_z_center in slices:
            # Check minimum number of points
            if len(slice_points) < self.min_slice_points:
                continue
                
            # Extract 2D coordinates (X, Y)
            xy_coords = slice_points[:, :2]
            
            circle_fit = CircleFit()
            r, xc, yc, inliers = circle_fit.fit(xy_coords, use_ransac=False)
            residuals = circle_fit.residuals(xy_coords)

            if r is None or xc is None or yc is None:
                continue
                        
            # Calculate rmsd of fit
            rmsd = np.sqrt(np.mean(residuals**2))
            
            # Calculate concentricity
            concentricity = np.sqrt(xc**2 + yc**2)
            
            slice_results.append({
                "z_center": slice_z_center,
                "rmsd": rmsd,
                "concentricity": concentricity,
                "radius": r,
                "center_x": xc,
                "center_y": yc,
                "num_points": len(slice_points)
            })
        
        if len(slice_results) == 0:
            self.results = 0.0
            self.detailed_results = {
                "quality_score": 0.0,
                "total_RMSD_median": 0.0,
                "total_C_median": 0.0,
                "num_slices_analyzed": 0,
                "slice_data": []
            }
            return
        
        # Extract RMSD and concentricity values
        rmsd_values = np.array([s["rmsd"] for s in slice_results])
        concentricity_values = np.array([s["concentricity"] for s in slice_results])
        
        # Calculate medians
        total_RMSD_median = np.median(rmsd_values)
        total_C_median = np.median(concentricity_values)
        
        # Calculate quality score
        quality_score = np.sqrt(total_RMSD_median * total_C_median)
        
        # Store results
        self.results = quality_score
        self.detailed_results = {
            "quality_score": quality_score,
            "total_RMSD_median": total_RMSD_median,
            "total_C_median": total_C_median,
            "num_slices_analyzed": len(slice_results),
            "slice_data": slice_results,
            "rmsd_stats": calculate_stats(rmsd_values),
            "concentricity_stats": calculate_stats(concentricity_values)
        }
    
    def get_result(self):
        """
        Get the quality score.
        
        Returns:
            float: quality score (sqrt(total_RMSD_median * total_C_median))
        """
        if self.results is None:
            self.analyze()
        return self.results
    
    def get_results_detailed(self):
        """
        Get detailed results including statistics.
        
        Returns:
            dict: detailed results with quality score, medians, and statistics
        """
        if self.detailed_results is None:
            self.analyze()
        return self.detailed_results


# Example usage:
# analyzer = AnalyzerGeometricMedianComposite(path_to_STL, slice_height=0.5, margin=2.0, use_ransac=False)
# quality_score = analyzer.get_result()
# detailed_results = analyzer.get_results_detailed()
# print(f"Quality Score: {quality_score:.6f}")
# print(f"RMSD Median: {detailed_results['total_RMSD_median']:.6f}")
# print(f"Concentricity Median: {detailed_results['total_C_median']:.6f}")
# print(f"Number of slices analyzed: {detailed_results['num_slices_analyzed']}")
# print(f"RMSD Stats: {detailed_results['rmsd_stats']}")
# print(f"Concentricity Stats: {detailed_results['concentricity_stats']}")