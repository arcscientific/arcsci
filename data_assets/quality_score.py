"""
Python Implementation of Analytic Framework for Evaluating Precision of Ancient Egyptian Artifacts

This software implements the analytic framework developed by Max Fomitchev-Zamilov (https://maximus.energy/) for
assessing the geometric precision and craftsmanship quality of ancient Egyptian stone vessels
through 3D scanning analysis. The framework evaluates fabrication quality by analyzing
cross-sectional circularity and concentricity along the vessel's height.

Copyright (c) 2025 Stine Gerdes (arcsci.org)

License: CC BY-NC-SA 4.0

"""

import numpy as np
from stl import mesh
from skimage.measure import CircleModel, ransac
from typing import List, Optional
import os
import warnings

import matplotlib.pyplot as plt

class CircleFit:
    def __init__(self):
        self.r = None
        self.xc = None
        self.yc = None
        self.inliers = None
        self._model = None

    def fit(self, sample_coords, ransac=False, residuals_threshold=None, kasa=True):
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
        if ransac:
            result = self._fit_ransac(sample_coords, residuals_threshold)
        elif kasa:
            result = self._fit_kasa(sample_coords)
        else:
            result = self._fit_standard(sample_coords)
        
        self.ransac = ransac
        self.kasa = kasa

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

    def _fit_kasa(self, points):
        ### Fitting algorithm used by Max Fomitchev-Zamilov

        # Extract x and y coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        # Validate input
        if len(x) != len(y) or len(x) < 3:
            raise ValueError('x and y must have the same length and at least 3 points')
        
        # Form matrix A and vector b
        A = np.column_stack((2*x, 2*y, np.ones_like(x)))
        b = -(x**2 + y**2)
        
        # Solve linear system: A*p = b, where p = [a, b, c]
        p, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Extract parameters
        a = p[0]
        b = p[1]
        c = p[2]
        
        # Compute center and radius
        xc = -a
        yc = -b
        r = np.sqrt(xc**2 + yc**2 - c)
        # rmse = np.sqrt(np.mean((np.sqrt((x - x_c)**2 + (y - y_c)**2) - r)**2))
        
        # Check for invalid radius (e.g., imaginary)
        if not np.isreal(r):
            print('Warning: Invalid circle fit: radius is imaginary')
        
        center = np.array([xc, yc])
        return r, xc, yc, None    

    def residuals(self, coords2D):
        """
        Calculate residuals for given 2D coordinates using the fitted circle model.

        Parameters:
        - coords2D: ndarray of shape (N, 2) - the points to calculate residuals for.

        Returns:
        - residuals: ndarray of shape (N,) - residuals for each point.
        """
        if self._model is None and not self.kasa:
            raise ValueError("No circle model has been fitted yet. Call fit() first.")

        if self.kasa:
            residuals = np.sqrt((coords2D[:,0] - self.xc)**2 + (coords2D[:,1] - self.yc)**2) - self.r
        else:
            residuals = self._model.residuals(coords2D)

        return residuals

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


def slice_mesh_coords(points, z_points, slice_height):
    """
    Slice a 3D point cloud into horizontal cross-sections according to Max's method.
    
    Parameters:
        points: numpy array of shape (n, 3) with [X, Y, Z] coordinates
        z_points: array of z-coordinates for the center of each slice
        slice_height: height of each slice in mm
    
    Returns:
        List of tuples: (slice_points, slice_z_center)
    """
    z_coords = points[:, 2]
    
    # Sort points by z-coordinate for efficient slicing
    sorted_indices = np.argsort(z_coords)
    sorted_points = points[sorted_indices]
    sorted_z = z_coords[sorted_indices]
    
    sliced_mesh_coords = []
    
    for z_center in z_points:
        z_min_slice = z_center - slice_height / 2
        z_max_slice = z_center + slice_height / 2
        
        # Find indices within slice using searchsorted for efficient slicing
        idx_min = np.searchsorted(sorted_z, z_min_slice, side='left')
        idx_max = np.searchsorted(sorted_z, z_max_slice, side='left')
        
        if idx_max > idx_min:
            slice_points = sorted_points[idx_min:idx_max]
            sliced_mesh_coords.append((slice_points[:,:2], z_center))
    
    return sliced_mesh_coords


class STLSliceExtractor:
    """
    A class to extract interpolated slices from an STL mesh at given z-heights
    using the edge intersection method.
    """
    
    def __init__(self, stl_file_path):
        """
        Initialize the STLSliceExtractor with an STL file.
        
        Parameters:
        stl_file_path (str): Path to the STL file
        """
        self.stl_file_path = stl_file_path
        self.stl_mesh = None
        self.min_bound = None
        self.max_bound = None
        self._load_mesh()
    
    def _load_mesh(self):
        """Load the STL mesh and compute bounds."""
        try:
            self.stl_mesh = mesh.Mesh.from_file(self.stl_file_path)
            self.min_bound = self.stl_mesh.min_
            self.max_bound = self.stl_mesh.max_
            print(f"Mesh loaded successfully. Bounds: X[{self.min_bound[0]:.3f}, {self.max_bound[0]:.3f}], "
                  f"Y[{self.min_bound[1]:.3f}, {self.max_bound[1]:.3f}], "
                  f"Z[{self.min_bound[2]:.3f}, {self.max_bound[2]:.3f}]")
        except Exception as e:
            raise FileNotFoundError(f"Could not load STL file '{self.stl_file_path}': {str(e)}")
    
    def extract_slices(self, z_heights):
        """
        Extract slices from the mesh at given z-heights.
        Slice points list of 2D arrays representing the slice contours at each z-height
        
        Parameters:
        z_heights (np.ndarray): Array of z-heights at which to extract slices
        
        Returns:
        List of tuples: (slice_points, slice_z_center)
        """
        if self.stl_mesh is None:
            raise ValueError("Mesh not loaded. Please check the STL file path.")
        
        # Validate z_heights are within mesh bounds
        valid_z_heights = self._validate_z_heights(z_heights)
        
        # Extract slices
        sliced_mesh_coords = []
        for z_height in valid_z_heights:
            slice_contour = self._extract_single_slice(z_height)
            sliced_mesh_coords.append((slice_contour, z_height))
        
        return sliced_mesh_coords
    
    def _validate_z_heights(self, z_heights):
        """
        Validate that z-heights are within mesh bounds.
        
        Parameters:
        z_heights (np.ndarray): Array of z-heights to validate
        
        Returns:
        np.ndarray: Validated z-heights within mesh bounds
        """
        if len(z_heights) == 1:
            return z_heights

        min_z, max_z = self.min_bound[2], self.max_bound[2]
        valid_mask = (z_heights >= min_z) & (z_heights <= max_z)
        valid_z_heights = z_heights[valid_mask]

        if len(valid_z_heights) != len(z_heights):
            invalid_count = len(z_heights) - len(valid_z_heights)
            print(f"Warning: {invalid_count} z-height(s) outside mesh bounds [{min_z:.3f}, {max_z:.3f}] ignored")
        
        return valid_z_heights
    
    def _extract_single_slice(self, z_height):
        """
        Extract a single slice from the mesh at a given z-height.
        
        Parameters:
        z_height (float): The z-height at which to extract the slice
        
        Returns:
        np.ndarray: Array of [x,y] points representing the slice contour
        """
        intersection_points = []
        
        # Check each triangle for intersection with the z-plane
        for triangle in self.stl_mesh.points:
            # Get the three vertices of the triangle
            v0, v1, v2 = triangle[0:3], triangle[3:6], triangle[6:9]
            
            # Find intersections between triangle edges and the z-plane
            points = self._find_triangle_z_intersections(v0, v1, v2, z_height)
            intersection_points.extend(points)
        
        # Convert to numpy array
        if len(intersection_points) > 0:
            return np.array(intersection_points)
        else:
            return np.empty((0, 2))  # Return empty array if no intersections
    
    def _find_triangle_z_intersections(self, v0, v1, v2, z_height):
        """
        Find intersection points between a triangle and a z-plane.
        
        Parameters:
        v0, v1, v2 (np.ndarray): The three vertices of the triangle
        z_height (float): The z-height of the slicing plane
        
        Returns:
        List[List[float]]: List of [x,y] intersection points
        """
        vertices = [v0, v1, v2]
        intersections = []
        
        # Check each edge of the triangle
        for i in range(3):
            v_start = vertices[i]
            v_end = vertices[(i + 1) % 3]
            
            # Check if the edge crosses the z-plane
            if (v_start[2] <= z_height <= v_end[2]) or (v_end[2] <= z_height <= v_start[2]):
                # Avoid division by zero
                if abs(v_end[2] - v_start[2]) > 1e-10:
                    # Linear interpolation to find intersection point
                    t = (z_height - v_start[2]) / (v_end[2] - v_start[2])
                    x = v_start[0] + t * (v_end[0] - v_start[0])
                    y = v_start[1] + t * (v_end[1] - v_start[1])
                    intersections.append([x, y])
        
        return intersections
    
    def get_slice_info(self, z_heights):
        """
        Get information about the slices without computing the full intersection points.
        
        Parameters:
        z_heights (np.ndarray): Array of z-heights
        
        Returns:
        dict: Dictionary containing slice information
        """
        valid_z_heights = self._validate_z_heights(z_heights)
        
        info = {
            'total_triangles': len(self.stl_mesh.points),
            'mesh_bounds': {
                'x': [self.min_bound[0], self.max_bound[0]],
                'y': [self.min_bound[1], self.max_bound[1]],
                'z': [self.min_bound[2], self.max_bound[2]]
            },
            'requested_z_heights': len(z_heights),
            'valid_z_heights': len(valid_z_heights),
            'z_heights_range': [float(valid_z_heights.min()) if len(valid_z_heights) > 0 else None,
                              float(valid_z_heights.max()) if len(valid_z_heights) > 0 else None]
        }
        
        return info
    
    def plot_slice(self, z_height, ax=None, show=True):
        """
        Plot a single slice.
        
        Parameters:
        z_height (float): The z-height to plot
        ax (plt.Axes, optional): Matplotlib axes to plot on
        
        Returns:
        plt.Axes: The matplotlib axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        slice_data = self._extract_single_slice(z_height)
        
        if len(slice_data) > 0:
            ax.scatter(slice_data[:, 0], slice_data[:, 1], s=1, alpha=0.7)
            ax.set_title(f'Slice at z={z_height:.3f}')
        else:
            ax.set_title(f'Slice at z={z_height:.3f} (No intersections)')
        
        def metric2imperial(x):
            return x / 25.4
        def imperial2metric(x):
            return x * 25.4

        sec_x = ax.secondary_xaxis("top", functions=(metric2imperial, imperial2metric))
        sec_y = ax.secondary_yaxis("right", functions=(metric2imperial, imperial2metric))

        ticklabelpad = 10
        ax.annotate("mm", xy=(0,0), xytext=(-ticklabelpad, -ticklabelpad), ha="right", va="top",
            xycoords="axes fraction", textcoords="offset points", weight="bold")
        ax.annotate("in", xy=(1,1), xytext=(ticklabelpad, ticklabelpad), ha="left", va="bottom",
            xycoords="axes fraction", textcoords="offset points", weight="bold")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        if show:
            plt.show()

        return ax


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


class Analyzer:
    """
    Analyzer implementing Max Fomitchev-Zamilov's quality metric for ancient Egyptian stone vessels.
    """
    
    # Class variables for default parameters
    NUM_SLICES = 100
    SLICE_HEIGHT = 0.2  # 0.2 mm, 0.00787 inch
    MARGIN = 2.54  # 2.54mm, 0.1 inch
    MIN_SLICE_POINTS = 20  # Minimum points required per slice
    CLEANUP_STEPS = 1  # Number of cleanup iterations
    MIN_ANGLE_COVERAGE = 3 * np.pi / 4  # 135 degrees - maximum allowed gap
    
    def __init__(self, source, num_slices=None, slice_height=None, margin=None, min_slice_points=None, 
                 ransac=False, cleanup_steps=None, apply_obliquity_correction=False, vertice_slicer=True):
        """
        Initialize the analyzer with 3D point cloud data.

        If source is path to an stl file - Mesh will be sliced using interpolation
        If source is numpy array of shape (n, 3) coords - Mesh will be sliced with defined slice height
        
        Parameters:
            source: numpy array of shape (n, 3) with [X, Y, Z] coordinates OR path to stl
            num_slices: number of slices to use (default: 100)
            slice_height: height of each slice in mm (default: 0.2)
            margin: margin to discard from top and bottom in mm (default: 2.54)
            min_slice_points: minimum number of points required per slice (default: 20)
            cleanup_steps: number of cleanup iterations (default: 1)
            apply_obliquity_correction: whether to apply obliquity correction (default: True)
        """

        self.points = None
        self.stl_path = None
       # Validate source input
        if isinstance(source, str) and os.path.isfile(source) and source.lower().endswith('.stl'):
            self.stl_path = source
        elif isinstance(source, np.ndarray) and source.ndim == 2 and source.shape[1] == 3:
            self.points = source
        else:
            print("Invalid source. Please provide either a path to an STL file or a numpy array with shape (n, 3) containing [X, Y, Z] coordinates.")
            return
        self.num_slices = num_slices if num_slices is not None else self.NUM_SLICES
        self.slice_height = slice_height if slice_height is not None else self.SLICE_HEIGHT
        self.margin = margin if margin is not None else self.MARGIN
        self.min_slice_points = min_slice_points if min_slice_points is not None else self.MIN_SLICE_POINTS
        self.cleanup_steps = cleanup_steps if cleanup_steps is not None else self.CLEANUP_STEPS
        self.apply_obliquity_correction = apply_obliquity_correction
        self.ransac = ransac
        self.results = None
        self.detailed_results = None
    

    def slicing_data_compare(self, sample_slice=50):
        """Compare the circle fit centers of different slicing methods"""

        if self.stl_path is None:
            print("STL path missing")
            return

        coords = extract_vertices_from_stl(self.stl_path)

        trimmed_coords, z_min_trimmed, z_max_trimmed = self._trim_margins(coords)

        z_points = np.linspace(z_min_trimmed, z_max_trimmed, self.num_slices)

        slices_vertices = slice_mesh_coords(trimmed_coords, z_points, self.slice_height)
        # slices_mesh1 = slice_mesh_coords(trimmed_coords, z_points, self.slice_height)
        slices_mesh1 = STLSliceExtractor(self.stl_path).extract_slices(z_points)

        # Fit circle on vertices slice 50
        circle_fit = CircleFit()
        r_v, xc_v, yc_v, inliers = circle_fit.fit(slices_vertices[sample_slice][0], ransac=False, kasa=True)

        # Fit circle on mesn1 slice 50
        circle_fit = CircleFit()
        r_m1, xc_m1, yc_m1, inliers = circle_fit.fit(slices_mesh1[sample_slice][0], ransac=False, kasa=True)

        # Process slices
        slice_results_v, rmsd_values_v, centers_v, z_values_v = self._analyze_slices(slices_mesh1)
        slice_results_m, rmsd_values_m, centers_m, z_values_m = self._analyze_slices(slices_vertices)
                
        # Apply cleanup process
        rmsd_values_v, centers_v, z_values_v, slice_results_v = self._clean_up(
            rmsd_values_v, centers_v, z_values_v, slice_results_v, self.cleanup_steps
        )
        rmsd_values_m, centers_m, z_values_m, slice_results_m = self._clean_up(
            rmsd_values_m, centers_m, z_values_m, slice_results_m, self.cleanup_steps
        )
        
        # Calculate metric statistics
        dist_mean_center_v = self._dist_mean_center(centers_v)
        dist_mean_center_m = self._dist_mean_center(centers_m)
        mean_rmsd_v, mean_dist_mean_center_v, std_dist_mean_center_v = self._metric_stats(rmsd_values_v, dist_mean_center_v)
        mean_rmsd_m, mean_dist_mean_center_m, std_dist_mean_center_m = self._metric_stats(rmsd_values_m, dist_mean_center_m)

        concentricity_v = [np.sqrt(xc**2 + yc**2) for xc,yc in centers_v]
        concentricity_m = [np.sqrt(xc**2 + yc**2) for xc,yc in centers_m]

        # Print comparison
        print(f"Slice stats, sample slice {sample_slice}:")
        print(f"Number of points, vertices: {len(slices_vertices[sample_slice][0])}")
        print(f"Number of points, mesh 1:   {len(slices_mesh1[sample_slice][0])}")

        print(f"\nCircle fit center, sample slice {sample_slice}")
        print(f"Fit center, vertices: {xc_v:.3f}, {xc_v:.3f}")
        print(f"Fit center, mesh 1:   {xc_m1:.3f}, {xc_m1:.3f}")

        print(f"\nCenter line stats, mean center location:")
        print(f"Vertices slicing:")
        print(f"\tMean C: {mean_dist_mean_center_v:.4f}")
        print(f"\tStd  C: {std_dist_mean_center_v:.4f}")
        print(f"Mesh interpolation:")
        print(f"\tMean C: {mean_dist_mean_center_m:.4f}")
        print(f"\tStd  C: {std_dist_mean_center_m:.4f}")

        print(f"\nCenter line stats, z-axis:")
        print(f"Vertices slicing:")
        print(f"\tMean C: {np.mean(concentricity_v):.4f}")
        print(f"\tStd  C: {np.std(concentricity_v):.4f}")
        print(f"Mesh interpolation:")
        print(f"\tMean C: {np.mean(concentricity_m):.4f}")
        print(f"\tStd  C: {np.std(concentricity_m):.4f}")

    def analyze(self):
        """Perform the complete analysis according to Max's method."""

        # Step 1: Remove top and bottom margin
        if not self.stl_path is None:
            coords = extract_vertices_from_stl(self.stl_path)
        elif not self.points is None:
            coords = self.points

        trimmed_coords, z_min_trimmed, z_max_trimmed = self._trim_margins(coords)

        if len(trimmed_coords) == 0:
            self._set_zero_results()
            return
        
        # Check if we have enough height for the required number of slices
        trimmed_height = z_max_trimmed - z_min_trimmed
        if trimmed_height < self.slice_height:
            self._set_zero_results()
            return
        
        # Step 2: Generate exactly num_slices evenly distributed z-points
        #         and slice the mesh
        z_points = np.linspace(z_min_trimmed, z_max_trimmed, self.num_slices)
        
        # Slice mesh using interpolation if path is given and vertice_slicer is False
        if not self.stl_path is None and not vertice_slicer:
            slices = STLSliceExtractor(self.stl_path).extract_slices(z_points)
        # Else use the vertices slicer
        elif not self.points is None:
            slices = slice_mesh_coords(trimmed_coords, z_points, self.slice_height)

        if len(slices) == 0:
            self._set_zero_results()
            return

        # Step 4: Process each slice
        slice_results, rmsd_values, centers, z_values, radii = self._analyze_slices(slices)
        
        if len(slice_results) == 0:
            self._set_zero_results()
            return
                
        # Apply obliquity correction BEFORE cleanup (as in MATLAB)
        # Default apply_obliquity_correction = False - May distort data
        if self.apply_obliquity_correction:
            rmsd_values, dist_mean_center = self._apply_obliquity_correction(rmsd_values, np.zeros_like(rmsd_values), z_values, radii)
        
        # Apply cleanup process
        valid_rmsd_values, valid_centers, valid_z_values, valid_slice_results, valid_radii = self._clean_up(
            rmsd_values, centers, z_values, slice_results, radii, self.cleanup_steps
        )
        
        # Apply obliquity correction AFTER cleanup
        # Default apply_obliquity_correction = False - May distort data
        dist_mean_center = self._dist_mean_center(valid_centers)
        if self.apply_obliquity_correction and len(valid_rmsd_values) > 0:
            valid_rmsd_values, dist_mean_center = self._apply_obliquity_correction(valid_rmsd_values, dist_mean_center, valid_z_values, valid_radii)
        
        # Calculate metric statistics
        mean_rmsd, mean_dist_mean_center, std_dist_mean_center = self._metric_stats(valid_rmsd_values, dist_mean_center)
        
        # Calculate alternative concentricity stats for comparison
        concentricity_values = np.array([slice_result["concentricity"] for slice_result in valid_slice_results])
        mean_concentricity = np.mean(concentricity_values)
        std_concentricity = np.std(concentricity_values)
        
        # Step 5: Calculate composite quality scores
        
        # Z-axis concentricity version: Q = ⟨RMDS⟩ + ⟨ΔC⟩ + σ(ΔC)
        # where ΔC = distance from centers to Z-axis (origin)
        quality_score_zaxis_metric = mean_rmsd + mean_concentricity + std_concentricity
        
        # Actual implementation version: Q = RMSD + ⟨dist_mean_center⟩ + σ(dist_mean_center)
        quality_score_metric = mean_rmsd + mean_dist_mean_center + std_dist_mean_center
        
        # Convert to micrometers/thou (1 mm = 1000 micrometers, 1 mm = 39.3701 thou)
        quality_score_zaxis_imperial = quality_score_zaxis_metric * 39.3701  # Convert to thou
        quality_score_zaxis_metric = quality_score_zaxis_metric * 1000  # Convert to micrometers
        
        quality_score_imperial = quality_score_metric * 39.3701  # Convert to thou
        quality_score_metric = quality_score_metric * 1000  # Convert to micrometers
        
        # Convert individual metrics to imperial
        mean_rmsd_imperial = mean_rmsd * 39.3701
        mean_concentricity_imperial = mean_concentricity * 39.3701
        std_concentricity_imperial = std_concentricity * 39.3701
        mean_dist_mean_center_imperial = mean_dist_mean_center * 39.3701
        std_dist_mean_center_imperial = std_dist_mean_center * 39.3701
        
        # Store results
        self.results = {
            "quality_score_metric": quality_score_metric,  # micrometers
            "quality_score_imperial": quality_score_imperial,  # thou
        }
        self.detailed_results = {
            # Actual implementation metrics
            "quality_score_metric": quality_score_metric,  # micrometers
            "quality_score_imperial": quality_score_imperial,  # thou
            "mean_rmsd_metric": mean_rmsd,
            "mean_rmsd_imperial": mean_rmsd_imperial,
            "mean_dist_mean_center_metric": mean_dist_mean_center,  # Mean distance from centers to mean center
            "mean_dist_mean_center_imperial": mean_dist_mean_center_imperial,
            "std_dist_mean_center_metric": std_dist_mean_center,  # Std of distance from centers to mean center
            "std_dist_mean_center_imperial": std_dist_mean_center_imperial,

            # Zaxis version metrics
            "quality_score_zaxis_metric": quality_score_zaxis_metric,  # micrometers
            "quality_score_zaxis_imperial": quality_score_zaxis_imperial,  # thou
            "mean_concentricity_metric": mean_concentricity,
            "mean_concentricity_imperial": mean_concentricity_imperial,
            "std_concentricity_metric": std_concentricity,
            "std_concentricity_imperial": std_concentricity_imperial,
                        
            # Configuration
            "obliquity_correction_applied": self.apply_obliquity_correction,
            
            # Common data
            "num_slices_analyzed": len(valid_slice_results),
            "total_slices": self.num_slices,
            "slice_data": valid_slice_results,
            "rmsd_stats": calculate_stats(rmsd_values),
            "dist_mean_center_stats": calculate_stats(dist_mean_center),  # Distance from centers to mean center
            "concentricity_stats": calculate_stats(concentricity_values)
        }


    def _trim_margins(self,coords3D):
        z_coords = coords3D[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_min_trimmed = z_min + self.margin
        z_max_trimmed = z_max - self.margin
        
        # Filter points within the trimmed range
        mask = (z_coords >= z_min_trimmed) & (z_coords <= z_max_trimmed)
        return coords3D[mask], z_min_trimmed, z_max_trimmed

    def _analyze_slices(self,slices):
        slice_results = []  # Combined stats for each slice
        rmsd_values = []    # Store for use in precision metric
        centers = []        # Store [xc, yc] for each slice for use in precision metric
        z_values = []       # Store z_center for each slice
        radii = []          # Radius for each slice
        
        for slice_points, slice_z_center in slices:
            # Check minimum number of points
            if len(slice_points) < self.min_slice_points:
                continue
                
            # Extract 2D coordinates (X, Y)
            xy_coords = slice_points[:, :2]

            # Fit circle
            circle_fit = CircleFit()
            r, xc, yc, inliers = circle_fit.fit(xy_coords, ransac=False, kasa=False)

            if r is None or xc is None or yc is None:
                continue
            
            # Calculate angular coverage to detect holes
            gap_coverage = calculate_gap_coverage(xy_coords, [xc, yc])
            
            # Exclude slices with holes larger than 3π/4 (135 degrees)
            if gap_coverage >= self.MIN_ANGLE_COVERAGE:
                continue
            
            # Calculate residuals            
            residuals = circle_fit.residuals(xy_coords)
            
            # Calculate RMSD of fit
            rmsd = np.sqrt(np.mean(residuals**2))
            
            # Calculate concentricity (distance from circle center to Z-axis - website version)
            concentricity = np.sqrt(xc**2 + yc**2)
            
            # Store results for this slice
            slice_result = {
                "z_center": slice_z_center,
                "rmsd": rmsd,                    # RMSD of circle fit
                "concentricity": concentricity,  # In relation to z-axis
                "gap_coverage": gap_coverage,    # Angular gap coverage
                "radius": r,
                "center_x": xc,
                "center_y": yc,
                "num_points": len(slice_points)
            }
            
            # Store separate list for easy access
            slice_results.append(slice_result)
            rmsd_values.append(rmsd)
            centers.append([xc, yc])
            z_values.append(slice_z_center)
            radii.append(r)

        # Convert to numpy arrays for easier processing
        rmsd_values = np.array(rmsd_values)
        centers = np.array(centers)
        z_values = np.array(z_values)
        radii = np.array(radii)

        return slice_results, rmsd_values, centers, z_values, radii


    def _clean_up(self, rmsd_values, centers, z_values, slice_results, radii, steps=1):
        """Clean up outliers slices.
           dist_mean_center -〈dist_mean_center〉> 2〈dist_mean_center〉 or  RMSD –〈RMSD〉> 2〈RMSD〉
           (equivalent to ΔC > 3⟨ΔC⟩ or RMSD > 3⟨RMSD⟩)
        """
        RMS_MULTIPLIER = 2
        AVGDR_MULTIPLIER = 2
        
        # Remove outliers
        for j in range(steps):
            # Calculate statistics
            dist_mean_center = self._dist_mean_center(centers)
            mean_rmsd, mean_dist_mean_center, std_dist_mean_center = self._metric_stats(rmsd_values, dist_mean_center)

            # Remove outliers
            range_mask = (dist_mean_center - mean_dist_mean_center < mean_dist_mean_center * AVGDR_MULTIPLIER) & (rmsd_values - mean_rmsd < mean_rmsd * RMS_MULTIPLIER)
            rmsd_values = rmsd_values[range_mask]
            centers = centers[range_mask]
            z_values = z_values[range_mask]
            radii = radii[range_mask]
            slice_results = [slice_results[i] for i in range(len(slice_results)) if range_mask[i]]
        
        return rmsd_values, centers, z_values, slice_results, radii

    def _metric_stats(self, rmsd_values, dist_mean_center):
        # Calculate stats
        mean_rmsd = np.mean(rmsd_values)
        mean_dist_mean_center = np.mean(dist_mean_center)
        std_dist_mean_center = np.std(dist_mean_center, ddof=1)
        
        return mean_rmsd, mean_dist_mean_center, std_dist_mean_center

    def _dist_mean_center(self, centers):
        """Calculate distance from each center to the mean center of all centers"""
        if len(centers) == 0:
            return np.array([])
        
        # Calculate mean center once for this dataset
        mean_center = np.mean(centers, axis=0)
        
        # Calculate Euclidean distance from each center to mean center
        distances = np.sqrt(np.sum((centers - mean_center)**2, axis=1))
        
        return distances

    def sort_points_by_angle(points, center):
        """Sort points by angle around a center point, as per MATLAP implementation"""
        # Translate to origin
        x_shifted = points[:, 0] - center[0]
        y_shifted = points[:, 1] - center[1]
        # Compute angle from centroid
        theta = np.arctan2(y_shifted, x_shifted)
        # Sort angles
        sorted_indices = np.argsort(theta)
        return theta[sorted_indices]

    def calculate_gap_coverage(points, center):
        """Calculate the gap coverage in angular distribution of points, as per MATLAP implementation"""
        if len(points) < 3:
            return 2 * np.pi  # Large gap if too few points
        
        # Sort points by angle
        sorted_angles = sort_points_by_angle(points, center)
        
        # Calculate angular differences
        angle_diffs = np.diff(sorted_angles)
        # Handle wrap-around
        angle_diffs = np.append(angle_diffs, 2 * np.pi + sorted_angles[0] - sorted_angles[-1])
        
        # Find gaps larger than pi/12 (15 degrees)
        large_gaps = angle_diffs[angle_diffs > np.pi / 12]
        
        # Sum up the large gaps
        gap_sum = np.sum(large_gaps)
        
        return gap_sum

    def _apply_obliquity_correction(self, rmsd_values, dr_values, z_values, radii):
        """Apply obliquity correction, as per MATLAB implementation"""
        if len(rmsd_values) < 2:
            return rmsd_values, dr_values
        
        # Calculate differences between consecutive slices
        dz = np.diff(z_values)
        dR = np.diff(radii)  # Use actual radius differences
        
        # Calculate weights (cosine of angle)
        slope_length = np.sqrt(dR**2 + dz**2)
        # Avoid division by zero
        weights = np.abs(dz) / np.where(slope_length > 1e-10, slope_length, 1e-10)
        # Add 1 for the last slice (no next slice to compare)
        weights = np.append(weights, 1.0)
        
        # Apply correction
        corrected_rmsd = rmsd_values * weights
        corrected_dr = dr_values * weights
        return corrected_rmsd, corrected_dr

    def _set_zero_results(self):
        """Set results to zero when no valid data is available."""
        self.results = 0.0
        self.detailed_results = {
            # Actual implementation metrics
            "quality_score_metric": 0.0,
            "quality_score_imperial": 0.0,
            "mean_rmsd_metric": 0.0,
            "mean_rmsd_imperial": 0.0,
            "mean_dist_mean_center_metric": 0.0,
            "mean_dist_mean_center_imperial": 0.0,
            "std_dist_mean_center_metric": 0.0,
            "std_dist_mean_center_imperial": 0.0,

            # Zaxis version metrics
            "quality_score_zaxis_metric": 0.0,
            "quality_score_zaxis_imperial": 0.0,
            "mean_concentricity_metric": 0.0,
            "mean_concentricity_imperial": 0.0,
            "std_concentricity_metric": 0.0,
            "std_concentricity_imperial": 0.0,
                        
            # Configuration
            "obliquity_correction_applied": self.apply_obliquity_correction,
            
            # Common data
            "num_slices_analyzed": 0.0,
            "total_slices": self.num_slices,
            "slice_data": [],
            "rmsd_stats": calculate_stats([]),
            "dist_mean_center_stats": calculate_stats([]),
            "concentricity_stats": calculate_stats([]),
        }
    
    def get_result(self):
        """
        Get the quality score in micrometers (actual implementation version).
        
        Returns:
            float: quality score in micrometers
        """
        if self.results is None:
            self.analyze()
        return self.results
    
    def get_results_detailed(self):
        """
        Get detailed results including both website and actual implementation metrics.
        
        Returns:
            dict: detailed results with quality scores and statistics
        """
        if self.detailed_results is None:
            self.analyze()
        return self.detailed_results

# Example usage:
# analyzer = AnalyzerMaxFomitchevZamilov(coords3D, apply_obliquity_correction=True)
# quality_score = analyzer.get_result()
# detailed_results = analyzer.get_results_detailed()