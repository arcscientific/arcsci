"""
Statistical Analysis of Mathematical Constants in Ancient Egyptian Vase Design
This script analyzes the statistical significance of mathematical constants (π, φ)
found in vase measurements from ancient Egyptian artifacts. It uses a null hypothesis
approach to determine the likelihood of such ratios occurring by chance.

Copyright (c) 2025 Stine Gerdes (arcsci.org)
License: CC BY-NC-SA 4.0
"""
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime
from math import pi, sqrt
warnings.filterwarnings('ignore')

class ConstrainedVaseGenerator:
    """
    Generates random vases based on empirically derived constraints from 
    3D scanned ancient Egyptian vessels. This ensures generated vessels
    maintain realistic proportions and relationships found in actual artifacts.
    """
    
    def __init__(self):
        """
        Initialize with measurement ranges and ratios derived from 
        30 scanned ancient Egyptian vessels.
        """
        # Opening diameter range (H)
        self.range_H = [22.096601664087043, 48.97771948361376]
        
        # Ratio ranges from empirical data
        self.ratioAH = [1.271556771858752, 1.7264139008998285]  # Lip / Opening
        self.ratioBH = [1.1145195868924607, 1.4632860743750264]  # Neck / Opening
        self.percentageHB = [0.3887682183546316, 0.6377714878544333]  # Neck position percentage
        self.ratioCH = [1.9872286586411911, 2.5149432955233286]  # Body / Opening
        self.ratioGC = [1.034554715705815, 1.1438441305440088]  # Handles / Body
        self.ratioBD = [1.05, 1.2]  # Neck / Foot min
        self.ratioED = [1.01, 1.06]  # Foot min / Foot max
    
    def generate_constrained_vase(self):
        """
        Generate a single vase with realistic proportions based on 
        archaeological constraints. This method ensures logical relationships
        between measurements are maintained.
        
        Returns:
            dict: Dictionary containing all vase measurements
        """
        # Start with opening diameter (base measurement)
        opening = random.uniform(*self.range_H)
        
        # Generate lip diameter (typically larger than opening)
        lip = random.uniform(opening * self.ratioAH[0], opening * self.ratioAH[1])
        
        # Generate neck diameter (between lip and opening)
        neck = random.uniform(
            (lip - opening) * self.percentageHB[0] + opening,
            (lip - opening) * self.percentageHB[1] + opening
        )
        
        # Generate body diameter (largest measurement)
        body = random.uniform(opening * self.ratioCH[0], opening * self.ratioCH[1])
        
        # Generate handles (related to body size)
        handles = random.uniform(body * self.ratioGC[0], body * self.ratioGC[1])
        
        # Generate foot measurements (smallest features)
        foot_min = random.uniform(neck * self.ratioBD[0], neck * self.ratioBD[1])
        foot_max = random.uniform(foot_min * self.ratioED[0], foot_min * self.ratioED[1])
        
        # Return complete measurement set
        measurements = {
            'opening': opening,      # Opening diameter
            'top_max': lip,          # Lip diameter (widest top point)
            'top_min': neck,         # Neck diameter (narrowest top point)  
            'body_max': body,        # Maximum body diameter
            'foot_min': foot_min,    # Minimum foot diameter
            'foot_max': foot_max,    # Maximum foot diameter
            'handles_max': handles   # Handle attachment diameter
        }
        
        return measurements

class StatisticalVaseAnalyzer:
    """
    Comprehensive statistical analyzer for mathematical constants in vase design.
    Uses Monte Carlo simulation to establish null distributions and assess
    statistical significance of observed ratios.
    """
    
    def __init__(self, target_constants=None, tolerance_range=0.002, error_deviation=0.4, 
                 check_n_matches=3, alpha=0.05, random_seed=42):
        """
        Initialize the analyzer with analysis parameters.
        
        Args:
            target_constants (list): Mathematical constants to test for
            tolerance_range (float): Percentage tolerance for ratio matching (0.002 = 0.2%)
            error_deviation (float): Maximum absolute deviation allowed
            check_n_matches (int): Minimum number of matches to look for
            alpha (float): Significance level for statistical tests
            random_seed (int): Seed for reproducible random generation
        """
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize vase generator with archaeological constraints
        self.generator = ConstrainedVaseGenerator()
        
        # Define target mathematical constants
        phi = (1 + sqrt(5)) / 2  # Golden ratio
        self.target_constants = target_constants or [
            {'name': 'pi', 'value': pi, 'symbol': 'π'},
            {'name': 'phi', 'value': phi, 'symbol': 'φ'},
            {'name': 'phi_squared', 'value': phi**2, 'symbol': 'φ²'},
            {'name': 'pi_over_phi_squared', 'value': pi / (phi**2), 'symbol': 'π/φ²'},
            {'name': 'phi_cubed', 'value': phi**3, 'symbol': 'φ³'},
            {'name': 'pi_over_phi', 'value': pi / phi, 'symbol': 'π/φ'}
        ]
        
        # Analysis parameters
        self.tolerance_range = tolerance_range
        self.error_deviation = error_deviation
        self.check_n_matches = check_n_matches
        self.alpha = alpha
        
        # Store generated null distribution
        self.null_distribution_data = None
    
    def generate_all_measurement_variants(self, measurements_dict):
        """
        Generate both diameter and radius variants for all measurements.
        This allows testing of ratios in both d/d, r/r, and d/r formats.
        
        Args:
            measurements_dict (dict): Original diameter measurements
            
        Returns:
            dict: Dictionary with both diameter (_d) and radius (_r) variants
        """
        variants = {}
        
        # For each measurement, create both diameter and radius versions
        for name, diameter_value in measurements_dict.items():
            variants[f"{name}_d"] = diameter_value      # Diameter variant
            variants[f"{name}_r"] = diameter_value / 2.0  # Radius variant
            
        return variants
    
    def generate_all_meaningful_ratios(self, measurements_series):
        """
        Generate all possible meaningful ratios between measurements.
        Excludes self-ratios (x/x) and d/d ratios to avoid duplication (since d/d = r/r).
        Includes: d/r, r/d, and r/r ratios.
        """
        ratios = {}
        measurements = measurements_series.dropna()
        measurement_names = list(measurements.index)
        
        # Generate all pairwise ratios (excluding self-ratios and d/d ratios)
        for i, name1 in enumerate(measurement_names):
            for j, name2 in enumerate(measurement_names):
                if i != j and name1 != name2:  # Don't compare measurement with itself
                    if measurements[name2] != 0:  # Avoid division by zero
                        # Check if first measurement is diameter and second is diameter (d/d)
                        is_name1_d = '_d' in name1
                        is_name2_d = '_d' in name2
                        
                        # Skip d/d ratios to avoid duplication (d/d = r/r)
                        if is_name1_d and is_name2_d:
                            continue
                        
                        ratio_value = measurements[name1] / measurements[name2]
                        ratio_name = f"{name1}/{name2}"
                        ratios[ratio_name] = ratio_value
                        
        return ratios
    
    def is_diameter_radius_ratio(self, ratio_name):
        """
        Check if a ratio is specifically diameter/radius (d/r format).
        
        Args:
            ratio_name (str): Name of the ratio (e.g., "opening_d/body_r")
            
        Returns:
            bool: True if ratio is diameter/radius format
        """
        parts = ratio_name.split('/')
        if len(parts) == 2:
            nominator, denominator = parts
            return '_d' in nominator and '_r' in denominator
        return False
    
    def count_constant_matches(self, ratios_dict, specific_type=None):
        """
        Count how many ratios match target mathematical constants within tolerance and deviation.
        
        Args:
            ratios_dict (dict): Dictionary of ratios to test
            specific_type (str): Filter for specific ratio types ('d_over_r' or None)
            
        Returns:
            list: List of matching ratios with detailed information
        """
        matches = []
        
        # Test each ratio against all target constants
        for ratio_name, ratio_value in ratios_dict.items():
            # Apply type filtering if specified
            if specific_type == 'd_over_r' and not self.is_diameter_radius_ratio(ratio_name):
                continue
                
            # Test against each mathematical constant
            for constant_info in self.target_constants:
                constant_value = constant_info['value']
                
                # Calculate tolerance bounds
                lower_bound = constant_value * (1 - self.tolerance_range)
                upper_bound = constant_value * (1 + self.tolerance_range)
                
                # Check if ratio falls within tolerance
                if lower_bound <= ratio_value <= upper_bound:
                    # Calculate error metrics
                    error_percentage = abs((ratio_value - constant_value) / constant_value) * 100
                    absolute_deviation = abs(ratio_value - constant_value)
                    
                    # Check if absolute deviation is within allowed limit
                    if absolute_deviation <= self.error_deviation:
                        # Store match information
                        matches.append({
                            'ratio_name': ratio_name,
                            'ratio_value': ratio_value,
                            'constant_name': constant_info['name'],
                            'constant_symbol': constant_info['symbol'],
                            'constant_value': constant_value,
                            'error_percentage': error_percentage,
                            'absolute_deviation': absolute_deviation,
                            'is_d_over_r': self.is_diameter_radius_ratio(ratio_name)
                        })
                    
        return matches
    
    def generate_null_distribution(self, n_samples=100000):
        """
        Generate a null distribution by creating random vessels and counting matches.
        This represents what we would expect to find by chance alone.
        
        Args:
            n_samples (int): Number of random vessels to generate
            
        Returns:
            list: List of dictionaries containing match data for each vessel
        """
        print(f"Generating null distribution with {n_samples:,} random vessels...")
        null_data = []
        
        # Generate random vessels and store match data
        for i in range(n_samples):
            # Progress indicator for large samples
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} / {n_samples:,} vessels")
            
            # Generate random vase with archaeological constraints
            measurements = self.generator.generate_constrained_vase()
            
            # Create measurement variants (d and r)
            variants = self.generate_all_measurement_variants(measurements)
            variants_series = pd.Series(variants)
            
            # Generate all possible ratios
            ratios = self.generate_all_meaningful_ratios(variants_series)
            
            # Count matches for different categories
            all_matches = self.count_constant_matches(ratios)
            phi_pi_phi2_matches = [m for m in all_matches 
                                  if m['constant_name'] in ['pi', 'phi', 'phi_squared']]
            d_over_r_phi_pi_phi2_matches = [m for m in phi_pi_phi2_matches if m['is_d_over_r']]
            
            # Store vessel data
            vessel_data = {
                'all_matches': all_matches,
                'phi_pi_phi2_matches': phi_pi_phi2_matches,
                'd_over_r_phi_pi_phi2_matches': d_over_r_phi_pi_phi2_matches,
                'measurements': measurements
            }
            
            null_data.append(vessel_data)
        
        print("Null distribution generation complete")
        self.null_distribution_data = null_data
        return null_data
    
    def calculate_confidence_interval(self, probability, n_samples, confidence_level=0.95):
        """
        Calculate confidence interval for binomial probability using Wilson score interval.
        """
        # Number of successes
        successes = probability * n_samples
        
        # Calculate Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        n = n_samples
        p = probability
        
        if n == 0:
            return 0.0, 0.0
            
        denominator = 1 + z**2/n
        centre_adjusted_probability = (p + z**2/(2*n)) / denominator
        adjusted_standard_deviation = np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        
        lower_bound = centre_adjusted_probability - z*adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z*adjusted_standard_deviation
        
        return max(0, lower_bound), min(1, upper_bound)
    
    def calculate_bayes_factor(self, probability):
        """
        Calculate Bayes Factor to quantify evidence strength.
        """
        # Likelihood of data under null hypothesis (random chance)
        likelihood_null = probability
        
        # Likelihood of data under alternative hypothesis (intentional design)
        # Using 1 as a conservative estimate for intentional design
        likelihood_alternative = 1.0
        
        # Bayes Factor
        if likelihood_null > 0:
            bayes_factor = likelihood_alternative / likelihood_null
        else:
            bayes_factor = float('inf')
        
        # Interpretation
        evidence_strength = ""
        if bayes_factor > 100:
            evidence_strength = "Decisive evidence"
        elif bayes_factor > 30:
            evidence_strength = "Very strong evidence"
        elif bayes_factor > 10:
            evidence_strength = "Strong evidence"
        elif bayes_factor > 3:
            evidence_strength = "Moderate evidence"
        elif bayes_factor > 1:
            evidence_strength = "Anecdotal evidence"
        else:
            evidence_strength = "No evidence/supports null"
        
        return {
            'bayes_factor': bayes_factor,
            'evidence_strength': evidence_strength
        }
    
    def calculate_effect_sizes(self, target_count, null_distribution):
        """
        Calculate multiple effect size measures.
        """
        null_mean = np.mean(null_distribution) if len(null_distribution) > 0 else 0
        null_std = np.std(null_distribution) if len(null_distribution) > 0 else 0
        
        # Cohen's d
        cohens_d = (target_count - null_mean) / null_std if null_std > 0 else 0
        
        # Hedge's g (bias-corrected Cohen's d)
        n_total = len(null_distribution)
        if n_total > 1:
            correction_factor = 1 - (3 / (4 * (1 + n_total) - 9))
            hedges_g = cohens_d * correction_factor
        else:
            hedges_g = 0
        
        # Odds Ratio
        if len(null_distribution) > 0:
            null_successes = np.sum(np.array(null_distribution) >= target_count)
            null_failures = len(null_distribution) - null_successes
            
            if null_successes > 0:
                odds_ratio = (1 / (0 + 1e-10)) / (null_successes / (null_failures + 1e-10))
            else:
                odds_ratio = float('inf') if target_count > 0 else 0
        else:
            odds_ratio = float('inf') if target_count > 0 else 0
        
        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'odds_ratio': odds_ratio
        }
    
    def analyze_precision_probabilities(self, target_measurements, n_null_samples=1000000):
        """
        Analyze the three specific probability questions:
        a) Probability of finding vessel with N+ matches
        b) Probability with specific precision requirements  
        c) Probability with interlocking properties
        
        Args:
            target_measurements (dict): Target vase measurements
            n_null_samples (int): Number of null samples to generate
            
        Returns:
            dict: Detailed probability analysis results
        """
        print("=== PRECISION PROBABILITY ANALYSIS ===")
        
        # Generate single null distribution if not already done
        if self.null_distribution_data is None or len(self.null_distribution_data) != n_null_samples:
            self.generate_null_distribution(n_null_samples)
        
        null_data = self.null_distribution_data
        
        # Test a): Probability of finding vessel with N+ matches for ALL target constants
        print(f"\nTest a): Probability of finding vessel with {self.check_n_matches}+ matches (all constants)")
        vessels_with_n_plus = 0
        for vessel in null_data:
            if len(vessel['all_matches']) >= self.check_n_matches:
                vessels_with_n_plus += 1
        
        prob_a = vessels_with_n_plus / n_null_samples
        ci_a_lower, ci_a_upper = self.calculate_confidence_interval(prob_a, n_null_samples)
        bayes_a = self.calculate_bayes_factor(prob_a)
        
        print(f"Vessels with {self.check_n_matches}+ matches: {vessels_with_n_plus:,} / {n_null_samples:,}")
        print(f"Probability: {prob_a:.5f}")
        print(f"95% CI: [{ci_a_lower:.5f}, {ci_a_upper:.5f}]")
        print(f"Bayes Factor: {bayes_a['bayes_factor']:.2f} ({bayes_a['evidence_strength']})")
        
        # Test b): Probability with specific precision requirements (π, φ, φ² only, d/r ratios)
        print(f"\nTest b): Probability with precision requirements (π, φ, φ², d/r only)")
        vessels_with_precision = 0
        for vessel in null_data:
            matches = vessel['d_over_r_phi_pi_phi2_matches']  # Only π, φ, φ² AND only d/r ratios
            if len(matches) >= self.check_n_matches:
                # For exactly 3 matches: check if we have 2 ultra-high + 1 high precision
                if self.check_n_matches >= 3:
                    ultra_high_precision = [m for m in matches if m['error_percentage'] < 0.07]
                    remaining_matches = [m for m in matches if not m in ultra_high_precision]
                    high_precision = [m for m in remaining_matches if m['error_percentage'] < 0.2]

                    # Need at least 2 ultra-high (< 0.07%) and 1 high or ultra-high (0.07% - 0.2%)
                    if len(ultra_high_precision) >= 3 or (len(ultra_high_precision) == 2 and len(high_precision) >= 1):
                        vessels_with_precision += 1
        
        prob_b = vessels_with_precision / n_null_samples if n_null_samples > 0 else 0
        ci_b_lower, ci_b_upper = self.calculate_confidence_interval(prob_b, n_null_samples)
        bayes_b = self.calculate_bayes_factor(prob_b)
        
        print(f"Vessels with precision requirements: {vessels_with_precision:,} / {n_null_samples:,}")
        print(f"Probability: {prob_b:.5f}")
        print(f"95% CI: [{ci_b_lower:.5f}, {ci_b_upper:.5f}]")
        print(f"Bayes Factor: {bayes_b['bayes_factor']:.2f} ({bayes_b['evidence_strength']})")
        
        # Test c): Probability with interlocking properties
        print(f"\nTest c): Probability with interlocking properties")
        vessels_with_interlocking = 0
        extraneous_matches_encountered = 0
        for vessel in null_data:
            matches = vessel['d_over_r_phi_pi_phi2_matches']  # Only π, φ, φ² AND only d/r ratios
            if len(matches) >= self.check_n_matches:
                # Check precision requirements first (for N=3: 2 ultra-high + 1 high)
                if self.check_n_matches >= 3:
                    ultra_high_precision = [m for m in matches if m['error_percentage'] < 0.07]
                    remaining_matches = [m for m in matches if not m in ultra_high_precision]
                    high_precision = [m for m in remaining_matches if m['error_percentage'] < 0.2]

                    # Only evaluate if we have matches with sufficient precision
                    # Need at least 2 ultra-high (< 0.07%) and 1 high or ultra-high (0.07% - 0.2%)
                    if len(ultra_high_precision) >= 3 or (len(ultra_high_precision) == 2 and len(high_precision) >= 1):
                        selected_matches = ultra_high_precision + high_precision
                        if len(selected_matches) > 3:
                            extraneous_matches_encountered += 1

                        # Extract components
                        nominators = [match['ratio_name'].split('/')[0] for match in selected_matches]
                        denominators = [match['ratio_name'].split('/')[1] for match in selected_matches]

                        # Count occurrences
                        nom_counts = {}
                        den_counts = {}
                        for nom in nominators:
                            nom_counts[nom] = nom_counts.get(nom, 0) + 1
                        for den in denominators:
                            den_counts[den] = den_counts.get(den, 0) + 1
                        
                        # Check for interlocking (at least one component appearing more than once)
                        interlocking_noms = len([c for c in nom_counts.values() if c > 1])
                        interlocking_dens = len([c for c in den_counts.values() if c > 1])
                        
                        if interlocking_noms >= 1 and interlocking_dens >= 1:
                            vessels_with_interlocking += 1
        
        prob_c = vessels_with_interlocking / n_null_samples if n_null_samples > 0 else 0
        ci_c_lower, ci_c_upper = self.calculate_confidence_interval(prob_c, n_null_samples)
        bayes_c = self.calculate_bayes_factor(prob_c)
        
        print(f"Vessels with interlocking properties: {vessels_with_interlocking:,} / {n_null_samples:,}")
        print(f"Probability: {prob_c:.5f}")
        print(f"95% CI: [{ci_c_lower:.5f}, {ci_c_upper:.5f}]")
        print(f"Bayes Factor: {bayes_c['bayes_factor']:.2f} ({bayes_c['evidence_strength']})")
        
        # Analyze target vase for comparison
        target_variants = self.generate_all_measurement_variants(target_measurements)
        target_variants_series = pd.Series(target_variants)
        target_ratios = self.generate_all_meaningful_ratios(target_variants_series)
        target_all_matches = self.count_constant_matches(target_ratios)
        target_phi_pi_phi2_matches = [m for m in target_all_matches 
                                     if m['constant_name'] in ['pi', 'phi', 'phi_squared']]
        target_d_over_r_phi_pi_phi2_matches = [m for m in target_phi_pi_phi2_matches if m['is_d_over_r']]
        
        # Analyze target precision
        target_ultra_high = [m for m in target_d_over_r_phi_pi_phi2_matches if m['error_percentage'] < 0.07]
        target_high = [m for m in target_d_over_r_phi_pi_phi2_matches if 0.07 <= m['error_percentage'] < 0.2]
        
        print(f"\n=== TARGET VASE ANALYSIS ===")
        print(f"Target vase d/r π,φ,φ² matches: {len(target_d_over_r_phi_pi_phi2_matches)}")
        print(f"Ultra-high precision (< 0.07%): {len(target_ultra_high)}")
        print(f"High precision (0.07%-0.2%): {len(target_high)}")
        
        # Check interlocking in target
        if len(target_d_over_r_phi_pi_phi2_matches) >= 3:
            first_three = target_d_over_r_phi_pi_phi2_matches[:3]
            nominators = [match['ratio_name'].split('/')[0] for match in first_three]
            denominators = [match['ratio_name'].split('/')[1] for match in first_three]
            
            nom_counts = {}
            den_counts = {}
            for nom in nominators:
                nom_counts[nom] = nom_counts.get(nom, 0) + 1
            for den in denominators:
                den_counts[den] = den_counts.get(den, 0) + 1
            
            interlocking_noms = [nom for nom, count in nom_counts.items() if count > 1]
            interlocking_dens = [den for den, count in den_counts.items() if count > 1]
            
            print(f"Interlocking nominators: {interlocking_noms}")
            print(f"Interlocking denominators: {interlocking_dens}")
        
        results = {
            'question_a': {
                'vessels_count': vessels_with_n_plus,
                'probability': prob_a,
                'n_samples': n_null_samples,
                'confidence_interval': [ci_a_lower, ci_a_upper],
                'bayes_factor': bayes_a['bayes_factor'],
                'evidence_strength': bayes_a['evidence_strength'],
                'expected_by_chance': prob_a * n_null_samples
            },
            'question_b': {
                'vessels_count': vessels_with_precision,
                'probability': prob_b,
                'n_samples': n_null_samples,
                'confidence_interval': [ci_b_lower, ci_b_upper],
                'bayes_factor': bayes_b['bayes_factor'],
                'evidence_strength': bayes_b['evidence_strength'],
                'expected_by_chance': prob_b * n_null_samples
            },
            'question_c': {
                'vessels_count': vessels_with_interlocking,
                'probability': prob_c,
                'n_samples': n_null_samples,
                'confidence_interval': [ci_c_lower, ci_c_upper],
                'bayes_factor': bayes_c['bayes_factor'],
                'evidence_strength': bayes_c['evidence_strength'],
                'expected_by_chance': prob_c * n_null_samples
            },
            'target_analysis': {
                'all_matches': len(target_all_matches),
                'phi_pi_phi2_matches': len(target_phi_pi_phi2_matches),
                'd_over_r_phi_pi_phi2_matches': len(target_d_over_r_phi_pi_phi2_matches),
                'ultra_high_precision': len(target_ultra_high),
                'high_precision': len(target_high),
                'detailed_matches': target_d_over_r_phi_pi_phi2_matches
            }
        }

        if extraneous_matches_encountered:
            xn = extraneous_matches_encountered
            warning_str = f"{'='*5} WARNING {'='*50}"
            print()
            print(warning_str)
            print(f"During interlocking ratio analysis, {xn} vessels with more than 3 high- or")
            print(f"ultra-high precision ration matches were ecnountered. This will artificially")
            print(f"increase the probability of Test C. For a completely accurate simulation run,")
            print(f"the Test C implementation will need to account for resolution of chained")
            print(f"dependencies in the nominator/denominator analysis - a feature that has been")
            print(f"out of the scope of this implementation.")
            print("="*len(warning_str))
            print()
        
        return results
    
    def print_precision_probability_results(self, results):
        """
        Print comprehensive results for precision probability analysis.
        """
        print("\n" + "=" * 70)
        print("RATIO MATCH PROBABILITY ANALYSIS RESULTS")
        print("=" * 70)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tolerance Range: ±{self.tolerance_range * 100:.3f}%")
        print(f"Error Deviation: ≤{self.error_deviation}")
        print(f"Minimum Matches: {self.check_n_matches}")
        print(f"Significance Level (α): {self.alpha}")
        print()
        
        print("TEST A: Probability of finding vessel with 3+ matches (all constants)")
        print("-" * 50)
        qa = results['question_a']
        print(f"Observed vessels: {qa['vessels_count']:,} / {qa['n_samples']:,}")
        print(f"Probability: {qa['probability']:.5f}")
        print(f"95% Confidence Interval: [{qa['confidence_interval'][0]:.5f}, {qa['confidence_interval'][1]:.5f}]")
        print(f"Bayes Factor: {qa['bayes_factor']:.2f} ({qa['evidence_strength']})")
        print(f"Expected by chance: {qa['expected_by_chance']:.2f} vessels")
        print()
        
        print("TEST B: Probability with precision requirements (π, φ, φ², d/r only)")
        print("-" * 50)
        qb = results['question_b']
        print(f"Observed vessels: {qb['vessels_count']:,} / {qb['n_samples']:,}")
        print(f"Probability: {qb['probability']:.5f}")
        print(f"95% Confidence Interval: [{qb['confidence_interval'][0]:.5f}, {qb['confidence_interval'][1]:.5f}]")
        print(f"Bayes Factor: {qb['bayes_factor']:.2f} ({qb['evidence_strength']})")
        print(f"Expected by chance: {qb['expected_by_chance']:.2f} vessels")
        print()
        
        print("TEST C: Probability with interlocking properties")
        print("-" * 50)
        qc = results['question_c']
        print(f"Observed vessels: {qc['vessels_count']:,} / {qc['n_samples']:,}")
        print(f"Probability: {qc['probability']:.5f}")
        print(f"95% Confidence Interval: [{qc['confidence_interval'][0]:.5f}, {qc['confidence_interval'][1]:.5f}]")
        print(f"Bayes Factor: {qc['bayes_factor']:.2f} ({qc['evidence_strength']})")
        print(f"Expected by chance: {qc['expected_by_chance']:.2f} vessels")
        print()
        
        # Target analysis
        print("TARGET VASE ANALYSIS")
        print("-" * 50)
        ta = results['target_analysis']
        print(f"Total matches found: {ta['all_matches']}")
        print(f"π, φ, φ² matches: {ta['phi_pi_phi2_matches']}")
        print(f"d/r π, φ, φ² matches: {ta['d_over_r_phi_pi_phi2_matches']}")
        print(f"Ultra-high precision (< 0.07%): {ta['ultra_high_precision']}")
        print(f"High precision (0.07%-0.2%): {ta['high_precision']}")
        
        if ta['detailed_matches']:
            print("\nDetailed d/r π, φ, φ² matches:")
            for i, match in enumerate(ta['detailed_matches'][:5], 1):  # Show first 5
                print(f"  {i}. {match['ratio_name']} = {match['ratio_value']:.4f}")
                print(f"     ≈ {match['constant_symbol']} ({match['constant_name']})")
                print(f"     Error: {match['error_percentage']:.2f}%")
                print(f"     Deviation: {match['absolute_deviation']:.4f}")
    
    def export_results(self, results, filename=None):
        """
        Export detailed results to JSON file for further analysis.
        
        Args:
            results (dict): Analysis results
            filename (str): Output filename (optional)
        """
        if filename is None:
            filename = f"vase_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Deep copy to avoid modifying original results
        import copy
        exportable_results = copy.deepcopy(results)
        
        # Function to convert all numpy types and other non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(make_serializable(item) for item in obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.str_, np.bytes_)):
                return str(obj)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
            else:
                # For any other type, convert to string
                return str(obj)
        
        # Convert everything to JSON serializable format
        exportable_results = make_serializable(exportable_results)
        
        # Add metadata
        exportable_results['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'tolerance_range': float(self.tolerance_range),
            'error_deviation': float(self.error_deviation),
            'check_n_matches': int(self.check_n_matches),
            'alpha_level': float(self.alpha),
            'target_constants': [c['name'] for c in self.target_constants]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(exportable_results, f, indent=2, default=str)
            print(f"Results exported to {filename}")
        except Exception as e:
            print(f"Error exporting results: {e}")
            # Try with more aggressive string conversion
            try:
                with open(filename, 'w') as f:
                    json.dump(exportable_results, f, indent=2, default=lambda x: str(x))
                print(f"Results exported to {filename} (with fallback conversion)")
            except Exception as e2:
                print(f"Fallback export also failed: {e2}")

def main():
    """
    Main function demonstrating the analysis with PV001 vase data.
    Replace these measurements with actual target vase measurements.
    """

    # 10,000 for demo, increase for more accurate run
    n_null_samples = 100000

    print("Ancient Egyptian Vase Mathematical Constant Analysis")
    print("=" * 55)
    
    # Actual measurements of artifact catalogue ID PV001
    target_vase = {
        'opening': 37.45556,      # Opening diameter (example value)
        'top_max': 58.93250,      # Lip diameter (example value)
        'top_min': 48.99610,      # Neck diameter (example value)  
        'body_max': 84.82219,     # Maximum body diameter (example value)
        'foot_min': 44.42495,     # Minimum foot diameter (example value)
        'foot_max': 45.04342,     # Maximum foot diameter (example value)
        'handles_max': 59.21801   # Handle attachment diameter (example value)
    }
    print("Target vase measurements:")
    for name, value in target_vase.items():
        print(f"  {name}: {value:.2f} mm")
    print()
    
    # Initialize analyzer with default parameters - 0.2% tolerance, 0.4 deviation, looking for 3+ matches
    analyzer = StatisticalVaseAnalyzer(
        tolerance_range=0.002,    # 0.2% tolerance
        error_deviation=0.4,      # 0.4 maximum absolute deviation
        check_n_matches=3,        # Looking for 3+ matches
        alpha=0.05,              # Standard significance level
        random_seed=42           # For reproducible results
    )
    
    # Perform precision probability analysis
    precision_results = analyzer.analyze_precision_probabilities(
        target_measurements=target_vase,
        n_null_samples=n_null_samples
    )
    
    # Display detailed results
    analyzer.print_precision_probability_results(precision_results)
    
    # Export results for further analysis
    analyzer.export_results(precision_results, 'precision_probability_analysis_results.json')
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    
    print("A) Probability of finding vessel with 3+ matches (all constants):")
    qa = precision_results['question_a']
    print(f"   p = {qa['probability']:.5f} (95% CI: [{qa['confidence_interval'][0]:.5f}, {qa['confidence_interval'][1]:.5f}])")
    print(f"   Bayes Factor: {qa['bayes_factor']:.2f} ({qa['evidence_strength']})")
    print(f"   Expected by chance: {qa['expected_by_chance']:.2f} vessels")
    print(f"   Chance: {qa['probability']*100:.4f}% ({qa['expected_by_chance']}/{n_null_samples})")

    print("\nB) Probability with precision requirements (2×<0.07%, 1×<0.2%, π/φ/φ², d/r only):")
    qb = precision_results['question_b']
    print(f"   p = {qb['probability']:.5f} (95% CI: [{qb['confidence_interval'][0]:.5f}, {qb['confidence_interval'][1]:.5f}])")
    print(f"   Bayes Factor: {qb['bayes_factor']:.2f} ({qb['evidence_strength']})")
    print(f"   Expected by chance: {qb['expected_by_chance']:.2f} vessels")
    print(f"   Chance: {qb['probability']*100:.4f}% ({qb['expected_by_chance']}/{n_null_samples})")

    print("\nC) Probability with interlocking properties:")
    qc = precision_results['question_c']
    print(f"   p = {qc['probability']:.5f} (95% CI: [{qc['confidence_interval'][0]:.5f}, {qc['confidence_interval'][1]:.5f}])")
    print(f"   Bayes Factor: {qc['bayes_factor']:.2f} ({qc['evidence_strength']})")
    print(f"   Chance: {qc['probability']*100:.4f}% ({qc['expected_by_chance']}/{n_null_samples})")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()