"""
SPEAR: Structural Pore Efficient Analysis and Representation


"""

import os
import glob
import logging
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from Bio.PDB import *
import MDAnalysis as mda
from MDAnalysis.analysis import hole2
import panel as pn

# Import base functionality - remove the circular import
from frtmalign_2_msa_holeanalysis import (
    batch_hole,
    hole_annotation,
    single_frtmalign,
    get_struct,
    strip_tm_chains, 
    paths_dic,
    single_hole
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SPEAR')


warnings.filterwarnings('ignore', category=UserWarning)

# Defaults
DEFAULT_CONFIG = {
    'clustering': {
        'n_clusters': 3,
        'min_samples': 2,
        'eps': 0.5,
        'method': 'kmeans'
    },
    'visualization': {
        'color_scheme': 'viridis',
        'fig_size': (10, 6),
        'dpi': 300
    },
    'analysis': {
        'min_pore_radius': 1.0,
        'na_permissive_threshold': 3, #This is a guess; need to tune this based on experiments
        'ca_permissive_threshold': 4 #This is a guess; need to tune this based on experiments
    },
    'validation': {
        'min_structures': 3,
        'max_outlier_percentage': 10, #This is a guess; need to tune this based on experiments
        'quality_threshold': 0.9 #This is a guess; need to tune this based on experiments
    }
}

@dataclass
class SPEARConfig:
    """Configuration class"""
    clustering: dict
    visualization: dict
    analysis: dict
    validation: dict
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

class PoreProfile:
    """Class for analyzing and storing pore characteristics"""
    def __init__(self, radius_data: np.ndarray, pdb_id: str, config: Optional[SPEARConfig] = None):
        self.radius_data = radius_data
        self.pdb_id = pdb_id
        self.config = config or SPEARConfig.from_dict(DEFAULT_CONFIG)
        
        # Calculate basic statistics
        self.min_radius = np.min(radius_data)
        self.max_radius = np.max(radius_data)
        self.mean_radius = np.mean(radius_data)
        self.std_radius = np.std(radius_data)
        
        # Classify profile
        self.profile_type = self._classify_profile()
        self.features = self._extract_features()
        
    def _classify_profile(self) -> str:
        """Classify pore profile based on radius characteristics"""
        if self.min_radius < self.config.analysis['min_pore_radius']:
            return "closed"
        elif self.min_radius < self.config.analysis['na_permissive_threshold']:
            return "Na_permissive"
        else:
            return "Ca_permissive"
            
    def _extract_features(self) -> np.ndarray:
        """Extract features for clustering"""
        return np.array([
            self.min_radius,
            self.max_radius,
            self.mean_radius,
            self.std_radius,
            np.percentile(self.radius_data, 25),
            np.percentile(self.radius_data, 75)
        ])
        
    def plot_profile(self, ax=None):
        """Plot pore radius profile"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.visualization['fig_size'])
        
        ax.plot(self.radius_data, label=f'{self.pdb_id} ({self.profile_type})')
        ax.axhline(y=self.config.analysis['na_permissive_threshold'], 
                  color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=self.config.analysis['ca_permissive_threshold'], 
                  color='g', linestyle='--', alpha=0.5)
        ax.set_ylabel('Pore Radius (Å)')
        ax.set_xlabel('Position')
        ax.legend()
        
        return ax

class SPEARAnalysis:
    """Main class for SPEAR analysis pipeline"""
    def __init__(self, config: Optional[SPEARConfig] = None):
        self.config = config or SPEARConfig.from_dict(DEFAULT_CONFIG)
        self.profiles = {}
        self.clusters = None
        self.alignment_groups = None
        self.validation_results = {}
        
    def run_analysis(self, directory: str, category_df: pd.DataFrame) -> Dict:
        """Run complete SPEAR analysis pipeline"""
        logger.info("Starting SPEAR analysis pipeline")
        
        # Analyze profiles
        self.profiles = self._analyze_pore_profiles(directory)
        
        # Validate data
        self._validate_data()
        
        # Perform clustering
        self.clusters = self._cluster_structures()
        
        # Define alignment strategy
        self.alignment_groups = self._define_alignment_strategy(category_df)
        
        # Generate reports
        results = self._generate_results()
        
        logger.info("SPEAR analysis pipeline completed")
        return results

    def _analyze_pore_profiles(self, directory: str) -> Dict[str, PoreProfile]:
        """Analyze all pore profiles in directory"""
        profiles = {}
        radius_files = glob.glob(os.path.join(directory, '*/*_radius.csv'))
        
        if not radius_files:
            raise ValueError(f"No radius files found in {directory}")
            
        for radius_file in radius_files:
            pdb_id = os.path.basename(radius_file).split('_')[0]
            try:
                radius_df = pd.read_csv(radius_file)
                profiles[pdb_id] = PoreProfile(radius_df['radius'].values, pdb_id, self.config)
            except Exception as e:
                logger.warning(f"Failed to process {radius_file}: {str(e)}")
                
        return profiles
        
    def _validate_data(self):
        """Helper function debugging: validate data completeness"""
        logger.info("Validating data quality")
        
        # Check minimum number of structures
        if len(self.profiles) < self.config.validation['min_structures']:
            raise ValueError(f"Insufficient structures for analysis. "
                           f"Found {len(self.profiles)}, need at least "
                           f"{self.config.validation['min_structures']}")
        
        # Check for outliers
        outliers = self._detect_outliers()
        outlier_percentage = (len(outliers) / len(self.profiles)) * 100
        
        if outlier_percentage > self.config.validation['max_outlier_percentage']:
            logger.warning(f"High percentage of outliers detected: {outlier_percentage:.1f}%")
            
        self.validation_results['outliers'] = outliers
        self.validation_results['outlier_percentage'] = outlier_percentage
        
    def _detect_outliers(self) -> List[str]:
        """Detect outlier structures based on profile characteristics"""
        features = np.array([p.features for p in self.profiles.values()])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use DBSCAN for outlier detection
        dbscan = DBSCAN(eps=self.config.clustering['eps'], 
                       min_samples=self.config.clustering['min_samples'])
        labels = dbscan.fit_predict(features_scaled)
        
        # Structures labeled as -1 are outliers
        outlier_indices = np.where(labels == -1)[0]
        return [list(self.profiles.keys())[i] for i in outlier_indices]
        
    def _cluster_structures(self) -> pd.DataFrame:
        """Cluster structures based on pore characteristics"""
        logger.info("Clustering structures")
        
        # Prepare features for clustering
        features = np.array([p.features for p in self.profiles.values()])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine optimal number of clusters if not specified
        if self.config.clustering['method'] == 'kmeans':
            if self.config.clustering['n_clusters'] is None:
                n_clusters = self._optimize_cluster_number(features_scaled)
            else:
                n_clusters = self.config.clustering['n_clusters']
                
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            
        else:
            # Hierarchical clustering
            linkage_matrix = linkage(features_scaled, method='ward')
            labels = fcluster(linkage_matrix, 
                            t=self.config.clustering['n_clusters'], 
                            criterion='maxclust')
            
        # Create clustering results
        cluster_results = pd.DataFrame({
            'PDB_ID': list(self.profiles.keys()),
            'Cluster': labels,
            'Profile_Type': [p.profile_type for p in self.profiles.values()],
            'Min_Radius': [p.min_radius for p in self.profiles.values()],
            'Max_Radius': [p.max_radius for p in self.profiles.values()],
            'Mean_Radius': [p.mean_radius for p in self.profiles.values()]
        })
        
        return cluster_results
        
    def _optimize_cluster_number(self, features: np.ndarray) -> int:
        """Determine optimal number of clusters using silhouette score"""
        silhouette_scores = []
        k_range = range(2, min(len(features), 10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
            
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return optimal_k
        
    def _define_alignment_strategy(self, category_df: pd.DataFrame) -> Dict:
        """Define structure alignment strategy based on clustering and metadata"""
        logger.info("Defining alignment strategy")
        
        alignment_groups = {
            'within_cluster': [],
            'between_cluster_representatives': [],
            'profile_type': [],
            'ligand_bound': []
        }
        
        # Within-cluster alignments
        for cluster in self.clusters['Cluster'].unique():
            cluster_pdbs = self.clusters[self.clusters['Cluster'] == cluster]['PDB_ID'].tolist()
            alignment_groups['within_cluster'].extend(list(itertools.combinations(cluster_pdbs, 2)))
            
        # Select/align cluster representatives
        representatives = self._select_cluster_representatives()
        alignment_groups['between_cluster_representatives'].extend(
            list(itertools.combinations(representatives, 2)))
            
        # Profile type alignments
        for profile_type in self.clusters['Profile_Type'].unique():
            profile_pdbs = self.clusters[
                self.clusters['Profile_Type'] == profile_type]['PDB_ID'].tolist()
            alignment_groups['profile_type'].extend(
                list(itertools.combinations(profile_pdbs[:5], 2)))  # Limit to top 5
                
        # Ligand-based alignments
        if 'Ligand' in category_df.columns:
            ligand_structures = category_df[category_df['Ligand'].notna()]['PDB ID'].tolist()
            alignment_groups['ligand_bound'].extend(
                [(l, r) for l, r in itertools.combinations(ligand_structures, 2)
                 if l in self.profiles and r in self.profiles])
                
        return alignment_groups
        
    def _select_cluster_representatives(self) -> List[str]:
        """Select representative structures from each cluster"""
        representatives = []
        
        for cluster in self.clusters['Cluster'].unique():
            cluster_subset = self.clusters[self.clusters['Cluster'] == cluster]
            
            # Select structure closest to cluster centroid
            centroid = cluster_subset[['Min_Radius', 'Max_Radius', 'Mean_Radius']].mean()
            distances = ((cluster_subset[['Min_Radius', 'Max_Radius', 'Mean_Radius']] - centroid) ** 2).sum(axis=1)
            representative = cluster_subset.iloc[distances.argmin()]['PDB_ID']
            representatives.append(representative)
            
        return representatives

    def _generate_interactive_visualizations(self) -> Dict[str, go.Figure]:
        """Generate interactive Plotly visualizations"""
        logger.info("Generating interactive visualizations")
        
        visualizations = {}
        
        # Cluster visualization
        fig = px.scatter_3d(self.clusters,
                           x='Min_Radius',
                           y='Max_Radius',
                           z='Mean_Radius',
                           color='Cluster',
                           symbol='Profile_Type',
                           hover_data=['PDB_ID'],
                           title='Structure Clustering Analysis')
        visualizations['cluster_3d'] = fig
        
        # Profile comparison
        fig = go.Figure()
        for pdb_id, profile in self.profiles.items():
            fig.add_trace(go.Scatter(y=profile.radius_data,
                                   name=f'{pdb_id} ({profile.profile_type})',
                                   mode='lines'))
        fig.update_layout(title='Pore Radius Profiles Comparison',
                         xaxis_title='Position',
                         yaxis_title='Pore Radius (Å)')
        visualizations['profile_comparison'] = fig
        
        # Cluster distribution
        fig = px.sunburst(self.clusters,
                         path=['Cluster', 'Profile_Type'],
                         values='Mean_Radius',
                         title='Cluster and Profile Type Distribution')
        visualizations['cluster_distribution'] = fig
        
        return visualizations
        
    def generate_html_report(self, output_dir: str):
        """Generate interactive HTML report"""
        logger.info("Generating HTML report")
        
        # Initialize Panel dashboard
        pn.extension()
        
        # Create visualizations
        visualizations = self._generate_interactive_visualizations()
        
        # Create dashboard components
        header = pn.pane.Markdown("""
        # SPEAR Analysis Report
        ## Structure Analysis Results
        """)
        
        # Statistics summary
        stats = pd.DataFrame({
            'Metric': ['Total Structures', 'Clusters', 'Profile Types', 'Outliers'],
            'Value': [
                len(self.profiles),
                len(self.clusters['Cluster'].unique()),
                len(self.clusters['Profile_Type'].unique()),
                self.validation_results['outlier_percentage']
            ]
        })
        stats_table = pn.pane.DataFrame(stats, width=400)
        
        # Create tabs for different visualizations
        tabs = pn.Tabs(
            ('3D Clustering', pn.pane.Plotly(visualizations['cluster_3d'])),
            ('Profile Comparison', pn.pane.Plotly(visualizations['profile_comparison'])),
            ('Distribution', pn.pane.Plotly(visualizations['cluster_distribution']))
        )
        
        # Combine components
        dashboard = pn.Column(
            header,
            pn.Row(stats_table),
            tabs,
            sizing_mode='stretch_width'
        )
        
        # Save dashboard
        dashboard.save(os.path.join(output_dir, 'spear_report.html'))
        
    def generate_analysis_summary(self, output_dir: str):
        """This is not a high priority for now but this would generate analysis summary files"""
        logger.info("Generating analysis summary")
        
        # Save cluster results
        self.clusters.to_csv(os.path.join(output_dir, 'cluster_results.csv'))
        
        # Save alignment strategy
        with open(os.path.join(output_dir, 'alignment_strategy.json'), 'w') as f:
            json.dump(self.alignment_groups, f, indent=2)
            
        # Save profile data
        profile_data = pd.DataFrame({
            'PDB_ID': list(self.profiles.keys()),
            'Profile_Type': [p.profile_type for p in self.profiles.values()],
            'Min_Radius': [p.min_radius for p in self.profiles.values()],
            'Max_Radius': [p.max_radius for p in self.profiles.values()],
            'Mean_Radius': [p.mean_radius for p in self.profiles.values()],
            'Std_Radius': [p.std_radius for p in self.profiles.values()]
        })
        profile_data.to_csv(os.path.join(output_dir, 'profile_data.csv'))
        
        # Generate statistical summary
        summary_stats = self._generate_summary_statistics()
        with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(summary_stats)
            
    def _generate_summary_statistics(self) -> str:
        """Generate statistical summary of analysis results"""
        summary = []
        summary.append("SPEAR Analysis Summary")
        summary.append("=" * 20)
        
        # Basic statistics
        summary.append(f"\nTotal structures analyzed: {len(self.profiles)}")
        summary.append(f"Number of clusters: {len(self.clusters['Cluster'].unique())}")
        summary.append(f"Number of profile types: {len(self.clusters['Profile_Type'].unique())}")
        
        # Profile type distribution
        summary.append("\nProfile Type Distribution:")
        profile_dist = self.clusters['Profile_Type'].value_counts()
        for ptype, count in profile_dist.items():
            summary.append(f"  {ptype}: {count} structures ({count/len(self.profiles)*100:.1f}%)")
            
        # Cluster statistics
        summary.append("\nCluster Statistics:")
        for cluster in self.clusters['Cluster'].unique():
            cluster_data = self.clusters[self.clusters['Cluster'] == cluster]
            summary.append(f"\nCluster {cluster}:")
            summary.append(f"  Structures: {len(cluster_data)}")
            summary.append(f"  Average pore radius: {cluster_data['Mean_Radius'].mean():.2f} Å")
            summary.append(f"  Profile types: {dict(cluster_data['Profile_Type'].value_counts())}")
            
        # Validation results
        summary.append("\nValidation Results:")
        summary.append(f"  Outlier percentage: {self.validation_results['outlier_percentage']:.1f}%")
        summary.append(f"  Number of outliers: {len(self.validation_results['outliers'])}")
        
        return "\n".join(summary)


#@log_execution_time
def run_spear_pipeline(clean_dir: str, frtmalign_dir: str, frtmalign_path: str,
                      original_dir: str, clean_dir_path: str, category_df: pd.DataFrame,
                      vdw_file: str, pore_point: Union[str, List[float]],
                      paths: Dict,
                      config: Optional[Dict] = None):
    """
    Main entry point for SPEAR pipeline
    """
    # Initialize configuration
    spear_config = SPEARConfig.from_dict(config if config else DEFAULT_CONFIG)
    
    # Create output directory structure
    output_dir = os.path.join(frtmalign_dir, 'spear_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SPEAR analysis
    spear = SPEARAnalysis(spear_config)
    
    try:
        # First run the standard preprocessing steps
        logger.info("Running preprocessing steps")
        
        # Convert DataFrame index to strings if they aren't already
        category_df.index = category_df.index.astype(str)
        struct_info_dic = category_df.to_dict('index')
        
        for pdb_id, value in struct_info_dic.items():
            success = get_struct(pdb_id, original_dir, paths['provided_struct'])
            if success:
                strip_tm_chains(clean_dir, pdb_id, 
                              os.path.join(original_dir, f"{pdb_id}.pdb"), 
                              value['tm_chains'])
        
        # Now run HOLE analysis
        logger.info("Running initial HOLE analysis")
        hole_results = []
        
        for pdb_file in glob.glob(os.path.join(clean_dir, "*_clean.pdb")):
            pdb_id = os.path.basename(pdb_file)[:4]
            hole_out_dir = os.path.join(output_dir, f"hole_analysis_{pdb_id}")
            os.makedirs(hole_out_dir, exist_ok=True)
            
            # Run HOLE analysis and collect results
            result = single_hole(
                filename=pdb_file,
                short_filename=pdb_id,
                pdb_id=pdb_id,
                out_dir=hole_out_dir,
                input_df=category_df,
                vdw_file=vdw_file,
                pore_point=pore_point
            )
            hole_results.append(result)
            
        # Verify HOLE analysis completed successfully
        logger.info("Verifying HOLE analysis results")
        radius_files = glob.glob(os.path.join(output_dir, "hole_analysis_*/*_radius.csv"))
        if not radius_files:
            raise RuntimeError("HOLE analysis did not generate expected radius files")
            
        # Now run SPEAR analysis
        logger.info("Running SPEAR analysis")
        results = spear.run_analysis(output_dir, category_df)
        
        # Perform strategic alignments
        # TODO look at parallelizing this
        logger.info("Performing strategic alignments")
        for group_type, alignments in spear.alignment_groups.items():
            group_dir = os.path.join(output_dir, f"alignments_{group_type}")
            os.makedirs(group_dir, exist_ok=True)
            
            for struct1, struct2 in alignments:
                single_frtmalign(
                    os.path.join(clean_dir, f"{struct1}_clean.pdb"),
                    os.path.join(clean_dir, f"{struct2}_clean.pdb"),
                    group_dir,
                    struct1,
                    struct2,
                    frtmalign_path,
                    original_dir,
                    clean_dir_path
                )
                
        # Generate reports
        logger.info("Generating reports")
        spear.generate_html_report(output_dir)
        spear.generate_analysis_summary(output_dir)
        
        return results, spear
        
    except Exception as e:
        logger.error(f"Error in SPEAR pipeline: {str(e)}")
        raise
