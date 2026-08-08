import os
import shutil
import time
from datetime import datetime
from typing import Dict, Any, Optional
import yaml
import json
import numpy as np


class ExperimentManager:
    """Documentation for this public API is provided in English."""
    
    def __init__(self, experiment_name: Optional[str] = None, base_dir: Optional[str] = None):
        """
        
        Args:
        """
        if base_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            self.base_dir = os.path.join(project_root, 'experiments')
        else:
            self.base_dir = base_dir
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
        
        self._create_experiment_structure()
        
        self.start_time = time.time()
        self.config = {}
        
    def _generate_experiment_name(self) -> str:
        """Documentation for this public API is provided in English."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sac_experiment_{timestamp}"
    
    def _create_experiment_structure(self):
        """Documentation for this public API is provided in English."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.models_dir = os.path.join(self.experiment_dir, "models")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        self.results_dir = os.path.join(self.experiment_dir, "results")
        self.configs_dir = os.path.join(self.experiment_dir, "configs")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        
        print(f"Created experiment directory: {self.experiment_dir}")
    
    def save_config(self, config: Dict[str, Any], config_name: str = "experiment_config.yaml"):
        """Documentation for this public API is provided in English."""
        config_path = os.path.join(self.configs_dir, config_name)
        
        config_with_metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_with_metadata, f, default_flow_style=False, allow_unicode=True)
        
        self.config = config
        print(f"Configuration saved to: {config_path}")
        return config_path
    
    def save_model(self, model_path: str, model_name: Optional[str] = None):
        """Documentation for this public API is provided in English."""
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}")
            return None
            
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        dest_path = os.path.join(self.models_dir, model_name)
        shutil.copy2(model_path, dest_path)
        print(f"Model saved to: {dest_path}")
        return dest_path
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        filename: str = "training_results.json", 
        metadata: Optional[Dict[str, Any]] = None,
        compute_resources: Optional[Dict[str, Any]] = None
    ):
        """
        
        Args:
        """
        results_path = os.path.join(self.results_dir, filename)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert results to JSON-serializable format
        serializable_results = convert_numpy_types(results)
        
        results_with_metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time,
            "results": serializable_results,
            "metadata": convert_numpy_types(metadata) if metadata is not None else {}
        }
        
        if compute_resources is not None:
            results_with_metadata["compute_resources"] = convert_numpy_types(compute_resources)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {results_path}")
        return results_path
    
    def save_plot(self, plot_path: str, plot_name: Optional[str] = None):
        """Documentation for this public API is provided in English."""
        if not os.path.exists(plot_path):
            print(f"Warning: Plot file not found: {plot_path}")
            return None
            
        if plot_name is None:
            plot_name = os.path.basename(plot_path)
        
        dest_path = os.path.join(self.results_dir, plot_name)
        shutil.copy2(plot_path, dest_path)
        print(f"Plot saved to: {dest_path}")
        return dest_path
    
    def save_metrics(self, metrics_path: str, metrics_name: Optional[str] = None):
        """Documentation for this public API is provided in English."""
        if not os.path.exists(metrics_path):
            print(f"Warning: Metrics file not found: {metrics_path}")
            return None
            
        if metrics_name is None:
            metrics_name = os.path.basename(metrics_path)
        
        dest_path = os.path.join(self.results_dir, metrics_name)
        shutil.copy2(metrics_path, dest_path)
        print(f"Metrics saved to: {dest_path}")
        return dest_path
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Documentation for this public API is provided in English."""
        duration = time.time() - self.start_time
        
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration_seconds": duration,
            "duration_formatted": f"{duration:.2f}s",
            "models_count": len(os.listdir(self.models_dir)),
            "results_count": len(os.listdir(self.results_dir)),
            "configs_count": len(os.listdir(self.configs_dir))
        }
        
        return summary
    
    def print_experiment_summary(self):
        """Documentation for this public API is provided in English."""
        summary = self.get_experiment_summary()
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*50)
    
    def cleanup_temp_files(self, temp_patterns: list = None):
        """Documentation for this public API is provided in English."""
        if temp_patterns is None:
            temp_patterns = ["*.pth", "*.png", "*.npz", "*.json"]
        
        current_dir = os.getcwd()
        cleaned_files = []
        
        for pattern in temp_patterns:
            import glob
            files = glob.glob(pattern)
            for file in files:
                try:
                    os.remove(file)
                    cleaned_files.append(file)
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")
        
        if cleaned_files:
            print(f"Cleaned up {len(cleaned_files)} temporary files")
        else:
            print("No temporary files to clean up")
    
    def archive_experiment(self, archive_name: Optional[str] = None):
        """Documentation for this public API is provided in English."""
        if archive_name is None:
            archive_name = f"{self.experiment_name}.zip"
        
        archive_path = os.path.join(self.base_dir, archive_name)
        
        try:
            shutil.make_archive(
                os.path.splitext(archive_path)[0],
                'zip',
                self.experiment_dir
            )
            print(f"Experiment archived to: {archive_path}")
            return archive_path
        except Exception as e:
            print(f"Error archiving experiment: {e}")
            return None


def create_experiment_from_config(config_path: str, experiment_name: Optional[str] = None) -> ExperimentManager:
    """Documentation for this public API is provided in English."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    exp_manager = ExperimentManager(experiment_name)
    
    exp_manager.save_config(config)
    
    return exp_manager 