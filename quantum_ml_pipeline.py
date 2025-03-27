import numpy as np
from decimal import Decimal
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine
from quantum_state import QuantumState
from enterprise_framework import QuantumEnterpriseFramework
from typing import Dict, Any, Optional, Callable

# Global instances
PRECISION_CONFIG = PrecisionControls()
COMPRESSION_ENGINE = SpatialLogCompressionEngine()

class QuantumMLPipeline:
    """Machine learning pipeline using compressed quantum states."""
    
    def __init__(self, compress_depth: int = 3, framework: Optional[QuantumEnterpriseFramework] = None):
        """
        Initialize the ML pipeline.
        
        Args:
            compress_depth (int): Depth for logarithmic compression.
            framework (QuantumEnterpriseFramework, optional): External framework instance.
        """
        self.compress_depth = compress_depth
        self.framework = framework if framework else QuantumEnterpriseFramework(compress_depth=compress_depth)
        self.feature_transforms: Dict[str, Callable] = {}
    
    def register_feature_transform(self, transform_id: str, transform_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Register a custom feature transformation function."""
        self.feature_transforms[transform_id] = transform_fn
    
    def preprocess_data(self, data: np.ndarray, state_id: str) -> None:
        """Preprocess input data into a compressed quantum state."""
        # Normalize and register as a quantum state
        self.framework.register_state(state_id, data)
        self.framework.compress_state(state_id)
    
    def apply_quantum_transform(self, state_id: str, gate: Optional[np.ndarray] = None) -> None:
        """Apply a quantum transformation to the compressed state."""
        if state_id not in self.framework.state_registry:
            raise ValueError(f"State {state_id} not registered")
        self.framework.process_quantum_workflow(state_id, gate=gate)
    
    def extract_features(self, state_id: str, target_shape: tuple, 
                        transform_id: Optional[str] = None) -> np.ndarray:
        """Extract features from the quantum state for ML."""
        weights = self.framework.bridge_ai_weights(state_id, target_shape)
        
        if transform_id and transform_id in self.feature_transforms:
            weights = self.feature_transforms[transform_id](weights)
        
        return weights
    
    def run_pipeline(self, data: np.ndarray, state_id: str, 
                    gate: Optional[np.ndarray] = None, 
                    target_shape: Optional[tuple] = None, 
                    transform_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the full ML pipeline: preprocess, transform, extract."""
        # Preprocess and compress
        self.preprocess_data(data, state_id)
        
        # Apply quantum transform
        self.apply_quantum_transform(state_id, gate)
        
        # Extract features
        result = {
            "compressed_size_bytes": self.framework.state_registry[state_id].compressed_size,
            "measurement_results": self.framework.results_cache.get(state_id, {})
        }
        
        if target_shape:
            result["features"] = self.extract_features(state_id, target_shape, transform_id)
        
        return result

if __name__ == "__main__":
    # Test the pipeline
    pipeline = QuantumMLPipeline(compress_depth=3)
    
    # Test data: 2D tensor with mixed base values
    input_data = np.array([
        [float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
        [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]
    ])
    
    # Define a simple feature transform (e.g., normalization)
    def normalize_features(features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features
    
    pipeline.register_feature_transform("normalize", normalize_features)
    
    # Hadamard gate
    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Run pipeline
    result = pipeline.run_pipeline(
        data=input_data,
        state_id="ml_test",
        gate=gate,
        target_shape=(2, 2),
        transform_id="normalize"
    )
    
    print("Compressed Size (bytes):", result["compressed_size_bytes"])
    print("Measurement Results:", result["measurement_results"])
    print("Extracted Features:\n", result["features"])
