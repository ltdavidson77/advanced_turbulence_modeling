import numpy as np
from decimal import Decimal
from precision_controls import PrecisionControls
from quantum_state import QuantumState
from typing import Callable, Dict, Optional, Tuple

# Global precision config
PRECISION_CONFIG = PrecisionControls()

class FeatureMapper:
    """Maps compressed quantum states of any dimension to ML-ready feature embeddings."""
    
    def __init__(self):
        """Initialize the feature mapper with an empty transform registry."""
        self.transform_registry: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    
    def register_transform(self, transform_id: str, transform_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Register a custom feature transformation function for N-dimensional arrays."""
        self.transform_registry[transform_id] = transform_fn
    
    def extract_raw_features(self, qstate: QuantumState, 
                            target_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Extract raw features from a quantum state, supporting N-dimensional tensors."""
        if qstate.metadata["is_compressed"]:
            qstate.decompress()
        
        # Convert to float array, preserving original shape
        features = np.vectorize(float)(qstate.state)
        
        # Reshape to target shape if provided, handling multi-dimensional cases
        if target_shape:
            current_size = features.size
            target_size = np.prod(target_shape)
            
            flat_features = features.flatten()
            if current_size < target_size:
                # Pad with zeros to match target size
                padded = np.pad(flat_features, (0, target_size - current_size), 
                              mode='constant', constant_values=0.0)
            elif current_size > target_size:
                # Trim excess elements
                padded = flat_features[:target_size]
            else:
                padded = flat_features
            
            try:
                features = padded.reshape(target_shape)
            except ValueError as e:
                raise ValueError(f"Cannot reshape {current_size} elements into {target_shape}: {e}")
        
        return features
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize N-dimensional features to unit norm across all elements."""
        norm = np.linalg.norm(features.flatten())
        return features / norm if norm > 0 else features
    
    def apply_transform(self, features: np.ndarray, transform_id: str) -> np.ndarray:
        """Apply a registered transformation to N-dimensional features."""
        if transform_id not in self.transform_registry:
            raise ValueError(f"Transform {transform_id} not registered")
        return self.transform_registry[transform_id](features)
    
    def map_features(self, qstate: QuantumState, 
                    target_shape: Optional[Tuple[int, ...]] = None, 
                    transform_id: Optional[str] = None) -> np.ndarray:
        """Full feature mapping pipeline for N-dimensional tensors."""
        # Extract raw features
        raw_features = self.extract_raw_features(qstate, target_shape)
        
        # Normalize across all dimensions
        norm_features = self.normalize_features(raw_features)
        
        # Apply custom transform if specified
        if transform_id:
            norm_features = self.apply_transform(norm_features, transform_id)
        
        return norm_features

if __name__ == "__main__":
    # Test the feature mapper with multi-dimensional data
    mapper = FeatureMapper()
    
    # Test state: 3D tensor with mixed base values
    initial_state = np.array([
        [[float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
         [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]],
        [[float(Decimal('10') ** 6), float(Decimal('2') ** 10)],
         [float(Decimal('10') ** 4), 0.0]]
    ])
    qstate = QuantumState(initial_state, compress_depth=3)
    
    # Register a transform (e.g., scaling across all dimensions)
    def scale_features(features: np.ndarray) -> np.ndarray:
        return features * 2.0
    
    mapper.register_transform("scale", scale_features)
    
    # Map features to various shapes
    # 1. Keep original 3D shape (2, 2, 2)
    features_3d = mapper.map_features(qstate, transform_id="scale")
    print("Mapped 3D Features (original shape):\n", features_3d)
    print("Shape:", features_3d.shape)
    
    # 2. Reshape to 2D (4, 2)
    features_2d = mapper.map_features(qstate, target_shape=(4, 2), transform_id="scale")
    print("\nMapped 2D Features (4, 2):\n", features_2d)
    print("Shape:", features_2d.shape)
    
    # 3. Reshape to 1D (8,)
    features_1d = mapper.map_features(qstate, target_shape=(8,), transform_id="scale")
    print("\nMapped 1D Features (8,):\n", features_1d)
    print("Shape:", features_1d.shape)
    
    # 4. Oversized reshape with padding (3, 3)
    features_padded = mapper.map_features(qstate, target_shape=(3, 3), transform_id="scale")
    print("\nMapped Padded Features (3, 3):\n", features_padded)
    print("Shape:", features_padded.shape)
