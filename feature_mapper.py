from decimal import Decimal
from typing import Callable, Dict, Optional, Tuple, List, Any
from precision_controls import PrecisionControls
from quantum_state import QuantumState
from quantum_tensor import QuantumTensor
from api_integration import QuantumLLMFusion

# Global precision config
PRECISION_CONFIG: PrecisionControls = PrecisionControls()

class FeatureMapper:
    """Maps compressed quantum states of any dimension to ML-ready feature embeddings."""
    
    def __init__(self):
        """Initialize the feature mapper with an empty transform registry."""
        self.transform_registry: Dict[str, Callable[['QuantumTensor'], 'QuantumTensor']] = {}
        self.llm_fusion = QuantumLLMFusion()
    
    def register_transform(self, transform_id: str, transform_fn: Callable[['QuantumTensor'], 'QuantumTensor']) -> None:
        """Register a custom feature transformation function for N-dimensional tensors."""
        self.transform_registry[transform_id] = transform_fn
    
    def extract_raw_features(self, qstate: QuantumState, 
                            target_shape: Optional[Tuple[int, ...]] = None) -> 'QuantumTensor':
        """Extract raw features from a quantum state, supporting N-dimensional tensors."""
        if qstate.metadata.get("is_compressed", False):
            qstate.decompress()
        
        # Extract features as a QuantumTensor
        features = QuantumTensor(qstate.state, qstate.shape)
        
        # Reshape to target shape if provided, handling multi-dimensional cases
        if target_shape:
            current_size = len(features.data)
            target_size = 1
            for dim in target_shape:
                target_size *= dim
            
            flat_features = features.data
            if current_size < target_size:
                # Pad with zeros to match target size
                padded = flat_features + [Decimal('0')] * (target_size - current_size)
            elif current_size > target_size:
                # Trim excess elements
                padded = flat_features[:target_size]
            else:
                padded = flat_features
            
            features = QuantumTensor(padded, target_shape)
        
        return features
    
    def normalize_features(self, features: 'QuantumTensor') -> 'QuantumTensor':
        """Normalize N-dimensional features to unit norm across all elements using Decimal."""
        # Compute norm as the square root of the sum of squares
        squared_sum = sum(x * x for x in features.data)
        norm = PRECISION_CONFIG.sqrt_decimal(squared_sum)
        if norm <= PRECISION_CONFIG.MIN_VALUE:
            return features
        normalized_data = [x / norm for x in features.data]
        return QuantumTensor(normalized_data, features.shape)
    
    def _lyapunov_stabilize(self, features: 'QuantumTensor', dt: Decimal) -> 'QuantumTensor':
        """Apply Lyapunov stabilization to features."""
        data = [x + dt * (Decimal('1') - x) for x in features.data]
        return QuantumTensor(data, features.shape)
    
    async def apply_transform(self, features: 'QuantumTensor', transform_id: str) -> 'QuantumTensor':
        """Apply a registered transformation to N-dimensional features with LLM augmentation."""
        if transform_id not in self.transform_registry:
            raise ValueError(f"Transform {transform_id} not registered")
        
        # Apply the transformation
        transformed = self.transform_registry[transform_id](features)
        
        # Augment with LLM for optimization
        features_str = "".join([chr(int(x.to_integral_value() % 128)) for x in transformed.data])
        prompt = f"Optimize this feature embedding for ML: {features_str}"
        llm_response = await self.llm_fusion.quantum_augmented_response(prompt)
        # For simplicity, use LLM response length as a scaling factor
        scaling_factor = Decimal(len(llm_response)) / Decimal('1000')
        
        # Adjust features based on LLM feedback
        adjusted_data = [x * scaling_factor for x in transformed.data]
        return QuantumTensor(adjusted_data, transformed.shape)
    
    async def map_features(self, qstate: QuantumState, 
                          target_shape: Optional[Tuple[int, ...]] = None, 
                          transform_id: Optional[str] = None) -> 'QuantumTensor':
        """Full feature mapping pipeline for N-dimensional tensors."""
        # Extract raw features
        raw_features = self.extract_raw_features(qstate, target_shape)
        
        # Normalize across all dimensions
        norm_features = self.normalize_features(raw_features)
        
        # Apply Lyapunov stabilization
        norm_features = self._lyapunov_stabilize(norm_features, Decimal('0.01'))
        
        # Apply custom transform if specified
        if transform_id:
            norm_features = await self.apply_transform(norm_features, transform_id)
        
        return norm_features

# Test cases
async def run_tests():
    mapper = FeatureMapper()
    
    # Test state: 3D tensor with mixed base values
    initial_data = [
        Decimal('1E10') + Decimal('1E-20'), Decimal('4294967296') + Decimal('1E-40'),  # 10^10 + 10^-20, 2^32 + 2^-40
        Decimal('1E8') + Decimal('1E-15'), Decimal('65536') + Decimal('1E-20'),        # 10^8 + 10^-15, 2^16 + 2^-20
        Decimal('1E6'), Decimal('1024'),                                              # 10^6, 2^10
        Decimal('1E4'), Decimal('0')                                                  # 10^4, 0
    ]
    initial_state = QuantumTensor(initial_data, (2, 2, 2))
    qstate = QuantumState(initial_state, compress_depth=3)
    
    # Register a transform (e.g., scaling across all dimensions)
    def scale_features(features: 'QuantumTensor') -> 'QuantumTensor':
        data = [x * Decimal('2') for x in features.data]
        return QuantumTensor(data, features.shape)
    
    mapper.register_transform("scale", scale_features)
    
    # Map features to various shapes
    # 1. Keep original 3D shape (2, 2, 2)
    features_3d = await mapper.map_features(qstate, transform_id="scale")
    print("Mapped 3D Features (original shape):\n", features_3d.to_nested_list())
    print("Shape:", features_3d.shape)
    
    # 2. Reshape to 2D (4, 2)
    features_2d = await mapper.map_features(qstate, target_shape=(4, 2), transform_id="scale")
    print("\nMapped 2D Features (4, 2):\n", features_2d.to_nested_list())
    print("Shape:", features_2d.shape)
    
    # 3. Reshape to 1D (8,)
    features_1d = await mapper.map_features(qstate, target_shape=(8,), transform_id="scale")
    print("\nMapped 1D Features (8,):\n", features_1d.to_nested_list())
    print("Shape:", features_1d.shape)
    
    # 4. Oversized reshape with padding (3, 3)
    features_padded = await mapper.map_features(qstate, target_shape=(3, 3), transform_id="scale")
    print("\nMapped Padded Features (3, 3):\n", features_padded.to_nested_list())
    print("Shape:", features_padded.shape)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_tests())
