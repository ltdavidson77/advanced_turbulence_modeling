from decimal import Decimal
from typing import Dict, Any, Optional, Callable, List, Tuple
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine, CompressionSimulation
from quantum_state import QuantumState
from quantum_tensor import QuantumTensor
from feature_mapper import FeatureMapper
from api_integration import QuantumEnterpriseFramework

# Global instances
PRECISION_CONFIG: PrecisionControls = PrecisionControls()
COMPRESSION_ENGINE: SpatialLogCompressionEngine = SpatialLogCompressionEngine()
COMPRESSION_SIM: CompressionSimulation = CompressionSimulation()
FEATURE_MAPPER: FeatureMapper = FeatureMapper()

class QuantumMLPipeline:
    """Machine learning pipeline using compressed quantum states with classical emulation."""
    
    def __init__(self, compress_depth: int = 3, framework: Optional[QuantumEnterpriseFramework] = None):
        """
        Initialize the ML pipeline.
        
        Args:
            compress_depth (int): Depth for logarithmic compression.
            framework (QuantumEnterpriseFramework, optional): External framework instance.
        """
        self.compress_depth = compress_depth
        self.framework = framework if framework else QuantumEnterpriseFramework()
        self.feature_transforms: Dict[str, Callable[['QuantumTensor'], 'QuantumTensor']] = {}
        self.state_registry: Dict[str, QuantumState] = {}
    
    def register_feature_transform(self, transform_id: str, transform_fn: Callable[['QuantumTensor'], 'QuantumTensor']) -> None:
        """Register a custom feature transformation function."""
        self.feature_transforms[transform_id] = transform_fn
    
    async def preprocess_data(self, data: 'QuantumTensor', state_id: str) -> None:
        """Preprocess input data into a compressed quantum state."""
        # Register the state
        qstate = QuantumState(data, compress_depth=self.compress_depth)
        self.state_registry[state_id] = qstate
        
        # Compress the state using SpatialLogCompressionEngine
        compressed_tensor = await COMPRESSION_SIM.framework_compute(data, self.compress_depth)
        self.state_registry[state_id].state = compressed_tensor.data
        self.state_registry[state_id].metadata["is_compressed"] = True
    
    def _get_indices(self, shape: Tuple[int, ...], flat_index: int) -> Tuple[int, ...]:
        """Convert a flat index to multi-dimensional indices based on shape."""
        indices = []
        temp = flat_index
        for dim in reversed(shape[1:]):  # Exclude the first dimension
            indices.insert(0, temp % dim)
            temp //= dim
        indices.insert(0, temp)  # First dimension
        return tuple(indices)
    
    def _get_flat_index(self, indices: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
        """Convert multi-dimensional indices to a flat index."""
        flat_index = 0
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            flat_index += indices[i] * stride
            stride *= shape[i]
        return flat_index
    
    async def apply_classical_transform(self, state_id: str, transform_matrix: Optional['QuantumTensor'] = None) -> None:
        """Apply a classical transformation to the compressed state, supporting N-D tensors."""
        if state_id not in self.state_registry:
            raise ValueError(f"State {state_id} not registered")
        
        qstate = self.state_registry[state_id]
        if qstate.metadata.get("is_compressed", False):
            qstate.decompress()
        
        # Apply a classical transformation (e.g., matrix multiplication for N-D tensors)
        if transform_matrix:
            # Flatten the state tensor for transformation
            state_data = qstate.state
            state_shape = qstate.shape
            
            # Determine the matrix dimensions
            # For simplicity, assume transform_matrix is a square matrix applied to the last dimension
            if len(transform_matrix.shape) != 2 or transform_matrix.shape[0] != transform_matrix.shape[1]:
                raise ValueError("Transform matrix must be square")
            matrix_size = transform_matrix.shape[0]
            
            # The last dimension of the state must match the matrix size
            if state_shape[-1] != matrix_size:
                raise ValueError(f"Last dimension of state {state_shape[-1]} must match matrix size {matrix_size}")
            
            # Compute the output shape (replace the last dimension with matrix_size)
            output_shape = state_shape[:-1] + (matrix_size,)
            output_size = 1
            for dim in output_shape:
                output_size *= dim
            
            # Perform the transformation
            transformed_data = [Decimal('0')] * output_size
            for i in range(output_size):
                state_indices = self._get_indices(output_shape, i)
                state_last_idx = state_indices[-1]  # Index in the last dimension
                for j in range(matrix_size):
                    # Compute the corresponding state index with the same prefix but j in the last dimension
                    state_idx_tuple = state_indices[:-1] + (j,)
                    state_idx = self._get_flat_index(state_idx_tuple, state_shape)
                    matrix_idx = state_last_idx * matrix_size + j
                    transformed_data[i] += state_data[state_idx] * transform_matrix.data[matrix_idx]
            
            qstate.state = transformed_data
            qstate.shape = output_shape
        
        # Re-compress after transformation
        compressed_tensor = await COMPRESSION_SIM.framework_compute(
            QuantumTensor(qstate.state, qstate.shape), self.compress_depth
        )
        qstate.state = compressed_tensor.data
        qstate.metadata["is_compressed"] = True
    
    def _lyapunov_stabilize(self, tensor: 'QuantumTensor', dt: Decimal) -> 'QuantumTensor':
        """Apply Lyapunov stabilization to tensor data."""
        data = [x + dt * (Decimal('1') - x) for x in tensor.data]
        return QuantumTensor(data, tensor.shape)
    
    async def extract_features(self, state_id: str, target_shape: Tuple[int, ...], 
                              transform_id: Optional[str] = None) -> 'QuantumTensor':
        """Extract features from the quantum state for ML."""
        if state_id not in self.state_registry:
            raise ValueError(f"State {state_id} not registered")
        
        qstate = self.state_registry[state_id]
        
        # Extract features using FeatureMapper
        features = await FEATURE_MAPPER.map_features(qstate, target_shape, transform_id)
        
        # Apply Lyapunov stabilization
        features = self._lyapunov_stabilize(features, Decimal('0.01'))
        
        # Augment with LLM via QuantumEnterpriseFramework
        features_str = "".join([chr(int(x.to_integral_value() % 128)) for x in features.data])
        prompt = f"Optimize these ML features: {features_str}"
        llm_response = await self.framework.execute_quantum_workflow(prompt)
        # For simplicity, use LLM response length as a scaling factor
        scaling_factor = Decimal(len(llm_response)) / Decimal('1000')
        
        adjusted_data = [x * scaling_factor for x in features.data]
        return QuantumTensor(adjusted_data, features.shape)
    
    async def run_pipeline(self, data: 'QuantumTensor', state_id: str, 
                          transform_matrix: Optional['QuantumTensor'] = None, 
                          target_shape: Optional[Tuple[int, ...]] = None, 
                          transform_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the full ML pipeline: preprocess, transform, extract."""
        # Preprocess and compress
        await self.preprocess_data(data, state_id)
        
        # Apply classical transform
        await self.apply_classical_transform(state_id, transform_matrix)
        
        # Extract features
        result = {
            "compressed_size_bytes": sum(len(str(x)) for x in self.state_registry[state_id].state),
            "measurement_results": {"state_id": state_id, "status": "processed"}
        }
        
        if target_shape:
            result["features"] = await self.extract_features(state_id, target_shape, transform_id)
        
        return result

# Test cases
async def run_tests():
    pipeline = QuantumMLPipeline(compress_depth=3)
    
    # Define a simple feature transform (e.g., scaling)
    def scale_features(features: 'QuantumTensor') -> 'QuantumTensor':
        data = [x * Decimal('2') for x in features.data]
        return QuantumTensor(data, features.shape)
    
    pipeline.register_feature_transform("scale", scale_features)
    
    # Test 1: 2D tensor
    print("\n=== Testing 2D Tensor ===")
    input_data_2d = [
        Decimal('1E10') + Decimal('1E-20'), Decimal('4294967296') + Decimal('1E-40'),
        Decimal('1E8') + Decimal('1E-15'), Decimal('65536') + Decimal('1E-20')
    ]
    input_tensor_2d = QuantumTensor(input_data_2d, (2, 2))
    
    # Transform matrix for 2D (2x2 matrix applied to last dimension)
    transform_data_2d = [
        Decimal('1'), Decimal('0'),
        Decimal('0'), Decimal('1')
    ]
    transform_matrix_2d = QuantumTensor(transform_data_2d, (2, 2))
    
    result_2d = await pipeline.run_pipeline(
        data=input_tensor_2d,
        state_id="ml_test_2d",
        transform_matrix=transform_matrix_2d,
        target_shape=(2, 2),
        transform_id="scale"
    )
    print("Compressed Size (bytes):", result_2d["compressed_size_bytes"])
    print("Measurement Results:", result_2d["measurement_results"])
    print("Extracted Features:\n", result_2d["features"].to_nested_list())
    
    # Test 2: 3D tensor
    print("\n=== Testing 3D Tensor ===")
    input_data_3d = [
        Decimal('1E10'), Decimal('4294967296'), Decimal('1E8'), Decimal('65536'),
        Decimal('1E6'), Decimal('1024'), Decimal('1E4'), Decimal('0')
    ]
    input_tensor_3d = QuantumTensor(input_data_3d, (2, 2, 2))
    
    # Transform matrix for 3D (2x2 matrix applied to last dimension)
    transform_matrix_3d = QuantumTensor(transform_data_2d, (2, 2))  # Same 2x2 matrix
    
    result_3d = await pipeline.run_pipeline(
        data=input_tensor_3d,
        state_id="ml_test_3d",
        transform_matrix=transform_matrix_3d,
        target_shape=(2, 2, 2),
        transform_id="scale"
    )
    print("Compressed Size (bytes):", result_3d["compressed_size_bytes"])
    print("Measurement Results:", result_3d["measurement_results"])
    print("Extracted Features:\n", result_3d["features"].to_nested_list())
    
    # Test 3: 4D tensor
    print("\n=== Testing 4D Tensor ===")
    input_data_4d = [
        Decimal('1E10'), Decimal('4294967296'), Decimal('1E8'), Decimal('65536'),
        Decimal('1E6'), Decimal('1024'), Decimal('1E4'), Decimal('0'),
        Decimal('1E9'), Decimal('2147483648'), Decimal('1E7'), Decimal('32768'),
        Decimal('1E5'), Decimal('512'), Decimal('1E3'), Decimal('1')
    ]
    input_tensor_4d = QuantumTensor(input_data_4d, (2, 2, 2, 2))
    
    # Transform matrix for 4D (2x2 matrix applied to last dimension)
    transform_matrix_4d = QuantumTensor(transform_data_2d, (2, 2))  # Same 2x2 matrix
    
    result_4d = await pipeline.run_pipeline(
        data=input_tensor_4d,
        state_id="ml_test_4d",
        transform_matrix=transform_matrix_4d,
        target_shape=(2, 2, 2, 2),
        transform_id="scale"
    )
    print("Compressed Size (bytes):", result_4d["compressed_size_bytes"])
    print("Measurement Results:", result_4d["measurement_results"])
    print("Extracted Features:\n", result_4d["features"].to_nested_list())

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_tests())
