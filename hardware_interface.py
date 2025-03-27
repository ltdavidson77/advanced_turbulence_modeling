import numpy as np
from typing import Optional, Tuple
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine
from quantum_state import QuantumState
from feature_mapper import FeatureMapper
from cuda_kernels import CudaCompressionKernels

# Global instances
PRECISION_CONFIG = PrecisionControls()
COMPRESSION_ENGINE = SpatialLogCompressionEngine()
FEATURE_MAPPER = FeatureMapper()
CUDA_KERNELS = CudaCompressionKernels()

class HardwareInterface:
    """Abstracts hardware backends for compression and feature mapping operations."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the hardware interface.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available.
        """
        self.use_gpu = use_gpu and cp.cuda.is_available()
        if self.use_gpu:
            print("Using GPU acceleration via CUDA")
        else:
            print("Using CPU backend")
    
    def compress(self, qstate: QuantumState, depth: int) -> None:
        """Compress a quantum state using the appropriate hardware backend."""
        if self.use_gpu:
            CUDA_KERNELS.accelerate_compression(qstate, depth)
        else:
            COMPRESSION_ENGINE.forward_compress(qstate.state, depth)
            qstate.state = COMPRESSION_ENGINE.forward_compress(qstate.state, depth)
            qstate.metadata["is_compressed"] = True
    
    def decompress(self, qstate: QuantumState, depth: int) -> None:
        """Decompress a quantum state using the appropriate hardware backend."""
        if self.use_gpu:
            CUDA_KERNELS.accelerate_decompression(qstate, depth)
        else:
            qstate.state = COMPRESSION_ENGINE.reverse_decompress(qstate.state)
            qstate.state = PRECISION_CONFIG.denormalize_array(qstate.state)
            qstate.metadata["is_compressed"] = False
    
    def map_features(self, qstate: QuantumState, 
                    target_shape: Optional[Tuple[int, ...]] = None, 
                    transform_id: Optional[str] = None) -> np.ndarray:
        """Map features from a quantum state using the appropriate hardware backend."""
        if self.use_gpu:
            return CUDA_KERNELS.accelerate_feature_mapping(qstate, target_shape, transform_id)
        else:
            return FEATURE_MAPPER.map_features(qstate, target_shape, transform_id)
    
    def execute_operation(self, operation: str, qstate: QuantumState, 
                         depth: Optional[int] = None, 
                         target_shape: Optional[Tuple[int, ...]] = None, 
                         transform_id: Optional[str] = None) -> Optional[np.ndarray]:
        """Execute a specified operation on the hardware backend."""
        if operation == "compress":
            if depth is None:
                raise ValueError("Depth required for compression")
            self.compress(qstate, depth)
            return None
        elif operation == "decompress":
            if depth is None:
                raise ValueError("Depth required for decompression")
            self.decompress(qstate, depth)
            return None
        elif operation == "map_features":
            return self.map_features(qstate, target_shape, transform_id)
        else:
            raise ValueError(f"Unknown operation: {operation}")

if __name__ == "__main__":
    # Test the hardware interface
    interface = HardwareInterface(use_gpu=True)  # Toggle True/False to test GPU/CPU
    
    # Test state: 3D tensor
    test_tensor = np.array([
        [[float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
         [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]],
        [[float(Decimal('10') ** 6), float(Decimal('2') ** 10)],
         [float(Decimal('10') ** 4), 0.0]]
    ])
    qstate = QuantumState(test_tensor, compress_depth=3)
    
    # Register a transform for feature mapping
    def scale_features(features: np.ndarray) -> np.ndarray:
        return features * 2.0
    FEATURE_MAPPER.register_transform("scale", scale_features)
    
    # Test compression
    interface.execute_operation("compress", qstate, depth=3)
    print("Compressed Size (bytes):", qstate.compressed_size)
    
    # Test feature mapping
    features = interface.execute_operation("map_features", qstate, target_shape=(4, 2), transform_id="scale")
    print("Mapped Features (4, 2):\n", features)
    print("Shape:", features.shape)
    
    # Test decompression
    interface.execute_operation("decompress", qstate, depth=3)
    print("Decompressed State Sample:\n", np.vectorize(float)(qstate.state[0]))
