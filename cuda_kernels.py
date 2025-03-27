import numpy as np
import cupy as cp
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine
from quantum_state import QuantumState
from feature_mapper import FeatureMapper
from typing import Optional, Tuple

# Global instances
PRECISION_CONFIG = PrecisionControls()
COMPRESSION_ENGINE = SpatialLogCompressionEngine()
FEATURE_MAPPER = FeatureMapper()

class CudaCompressionKernels:
 """GPU-accelerated kernels optimizing framework compression and feature mapping."""
 
 def __init__(self):
 """Initialize with framework components and CUDA setup."""
 self.bases_priority = [float(PRECISION_CONFIG.get_base(i)) for i in range(3)] # 10, 60, 2
 
 def to_gpu(self, data: np.ndarray) -> cp.ndarray:
 """Transfer NumPy array to GPU."""
 return cp.asarray(data)
 
 def to_cpu(self, data: cp.ndarray) -> np.ndarray:
 """Transfer CuPy array back to CPU."""
 return cp.asnumpy(data)
 
 def cuda_log_apply(self, data: cp.ndarray, base: float) -> cp.ndarray:
 """Apply logarithm on GPU with base conversion."""
 return cp.log(data) / cp.log(base)
 
 def cuda_exp_apply(self, data: cp.ndarray, base: float) -> cp.ndarray:
 """Apply exponentiation on GPU with base."""
 return cp.power(base, data)
 
 def cuda_normalize(self, data: cp.ndarray) -> cp.ndarray:
 """Normalize N-dimensional data on GPU."""
 norm = cp.linalg.norm(data.flatten())
 return data / norm if norm > 0 else data
 
 def accelerate_compression(self, qstate: QuantumState, depth: int) -> None:
 """Accelerate compression in QuantumState using CUDA."""
 if not qstate.metadata["is_compressed"]:
 # Convert to GPU array
 gpu_state = self.to_gpu(np.vectorize(float)(qstate.state))
 
 # Use SpatialLogCompressionEngine logic, accelerated
 for _ in range(depth):
 for base in self.bases_priority:
 gpu_state = self.cuda_log_apply(gpu_state, base)
 gpu_state = cp.where(gpu_state < float(PRECISION_CONFIG.MIN_VALUE), 
 float(PRECISION_CONFIG.MIN_VALUE), gpu_state)
 
 # Update state
 qstate.state = PRECISION_CONFIG.denormalize_array(self.to_cpu(gpu_state))
 qstate.metadata["is_compressed"] = True
 
 def accelerate_decompression(self, qstate: QuantumState, depth: int) -> None:
 """Accelerate decompression in QuantumState using CUDA."""
 if qstate.metadata["is_compressed"]:
 gpu_state = self.to_gpu(np.vectorize(float)(qstate.state))
 
 # Reverse compression on GPU
 for _ in range(depth):
 for base in reversed(self.bases_priority):
 gpu_state = self.cuda_exp_apply(gpu_state, base)
 
 # Update state
 qstate.state = PRECISION_CONFIG.denormalize_array(self.to_cpu(gpu_state))
 qstate.metadata["is_compressed"] = False
 
 def accelerate_feature_mapping(self, qstate: QuantumState, 
 target_shape: Optional[Tuple[int, ...]] = None, 
 transform_id: Optional[str] = None) -> np.ndarray:
 """Accelerate feature mapping using CUDA."""
 # Extract raw features (decompress if needed)
 if qstate.metadata["is_compressed"]:
 self.accelerate_decompression(qstate, depth=qstate.compress_depth)
 
 gpu_features = self.to_gpu(np.vectorize(float)(qstate.state))
 
 # Normalize on GPU
 gpu_features = self.cuda_normalize(gpu_features)
 
 # Reshape if needed
 if target_shape:
 current_size = gpu_features.size
 target_size = np.prod(target_shape)
 flat_features = gpu_features.flatten()
 
 if current_size < target_size:
 padded = cp.pad(flat_features, (0, target_size - current_size), 
 mode='constant', constant_values=0.0)
 elif current_size > target_size:
 padded = flat_features[:target_size]
 else:
 padded = flat_features
 
 gpu_features = padded.reshape(target_shape)
 
 # Apply transform if specified (CPU fallback for custom logic)
 features = self.to_cpu(gpu_features)
 if transform_id and transform_id in FEATURE_MAPPER.transform_registry:
 features = FEATURE_MAPPER.apply_transform(features, transform_id)
 
 return features

if __name__ == "__main__":
 # Test the CUDA kernels
 cuda_kernels = CudaCompressionKernels()
 
 # Test state: 3D tensor
 test_tensor = np.array([
 [[float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
 [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]],
 [[float(Decimal('10') ** 6), float(Decimal('2') ** 10)],
 [float(Decimal('10') ** 4), 0.0]]
 ])
 qstate = QuantumState(test_tensor, compress_depth=3)
 
 # Register a transform in FeatureMapper for testing
 def scale_features(features: np.ndarray) -> np.ndarray:
 return features * 2.0
 FEATURE_MAPPER.register_transform("scale", scale_features)
 
 # Accelerate compression
 cuda_kernels.accelerate_compression(qstate, depth=3)
 print("Compressed Size (bytes):", qstate.compressed_size)
 
 # Accelerate feature mapping
 features = cuda_kernels.accelerate_feature_mapping(qstate, target_shape=(4, 2), transform_id="scale")
 print("Mapped Features (4, 2):\n", features)
 print("Shape:", features.shape)

