import numpy as np
import time
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine
from quantum_state import QuantumState
from enterprise_framework import QuantumEnterpriseFramework
from quantum_ml_pipeline import QuantumMLPipeline
from feature_mapper import FeatureMapper
from hardware_interface import HardwareInterface
from cuda_kernels import CudaCompressionKernels

# Global instances
PRECISION_CONFIG: PrecisionControls = PrecisionControls()
COMPRESSION_ENGINE: SpatialLogCompressionEngine = SpatialLogCompressionEngine()
FEATURE_MAPPER: FeatureMapper = FeatureMapper()
CUDA_KERNELS: CudaCompressionKernels = CudaCompressionKernels()
FRAMEWORK: QuantumEnterpriseFramework = QuantumEnterpriseFramework(compress_depth=3)
PIPELINE: QuantumMLPipeline = QuantumMLPipeline(compress_depth=3, framework=FRAMEWORK)
HARDWARE_CPU: HardwareInterface = HardwareInterface(use_gpu=False)
HARDWARE_GPU: HardwareInterface = HardwareInterface(use_gpu=True)

class BenchmarkSuite:
    """Benchmarks performance and precision of the quantum-classical framework."""
    
    def __init__(self):
        """Initialize the benchmark suite with test configurations."""
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def generate_test_data(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate multi-dimensional test data with mixed base values."""
        data = np.zeros(shape, dtype=np.float64)
        indices = np.ndindex(shape)
        for idx in indices:
            if np.random.rand() > 0.5:
                data[idx] = float(Decimal('10') ** np.random.randint(4, 11) + 
                                Decimal('10') ** -np.random.randint(15, 21))
            else:
                data[idx] = float(Decimal('2') ** np.random.randint(10, 33) + 
                                Decimal('2') ** -np.random.randint(20, 41))
        return data
    
    def measure_time(self, func: callable, *args, **kwargs) -> float:
        """Measure execution time of a function."""
        start = time.time()
        func(*args, **kwargs)
        return time.time() - start
    
    def calculate_drift(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate mean absolute drift between original and processed data."""
        return float(np.mean(np.abs(original - processed)))
    
    def benchmark_compression(self, data: np.ndarray, depth: int, state_id: str) -> Dict[str, Any]:
        """Benchmark compression performance and precision."""
        qstate_cpu = QuantumState(data.copy(), compress_depth=depth)
        qstate_gpu = QuantumState(data.copy(), compress_depth=depth)
        
        # CPU compression
        cpu_time = self.measure_time(HARDWARE_CPU.compress, qstate_cpu, depth)
        cpu_size = qstate_cpu.compressed_size
        
        # GPU compression
        gpu_time = self.measure_time(HARDWARE_GPU.compress, qstate_gpu, depth)
        gpu_size = qstate_gpu.compressed_size
        
        # Decompress to check precision
        HARDWARE_CPU.decompress(qstate_cpu, depth)
        HARDWARE_GPU.decompress(qstate_gpu, depth)
        
        cpu_drift = self.calculate_drift(np.vectorize(float)(data), np.vectorize(float)(qstate_cpu.state))
        gpu_drift = self.calculate_drift(np.vectorize(float)(data), np.vectorize(float)(qstate_gpu.state))
        
        return {
            "cpu_time_s": cpu_time,
            "gpu_time_s": gpu_time,
            "cpu_compressed_size_bytes": cpu_size,
            "gpu_compressed_size_bytes": gpu_size,
            "cpu_drift": cpu_drift,
            "gpu_drift": gpu_drift
        }
    
    def benchmark_feature_mapping(self, data: np.ndarray, depth: int, state_id: str, 
                                target_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark feature mapping performance."""
        qstate_cpu = QuantumState(data.copy(), compress_depth=depth)
        qstate_gpu = QuantumState(data.copy(), compress_depth=depth)
        
        # Register a transform
        def scale_features(features: np.ndarray) -> np.ndarray:
            return features * 2.0
        FEATURE_MAPPER.register_transform("scale", scale_features)
        
        # CPU feature mapping
        cpu_time = self.measure_time(
            HARDWARE_CPU.map_features, qstate_cpu, target_shape, "scale"
        )
        cpu_features = HARDWARE_CPU.map_features(qstate_cpu, target_shape, "scale")
        
        # GPU feature mapping
        gpu_time = self.measure_time(
            HARDWARE_GPU.map_features, qstate_gpu, target_shape, "scale"
        )
        gpu_features = HARDWARE_GPU.map_features(qstate_gpu, target_shape, "scale")
        
        return {
            "cpu_time_s": cpu_time,
            "gpu_time_s": gpu_time,
            "cpu_features_shape": cpu_features.shape,
            "gpu_features_shape": gpu_features.shape
        }
    
    def run_benchmarks(self, shape: Tuple[int, ...] = (2, 2, 2), depth: int = 3) -> Dict[str, Any]:
        """Run full benchmark suite."""
        test_data = self.generate_test_data(shape)
        state_id = "benchmark_test"
        
        # Compression benchmark
        compression_results = self.benchmark_compression(test_data, depth, state_id)
        
        # Feature mapping benchmark
        target_shape = (4, 2)
        feature_results = self.benchmark_feature_mapping(test_data, depth, state_id, target_shape)
        
        self.results = {
            "compression": compression_results,
            "feature_mapping": feature_results,
            "data_shape": shape
        }
        return self.results

if __name__ == "__main__":
    # Run the benchmark suite
    suite = BenchmarkSuite()
    
    # Test with a 3D tensor
    results = suite.run_benchmarks(shape=(2, 2, 2), depth=3)
    
    print("Benchmark Results:")
    print("Data Shape:", results["data_shape"])
    print("\nCompression:")
    for key, value in results["compression"].items():
        print(f"  {key}: {value}")
    print("\nFeature Mapping:")
    for key, value in results["feature_mapping"].items():
        print(f"  {key}: {value}")
