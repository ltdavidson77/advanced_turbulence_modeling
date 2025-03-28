from decimal import Decimal, getcontext
from typing import List, Tuple
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine
from quantum_state import QuantumState
from enterprise_framework import QuantumEnterpriseFramework
from quantum_ml_pipeline import QuantumMLPipeline
from data_preprocessor import DataPreprocessor
from feature_mapper import FeatureMapper
from cuda_kernels import CudaCompressionKernels
from hardware_interface import HardwareInterface
from api_client import APIClient
from benchmark_suite import BenchmarkSuite
from skip_tracing import SkipTracer
from quantum_tensor import QuantumTensor
from logarithmic_processor import LogarithmicProcessor

# Ensure high precision
getcontext().prec = 100

# Initialize global instances
PRECISION_CONFIG: PrecisionControls = PrecisionControls()
COMPRESSION_ENGINE: SpatialLogCompressionEngine = SpatialLogCompressionEngine()
FEATURE_MAPPER: FeatureMapper = FeatureMapper()
CUDA_KERNELS: CudaCompressionKernels = CudaCompressionKernels()
FRAMEWORK: QuantumEnterpriseFramework = QuantumEnterpriseFramework(compress_depth=3)
PIPELINE: QuantumMLPipeline = QuantumMLPipeline(compress_depth=3, framework=FRAMEWORK)
PREPROCESSOR: DataPreprocessor = DataPreprocessor()
HARDWARE: HardwareInterface = HardwareInterface(use_gpu=True)
API_CLIENT: APIClient = APIClient(api_endpoint="http://example.com/api", api_key="test_key")
BENCHMARK: BenchmarkSuite = BenchmarkSuite()
TRACER: SkipTracer = SkipTracer()
LOG_PROCESSOR: LogarithmicProcessor = LogarithmicProcessor()

def main():
    # Test data: 3D tensor with mixed base values
    raw_data = [
        10**10 + 10**-20, 2**32 + 2**-40,
        10**8 + 10**-15, 2**16 + 2**-20,
        10**6, 2**10,
        10**4, 0
    ]
    shape = (2, 2, 2)
    
    # Step 1: Pre-process with LogarithmicProcessor
    print("Pre-processing data...")
    tensor = LOG_PROCESSOR.process_and_compress(raw_data, shape, depth=3)
    print("Pre-processed and Compressed Tensor:", tensor.to_nested_list())
    
    # Step 2: Create QuantumState
    print("\nCreating QuantumState...")
    qstate = QuantumState(tensor.data, compress_depth=3)
    qstate.compress()
    print("Compressed QuantumState Size (bytes):", qstate.compressed_size)
    
    # Step 3: Run ML Pipeline
    print("\nRunning ML Pipeline...")
    target_shape = (4, 2)
    FEATURE_MAPPER.register_transform("scale", lambda features: [x * Decimal('2.0') for x in features])
    pipeline_result = PIPELINE.run_pipeline(tensor.data, "main_test", target_shape=target_shape, transform_id="scale")
    print("Pipeline Features:", pipeline_result.get("features"))
    
    # Step 4: Send to API
    print("\nSending to API...")
    api_result = API_CLIENT.bridge_to_ai(tensor.data, "main_test", target_shape=target_shape, transform_id="scale")
    print("API Response:", api_result["api_response"])
    
    # Step 5: Benchmark
    print("\nRunning Benchmarks...")
    benchmark_result = BENCHMARK.run_benchmarks(shape=shape, depth=3)
    print("Benchmark Results:", benchmark_result)

if __name__ == "__main__":
    main()
