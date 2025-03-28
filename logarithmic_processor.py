from decimal import Decimal, getcontext
from typing import List, Tuple, Any, Optional
from precision_controls import PrecisionControls
from skip_tracing import SkipTracer
from quantum_tensor import QuantumTensor

# Ensure high precision
getcontext().prec = 100

# Global instances
PRECISION_CONFIG: PrecisionControls = PrecisionControls()
TRACER: SkipTracer = SkipTracer()

class LogarithmicProcessor:
    """Pre-processing and logarithmic computation module with pure Decimal precision."""
    
    def __init__(self):
        """Initialize with hierarchical bases."""
        self.bases_priority: List[Decimal] = [PRECISION_CONFIG.get_base(i) for i in range(3)]  # 10, 60, 2
    
    def _log_decimal(self, x: Decimal, base: Decimal) -> Decimal:
        """Compute logarithm using Decimal's ln, no float conversion."""
        if x <= PRECISION_CONFIG.MIN_VALUE:
            return Decimal('0.0')
        return x.ln() / base.ln()
    
    def preprocess_data(self, data: List[Any], shape: Tuple[int, ...]) -> 'QuantumTensor':
        """Pre-process raw data into a QuantumTensor with Decimal precision."""
        # Convert all inputs to Decimal
        decimal_data = [Decimal(str(x)) if x != 0 else PRECISION_CONFIG.MIN_VALUE for x in data]
        
        # Handle negatives by shifting (per precision_controls logic)
        min_val = min(decimal_data)
        if min_val < Decimal('0.0'):
            decimal_data = [x - min_val + PRECISION_CONFIG.MIN_VALUE for x in decimal_data]
        
        # Create QuantumTensor
        return QuantumTensor(decimal_data, shape)
    
    def apply_log_transform(self, tensor: 'QuantumTensor', base_index: int = 0) -> 'QuantumTensor':
        """Apply a single logarithmic transformation to the tensor."""
        if base_index not in range(len(self.bases_priority)):
            raise ValueError(f"Base index {base_index} out of range")
        
        base = self.bases_priority[base_index]
        transformed_data = [self._log_decimal(x, base) for x in tensor.flatten()]
        return QuantumTensor(transformed_data, tensor.shape)
    
    def nested_log_transform(self, tensor: 'QuantumTensor') -> 'QuantumTensor':
        """Apply nested logarithmic transformation across all bases."""
        flat_data = tensor.flatten()
        transformed_data = []
        
        for x in flat_data:
            combined_log = Decimal('0.0')
            current = x
            for base in self.bases_priority:
                log_val = self._log_decimal(current, base)
                int_part = Decimal(str(int(log_val.to_integral_value())))
                combined_log += log_val
                remainder = current - base ** int_part
                if remainder <= PRECISION_CONFIG.MIN_VALUE:
                    remainder = PRECISION_CONFIG.MIN_VALUE
                current = remainder
            transformed_data.append(combined_log)
        
        return QuantumTensor(transformed_data, tensor.shape)
    
    def process_and_compress(self, data: List[Any], shape: Tuple[int, ...], depth: int) -> 'QuantumTensor':
        """Full pipeline: preprocess, apply nested logs, and compress."""
        # Preprocess into QuantumTensor
        tensor = self.preprocess_data(data, shape)
        
        # Apply nested log transform
        log_tensor = self.nested_log_transform(tensor)
        
        # Compress via SkipTracer
        compressed_data = TRACER.forward_trace(log_tensor.flatten(), depth)
        return QuantumTensor(compressed_data, shape)

if __name__ == "__main__":
    # Test the logarithmic processor
    processor = LogarithmicProcessor()
    
    # Test data: 3D tensor with mixed base values, as raw input
    raw_data = [
        10**10 + 10**-20, 2**32 + 2**-40,
        10**8 + 10**-15, 2**16 + 2**-20,
        10**6, 2**10,
        10**4, 0
    ]
    shape = (2, 2, 2)
    
    # Preprocess
    tensor = processor.preprocess_data(raw_data, shape)
    print("Preprocessed Tensor (nested):", tensor.to_nested_list())
    
    # Apply single log transform (base-10)
    log_tensor = processor.apply_log_transform(tensor, base_index=0)
    print("\nBase-10 Log Tensor (flat):", log_tensor.data)
    
    # Apply nested log transform
    nested_tensor = processor.nested_log_transform(tensor)
    print("\nNested Log Tensor (flat):", nested_tensor.data)
    
    # Process and compress
    compressed_tensor = processor.process_and_compress(raw_data, shape, depth=3)
    print("\nCompressed Tensor (flat):", compressed_tensor.data)
    
    # Decompress to verify
    decompressed_tensor = tensor.decompress(compressed_tensor.data)
    print("\nDecompressed Tensor (nested):", decompressed_tensor.to_nested_list())
    
    # Calculate drift with Decimal
    original_flat = tensor.data
    recovered_flat = decompressed_tensor.data
    drift = sum(abs(o - r) for o, r in zip(original_flat, recovered_flat)) / Decimal(str(len(original_flat)))
    print("\nMean Absolute Drift (Decimal):", drift)
