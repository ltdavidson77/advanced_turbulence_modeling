l
from decimal import Decimal, getcontext
from typing import List, Tuple, Any, Optional
from precision_controls import PrecisionControls
from skip_tracing import SkipTracer

# Ensure high precision
getcontext().prec = 100

# Global instances
PRECISION_CONFIG: PrecisionControls = PrecisionControls()
TRACER: SkipTracer = SkipTracer()

class QuantumTensor:
    """Multi-dimensional tensor abstraction for quantum-classical operations."""
    
    def __init__(self, data: List[Any], shape: Tuple[int, ...]):
        """
        Initialize the quantum tensor.
        
        Args:
            data (List[Any]): Flat list of values (int, float, Decimal) to convert to Decimal.
            shape (Tuple[int, ...]): Desired N-dimensional shape.
        """
        # Convert all inputs to Decimal
        self.data: List[Decimal] = [Decimal(str(x)) for x in data]
        self.shape: Tuple[int, ...] = shape
        self.size: int = len(self.data)
        
        if self.size != prod(self.shape):
            raise ValueError(f"Data size {self.size} does not match shape {self.shape}")
    
    def prod(self, values: Tuple[int, ...]) -> int:
        """Compute product of tuple values."""
        result = 1
        for v in values:
            result *= v
        return result
    
    def flatten(self) -> List[Decimal]:
        """Return flat list of tensor data."""
        return self.data.copy()
    
    def reshape(self, new_shape: Tuple[int, ...]) -> None:
        """Reshape the tensor in-place."""
        if self.size != prod(new_shape):
            raise ValueError(f"Cannot reshape size {self.size} into {new_shape}")
        self.shape = new_shape
    
    def compress(self, depth: int) -> 'QuantumTensor':
        """Compress the tensor using nested logarithmic skip tracing."""
        compressed_data = TRACER.forward_trace(self.data, depth)
        return QuantumTensor(compressed_data, self.shape)
    
    def decompress(self, compressed_data: List[Decimal]) -> 'QuantumTensor':
        """Decompress the tensor using skip tracing."""
        decompressed_data = TRACER.reverse_trace(compressed_data)
        if len(decompressed_data) != self.size:
            raise ValueError("Decompressed data size does not match original tensor size")
        return QuantumTensor(decompressed_data, self.shape)
    
    def apply_gate(self, gate: List[List[Decimal]]) -> None:
        """Apply a gate operation to the tensor (assumes last dimension matches gate size)."""
        if len(self.shape) < 1 or self.shape[-1] != len(gate):
            raise ValueError(f"Gate size {len(gate)} incompatible with tensor shape {self.shape}")
        
        # Flatten except last dimension
        flat_size = self.size // self.shape[-1]
        new_data = []
        
        for i in range(flat_size):
            start = i * self.shape[-1]
            vector = self.data[start:start + self.shape[-1]]
            result = [Decimal('0.0')] * len(gate[0])
            
            # Matrix-vector multiplication with Decimal
            for j in range(len(gate)):
                for k in range(len(gate[0])):
                    result[k] += gate[j][k] * vector[j]
            
            new_data.extend(result)
        
        self.data = new_data
    
    def to_nested_list(self) -> List[Any]:
        """Convert flat data back to nested list matching shape."""
        def recursive_build(data: List[Decimal], shape: Tuple[int, ...], index: List[int]) -> List[Any]:
            if len(shape) == 1:
                start = index[0] * shape[0]
                return data[start:start + shape[0]]
            result = []
            for i in range(shape[0]):
                new_index = index + [i]
                stride = prod(shape[1:])
                result.append(recursive_build(data, shape[1:], new_index))
            return result
        
        return recursive_build(self.data, self.shape, [0])

if __name__ == "__main__":
    # Test the quantum tensor
    tensor_data = [
        Decimal('10') ** 10 + Decimal('10') ** -20, Decimal('2') ** 32 + Decimal('2') ** -40,
        Decimal('10') ** 8 + Decimal('10') ** -15, Decimal('2') ** 16 + Decimal('2') ** -20,
        Decimal('10') ** 6, Decimal('2') ** 10,
        Decimal('10') ** 4, Decimal('0.0')
    ]
    shape = (2, 2, 2)
    
    # Initialize tensor
    qt = QuantumTensor(tensor_data, shape)
    print("Original Tensor (nested):", qt.to_nested_list())
    
    # Compress
    compressed_qt = qt.compress(depth=3)
    print("\nCompressed Tensor (flat):", compressed_qt.data)
    
    # Decompress
    decompressed_qt = qt.decompress(compressed_qt.data)
    print("\nDecompressed Tensor (nested):", decompressed_qt.to_nested_list())
    
    # Apply a simple gate (Hadamard-like, 2x2)
    gate = [
        [Decimal('0.70710678118654752440'), Decimal('0.70710678118654752440')],
        [Decimal('0.70710678118654752440'), Decimal('-0.70710678118654752440')]
    ]
    qt.apply_gate(gate)
    print("\nAfter Gate (nested):", qt.to_nested_list())
    
    # Calculate drift
    original_flat = tensor_data
    recovered_flat = decompressed_qt.data
    drift = sum(abs(o - r) for o, r in zip(original_flat, recovered_flat)) / Decimal(str(len(original_flat)))
    print("\nMean Absolute Drift (Decimal):", drift)
