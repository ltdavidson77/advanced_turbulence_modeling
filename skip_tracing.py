from decimal import Decimal, getcontext
from typing import Dict, Tuple, Any, List
from precision_controls import PrecisionControls

# Ensure high precision
getcontext().prec = 100

# Global precision config
PRECISION_CONFIG: PrecisionControls = PrecisionControls()

class SkipTracer:
    """Pure Decimal-based nested logarithmic skip tracing for hierarchical compression."""
    
    def __init__(self):
        """Initialize with hierarchical bases and tracing caches."""
        self.bases_priority: List[Decimal] = [PRECISION_CONFIG.get_base(i) for i in range(3)]  # 10, 60, 2
        self.forward_trace: Dict[bytes, List[Decimal]] = {}
        self.reverse_map: Dict[bytes, Tuple[List[Decimal], int, List[Decimal]]] = {}
    
    def _log_decimal(self, x: Decimal, base: Decimal) -> Decimal:
        """Compute logarithm using Decimal's ln, no float conversion."""
        if x <= PRECISION_CONFIG.MIN_VALUE:
            return Decimal('0.0')
        return x.ln() / base.ln()
    
    def _exp_decimal(self, x: Decimal, base: Decimal) -> Decimal:
        """Compute exponentiation using Decimal."""
        return base ** x
    
    def _nested_log_step(self, x: Decimal) -> Tuple[Decimal, List[Decimal]]:
        """Apply nested log compression across all bases, returning combined log and remainders."""
        if x <= PRECISION_CONFIG.MIN_VALUE:
            return Decimal('0.0'), [PRECISION_CONFIG.MIN_VALUE] * len(self.bases_priority)
        
        combined_log = Decimal('0.0')
        remainders = []
        current = x
        
        for base in self.bases_priority:
            log_val = self._log_decimal(current, base)
            int_part = Decimal(str(int(log_val.to_integral_value())))
            combined_log += log_val
            exp_part = self._exp_decimal(int_part, base)
            remainder = current - exp_part
            if remainder <= PRECISION_CONFIG.MIN_VALUE:
                remainder = PRECISION_CONFIG.MIN_VALUE
            remainders.append(remainder)
            current = remainder
        
        return combined_log, remainders
    
    def forward_trace(self, tensor: List[Decimal], depth: int) -> List[Decimal]:
        """Perform nested logarithmic compression with skip tracing."""
        key = (bytes(str(tensor), 'utf-8'), depth)
        if key in self.forward_trace:
            return self.forward_trace[key]
        
        if depth <= 0:
            result = tensor
        else:
            # Apply nested logs to each element
            log_tensor = []
            for x in tensor:
                combined_log, _ = self._nested_log_step(x)
                log_tensor.append(combined_log)
            
            # Recurse on the combined logs
            result = self.forward_trace(log_tensor, depth - 1)
        
        self.forward_trace[key] = result
        self.reverse_map[bytes(str(result), 'utf-8')] = (tensor, depth, self.bases_priority)
        return result
    
    def reverse_trace(self, compressed: List[Decimal]) -> List[Decimal]:
        """Reverse nested logarithmic compression with skip tracing."""
        key = bytes(str(compressed), 'utf-8')
        if key not in self.reverse_map:
            raise ValueError("No reverse trace available for this data")
        
        original, depth, bases = self.reverse_map[key]
        if depth <= 0:
            return original
        
        # Start with compressed data
        current = compressed
        
        for _ in range(depth):
            recovered = []
            for log_val in current:
                temp = Decimal('0.0')
                # Reverse bases in opposite order
                for base in reversed(bases):
                    temp = self._exp_decimal(log_val, base) + temp
                recovered.append(temp)
            # Recurse on recovered values
            current = self.forward_trace(recovered, depth - 1)
        
        return recovered
    
    def clear_cache(self) -> None:
        """Clear tracing caches."""
        self.forward_trace.clear()
        self.reverse_map.clear()

if __name__ == "__main__":
    # Test the skip tracer
    tracer = SkipTracer()
    
    # Test data: 3D tensor with mixed base values, all as Decimal
    test_tensor = [
        [[Decimal('10') ** 10 + Decimal('10') ** -20, Decimal('2') ** 32 + Decimal('2') ** -40],
         [Decimal('10') ** 8 + Decimal('10') ** -15, Decimal('2') ** 16 + Decimal('2') ** -20]],
        [[Decimal('10') ** 6, Decimal('2') ** 10],
         [Decimal('10') ** 4, Decimal('0.0')]]
    ]
    flat_tensor = [x for sublist in test_tensor for subsublist in sublist for x in subsublist]
    
    # Forward trace (compress)
    compressed = tracer.forward_trace(flat_tensor, depth=3)
    print("Compressed Tensor (flat):", compressed)
    print("Length:", len(compressed))
    
    # Reverse trace (decompress)
    recovered = tracer.reverse_trace(compressed)
    print("\nRecovered Tensor (flat):", recovered)
    print("Length:", len(recovered))
    
    # Calculate drift with Decimal
    drift = sum(abs(o - r) for o, r in zip(flat_tensor, recovered)) / Decimal(str(len(flat_tensor)))
    print("\nMean Absolute Drift (Decimal):", drift)
