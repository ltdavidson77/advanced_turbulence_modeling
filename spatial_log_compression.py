from decimal import Decimal, getcontext
import math
import numpy as np

# Set high precision
getcontext().prec = 100

# Factorial for scaling
def factorial(n: int) -> Decimal:
    result = Decimal(1)
    for i in range(1, n + 1):
        result *= Decimal(i)
    return result

SIXTY_FACTORIAL = factorial(60)

# Multi-dimensional skip tracing engine
class SpatialLogCompressionEngine:
    def __init__(self):
        self.bases_priority = [Decimal('10.0'), Decimal('60.0'), Decimal('2.0')]  # 10 > 60 > 2
        self.forward_trace = {}
        self.reverse_map = {}
    
    def _apply_log(self, x: Decimal, base: Decimal) -> Decimal:
        """Safe log application with Decimal."""
        if x <= 0:
            return Decimal('0.0')  # Clamp negatives for simplicity
        return Decimal(str(math.log(float(x), float(base))))
    
    def forward_compress(self, tensor: np.ndarray, depth: int) -> np.ndarray:
        """Compress multi-dimensional tensor with hierarchical logs."""
        key = (tensor.tobytes(), depth)
        if key in self.forward_trace:
            return self.forward_trace[key]
        
        if depth <= 0:
            result = tensor
        else:
            # Base-10: Primary compression
            log_10 = np.vectorize(lambda x: self._apply_log(Decimal(str(x)), self.bases_priority[0]))(tensor)
            remainder_10 = tensor - np.vectorize(lambda x: Decimal('10.0') ** Decimal(str(int(float(x)))))(log_10)
            remainder_10 = np.where(remainder_10 < 0, tensor, remainder_10)
            
            # Base-60: Remainder compression
            log_60 = np.vectorize(lambda x: self._apply_log(Decimal(str(x)), self.bases_priority[1]))(remainder_10)
            remainder_60 = remainder_10 - np.vectorize(lambda x: Decimal('60.0') ** Decimal(str(int(float(x)))))(log_60)
            remainder_60 = np.where(remainder_60 < 0, remainder_10, remainder_60)
            
            # Base-2: Final cleanup
            log_2 = np.vectorize(lambda x: self._apply_log(Decimal(str(x)), self.bases_priority[2]))(remainder_60)
            
            # Combine spatially (sum logs for compression)
            result = log_10 + log_60 + log_2
            result = self.forward_compress(result, depth - 1)
        
        self.forward_trace[key] = result
        self.reverse_map[result.tobytes()] = (tensor, depth)
        return result
    
    def reverse_decompress(self, compressed: np.ndarray) -> np.ndarray:
        """Decompress back to original tensor."""
        key = compressed.tobytes()
        if key not in self.reverse_map:
            raise ValueError("No reverse trace available")
        tensor, depth = self.reverse_map[key]
        
        if depth <= 0:
            return tensor
        
        # Reverse spatially
        result = compressed
        for _ in range(depth):
            # Unwind in reverse order: base-2, base-60, base-10
            temp = np.vectorize(lambda y: Decimal('2.0') ** Decimal(str(y)))(result)
            temp = np.vectorize(lambda y: Decimal('60.0') ** Decimal(str(y)))(temp)
            temp = np.vectorize(lambda y: Decimal('10.0') ** Decimal(str(y)))(temp)
            result = temp
        return result

# Simulation class
class CompressionSimulation:
    def __init__(self):
        self.engine = SpatialLogCompressionEngine()
    
    def traditional_compute(self, tensor: np.ndarray, base: int) -> np.ndarray:
        """Traditional float computation with drift."""
        log_result = np.vectorize(lambda x: math.log(float(x), base))(tensor)
        return np.vectorize(lambda x: base ** x)(log_result)
    
    def framework_compute(self, tensor: np.ndarray, depth: int) -> np.ndarray:
        """Hierarchical compression framework."""
        # Convert to Decimal array
        decimal_tensor = np.vectorize(Decimal)(tensor.astype(str))
        # Pre-scale and shift
        scaled_tensor = decimal_tensor / SIXTY_FACTORIAL + Decimal('1000') * SIXTY_FACTORIAL
        compressed = self.engine.forward_compress(scaled_tensor, depth)
        decompressed = self.engine.reverse_decompress(compressed)
        return (decompressed - Decimal('1000') * SIXTY_FACTORIAL) * SIXTY_FACTORIAL
    
    def run_simulation(self, tensor: np.ndarray, primary_base: int, depth: int = 3) -> dict:
        """Compare traditional vs framework."""
        trad_result = self.traditional_compute(tensor, primary_base)
        trad_drift = tensor - trad_result
        
        framework_result = self.framework_compute(tensor, depth)
        framework_drift = tensor - np.vectorize(float)(framework_result)
        
        return {
            "original": tensor,
            "traditional_result": trad_result,
            "traditional_drift": trad_drift,
            "framework_result": framework_result,
            "framework_drift": framework_drift
        }

# Test cases
def run_tests():
    sim = CompressionSimulation()
    
    # Test tensor: 2D with base-10 and base-2 values
    test_tensor = np.array([
        [float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
        [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]
    ])
    
    # Base-10 test
    result_base10 = sim.run_simulation(test_tensor, base=10)
    print("\nBase-10 Priority Test (2D Tensor)")
    print(f"Original:\n{result_base10['original']}")
    print(f"Traditional (float):\n{result_base10['traditional_result']}\nDrift:\n{result_base10['traditional_drift']}")
    print(f"Framework:\n{result_base10['framework_result']}\nDrift:\n{result_base10['framework_drift']}")

if __name__ == "__main__":
    run_tests()
