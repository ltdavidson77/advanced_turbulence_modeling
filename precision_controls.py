from decimal import Decimal, getcontext
import numpy as np

# Set global precision to handle tiny fractions (e.g., 10^-30, 2^-72)
getcontext().prec = 100

class PrecisionControls:
    """Configuration for precision, bases, and scaling in the compression framework."""
    
    def __init__(self):
        # Hierarchical base constants (base-10 primary, base-60 secondary, base-2 tertiary)
        self.BASE_10 = Decimal('10.0')
        self.BASE_60 = Decimal('60.0')
        self.BASE_2 = Decimal('2.0')
        self.bases_priority = [self.BASE_10, self.BASE_60, self.BASE_2]
        
        # Precomputed scaling factor: 60! for range normalization
        self.SIXTY_FACTORIAL = self._compute_factorial(60)
        
        # Minimum value to avoid underflow or negative log issues
        self.MIN_VALUE = Decimal('1e-100')
        
    def _compute_factorial(self, n: int) -> Decimal:
        """Compute factorial with Decimal precision."""
        result = Decimal('1')
        for i in range(1, n + 1):
            result *= Decimal(str(i))
        return result
    
    def get_base(self, index: int) -> Decimal:
        """Retrieve base by priority index (0 = base-10, 1 = base-60, 2 = base-2)."""
        if 0 <= index < len(self.bases_priority):
            return self.bases_priority[index]
        raise ValueError("Invalid base index")
    
    def normalize_scalar(self, x: Decimal) -> Decimal:
        """Normalize a scalar value to prevent overflow/underflow."""
        if x <= self.MIN_VALUE:
            return self.MIN_VALUE
        return x / self.SIXTY_FACTORIAL
    
    def denormalize_scalar(self, x: Decimal) -> Decimal:
        """Reverse normalization for scalar values."""
        return x * self.SIXTY_FACTORIAL
    
    def normalize_array(self, array: np.ndarray) -> np.ndarray:
        """Normalize a multi-dimensional array for compression."""
        return np.vectorize(lambda x: self.normalize_scalar(Decimal(str(x))))(array)
    
    def denormalize_array(self, array: np.ndarray) -> np.ndarray:
        """Reverse normalization for multi-dimensional arrays."""
        return np.vectorize(lambda x: self.denormalize_scalar(Decimal(str(x))))(array)
    
    def shift_positive(self, x: Decimal, shift_factor: Decimal = Decimal('1000')) -> Decimal:
        """Shift value to ensure positivity for logarithmic operations."""
        return x + shift_factor * self.SIXTY_FACTORIAL
    
    def unshift_positive(self, x: Decimal, shift_factor: Decimal = Decimal('1000')) -> Decimal:
        """Reverse the positivity shift."""
        return x - shift_factor * self.SIXTY_FACTORIAL
    
    def stabilize_carryover(self, x: Decimal, base: Decimal) -> Decimal:
        """Adjust value to minimize carryover drift for a given base."""
        if x <= self.MIN_VALUE:
            return self.MIN_VALUE
        log_val = Decimal(str(math.log(float(x), float(base))))
        int_part = Decimal(str(int(float(log_val))))
        remainder = x - base ** int_part
        return remainder if remainder > self.MIN_VALUE else self.MIN_VALUE

# Singleton instance for global access
PRECISION_CONFIG = PrecisionControls()

if __name__ == "__main__":
    # Quick test
    config = PrecisionControls()
    
    # Test scalar normalization
    x = Decimal('10') ** 10 + Decimal('10') ** -20
    normalized = config.normalize_scalar(x)
    denormalized = config.denormalize_scalar(normalized)
    print(f"Original: {x}")
    print(f"Normalized: {normalized}")
    print(f"Denormalized: {denormalized}")
    print(f"Drift: {x - denormalized}")
    
    # Test array normalization
    arr = np.array([float(x), float(Decimal('2') ** 32 + Decimal('2') ** -40)])
    norm_arr = config.normalize_array(arr)
    denorm_arr = config.denormalize_array(norm_arr)
    print(f"\nOriginal Array:\n{arr}")
    print(f"Normalized Array:\n{norm_arr}")
    print(f"Denormalized Array:\n{denorm_arr}")
