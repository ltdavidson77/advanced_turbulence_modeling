import numpy as np
from decimal import Decimal
from precision_controls import PrecisionControls
from typing import Union, Tuple, Optional

# Global precision config
PRECISION_CONFIG = PrecisionControls()

class DataPreprocessor:
    """Preprocesses raw data for quantum-classical compression and ML pipelines."""
    
    def __init__(self, target_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize the preprocessor with an optional target shape.
        
        Args:
            target_shape (tuple, optional): Desired output shape for reshaped data.
        """
        self.target_shape = target_shape
    
    def normalize_data(self, data: Union[float, np.ndarray]) -> np.ndarray:
        """Normalize raw data to a Decimal-based array for precision."""
        if isinstance(data, (int, float)):
            data = np.array([data], dtype=float)
        elif not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a scalar or NumPy array")
        
        # Convert to Decimal array with normalization
        norm_data = PRECISION_CONFIG.normalize_array(data)
        return norm_data
    
    def handle_edge_cases(self, data: np.ndarray) -> np.ndarray:
        """Handle zeros, negatives, and extreme values."""
        # Shift to positive range and clamp at min value
        shifted_data = np.vectorize(
            lambda x: PRECISION_CONFIG.shift_positive(Decimal(str(x)))
        )(data)
        return shifted_data
    
    def reshape_data(self, data: np.ndarray, target_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Reshape data to a target shape, padding or trimming as needed."""
        target = target_shape if target_shape else self.target_shape
        if not target:
            return data
        
        current_size = data.size
        target_size = np.prod(target)
        
        flat_data = data.flatten()
        if current_size < target_size:
            padded = np.pad(flat_data, (0, target_size - current_size), mode='constant', constant_values=PRECISION_CONFIG.MIN_VALUE)
        elif current_size > target_size:
            padded = flat_data[:target_size]
        else:
            padded = flat_data
        
        return padded.reshape(target)
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate data for compatibility with the framework."""
        if np.any(np.vectorize(lambda x: not isinstance(x, Decimal))(data)):
            raise ValueError("Data must be in Decimal format after normalization")
        if data.size == 0:
            raise ValueError("Data array cannot be empty")
    
    def preprocess(self, raw_data: Union[float, np.ndarray], 
                  target_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Full preprocessing pipeline: normalize, handle edges, reshape, validate."""
        # Step 1: Normalize
        norm_data = self.normalize_data(raw_data)
        
        # Step 2: Handle edge cases
        safe_data = self.handle_edge_cases(norm_data)
        
        # Step 3: Reshape if needed
        shaped_data = self.reshape_data(safe_data, target_shape)
        
        # Step 4: Validate
        self.validate_data(shaped_data)
        
        return shaped_data

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor(target_shape=(2, 2))
    
    # Test data: mixed scalar and array inputs
    raw_scalar = float(Decimal('10') ** 10 + Decimal('10') ** -20)
    raw_array = np.array([
        [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
        [float(Decimal('2') ** 16 + Decimal('2') ** -20), 0.0]
    ])
    
    # Preprocess scalar
    scalar_result = preprocessor.preprocess(raw_scalar)
    print("Preprocessed Scalar:\n", np.vectorize(float)(scalar_result))
    
    # Preprocess array
    array_result = preprocessor.preprocess(raw_array)
    print("Preprocessed Array:\n", np.vectorize(float)(array_result))
    
    # Test with custom shape
    custom_result = preprocessor.preprocess(raw_array, target_shape=(4, 1))
    print("Preprocessed with Custom Shape (4, 1):\n", np.vectorize(float)(custom_result))
