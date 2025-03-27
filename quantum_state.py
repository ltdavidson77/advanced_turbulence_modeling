import numpy as np
from decimal import Decimal
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine

# Initialize global precision and compression
PRECISION_CONFIG = PrecisionControls()
COMPRESSION_ENGINE = SpatialLogCompressionEngine()

class QuantumState:
    """Represents and manipulates multi-dimensional quantum-like states with compression."""
    
    def __init__(self, state: np.ndarray, compress_depth: int = 3):
        """
        Initialize a quantum state with an optional compression depth.
        
        Args:
            state (np.ndarray): Initial state as a multi-dimensional array (e.g., vector or tensor).
            compress_depth (int): Depth for logarithmic compression.
        """
        # Convert input to Decimal array for precision
        self.state = PRECISION_CONFIG.normalize_array(state)
        self.compress_depth = compress_depth
        self.metadata = {
            "shape": state.shape,
            "is_compressed": False
        }
    
    def compress(self) -> None:
        """Compress the state using the spatial log compression engine."""
        if not self.metadata["is_compressed"]:
            self.state = COMPRESSION_ENGINE.forward_compress(self.state, self.compress_depth)
            self.metadata["is_compressed"] = True
    
    def decompress(self) -> None:
        """Decompress the state back to its original form."""
        if self.metadata["is_compressed"]:
            self.state = COMPRESSION_ENGINE.reverse_decompress(self.state)
            self.state = PRECISION_CONFIG.denormalize_array(self.state)
            self.metadata["is_compressed"] = False
    
    def apply_gate(self, gate: np.ndarray) -> None:
        """Apply a quantum-like gate operation to the state."""
        if self.metadata["is_compressed"]:
            self.decompress()  # Gate ops on uncompressed state
        
        # Ensure gate matches state dimensions
        if gate.shape[0] != self.state.shape[-1]:
            raise ValueError("Gate dimensions must match state dimensions")
        
        # Multi-dimensional gate application (e.g., matrix multiplication along last axis)
        if len(self.state.shape) > 1:
            # For tensors, apply gate to each vector slice
            result = np.zeros(self.state.shape, dtype=object)
            for idx in np.ndindex(self.state.shape[:-1]):
                result[idx] = gate @ self.state[idx]
            self.state = result
        else:
            self.state = gate @ self.state
    
    def measure(self, shots: int = 1024) -> dict:
        """Simulate measurement on the uncompressed state."""
        if self.metadata["is_compressed"]:
            self.decompress()
        
        # Convert to float for probability calculation
        state_float = np.vectorize(float)(self.state)
        probs = np.abs(state_float) ** 2
        total_prob = np.sum(probs)
        if total_prob > 0:
            probs /= total_prob  # Normalize
        
        # Simulate measurements
        flat_probs = probs.flatten()
        outcomes = np.random.multinomial(shots, flat_probs)
        return {f"state_{i}": count / shots for i, count in enumerate(outcomes)}
    
    @property
    def compressed_size(self) -> int:
        """Return the size of the compressed state in bytes."""
        if not self.metadata["is_compressed"]:
            self.compress()
        return self.state.nbytes

if __name__ == "__main__":
    # Test the QuantumState class
    # 2D state with base-10 and base-2 values
    initial_state = np.array([
        [float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
        [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]
    ])
    
    qstate = QuantumState(initial_state, compress_depth=3)
    print("Original State:\n", initial_state)
    
    # Compress
    qstate.compress()
    print("Compressed Size (bytes):", qstate.compressed_size)
    
    # Decompress and check
    qstate.decompress()
    print("Decompressed State:\n", np.vectorize(float)(qstate.state))
    
    # Apply a simple gate (Hadamard-like)
    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    qstate.apply_gate(gate)
    print("After Gate:\n", np.vectorize(float)(qstate.state))
    
    # Measure
    results = qstate.measure(shots=1000)
    print("Measurement Results:", results)
