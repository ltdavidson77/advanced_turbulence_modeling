import numpy as np
from decimal import Decimal
from precision_controls import PrecisionControls
from spatial_log_compression import SpatialLogCompressionEngine
from quantum_state import QuantumState
from typing import Dict, Any, Optional

# Initialize global instances
PRECISION_CONFIG = PrecisionControls()
COMPRESSION_ENGINE = SpatialLogCompressionEngine()

class QuantumEnterpriseFramework:
    """Orchestrates quantum-classical workflows with compression and AI bridging."""
    
    def __init__(self, compress_depth: int = 3):
        """
        Initialize the enterprise framework.
        
        Args:
            compress_depth (int): Depth for logarithmic compression.
        """
        self.compress_depth = compress_depth
        self.state_registry: Dict[str, QuantumState] = {}
        self.results_cache: Dict[str, Any] = {}
    
    def register_state(self, state_id: str, initial_state: np.ndarray) -> None:
        """Register a new quantum state in the framework."""
        qstate = QuantumState(initial_state, compress_depth=self.compress_depth)
        self.state_registry[state_id] = qstate
    
    def compress_state(self, state_id: str) -> int:
        """Compress a registered state and return its compressed size."""
        if state_id not in self.state_registry:
            raise ValueError(f"State {state_id} not registered")
        qstate = self.state_registry[state_id]
        qstate.compress()
        return qstate.compressed_size
    
    def process_quantum_workflow(self, state_id: str, gate: Optional[np.ndarray] = None) -> None:
        """Process a quantum workflow: apply gate and measure."""
        if state_id not in self.state_registry:
            raise ValueError(f"State {state_id} not registered")
        
        qstate = self.state_registry[state_id]
        
        # Decompress if needed, apply gate, recompress
        if gate is not None:
            qstate.apply_gate(gate)
        results = qstate.measure(shots=1000)
        self.results_cache[state_id] = results
    
    def bridge_ai_weights(self, state_id: str, target_shape: tuple) -> np.ndarray:
        """Bridge quantum state to AI-compatible weights."""
        if state_id not in self.state_registry:
            raise ValueError(f"State {state_id} not registered")
        
        qstate = self.state_registry[state_id]
        if qstate.metadata["is_compressed"]:
            qstate.decompress()
        
        # Convert state to float array and reshape for AI
        weights = np.vectorize(float)(qstate.state)
        current_size = weights.size
        
        # Simple reshaping or padding to match target shape
        target_size = np.prod(target_shape)
        if current_size < target_size:
            weights = np.pad(weights.flatten(), (0, target_size - current_size), mode='constant')
        elif current_size > target_size:
            weights = weights.flatten()[:target_size]
        
        return weights.reshape(target_shape)
    
    def execute_full_workflow(self, state_id: str, initial_state: np.ndarray, 
                            gate: Optional[np.ndarray] = None, 
                            ai_target_shape: Optional[tuple] = None) -> Dict[str, Any]:
        """Execute a complete workflow: register, compress, process, and bridge."""
        self.register_state(state_id, initial_state)
        compressed_size = self.compress_state(state_id)
        self.process_quantum_workflow(state_id, gate)
        
        result = {
            "compressed_size_bytes": compressed_size,
            "measurement_results": self.results_cache.get(state_id, {})
        }
        
        if ai_target_shape:
            result["ai_weights"] = self.bridge_ai_weights(state_id, ai_target_shape)
        
        return result

if __name__ == "__main__":
    # Test the framework
    framework = QuantumEnterpriseFramework(compress_depth=3)
    
    # Test state: 2D tensor with mixed base values
    initial_state = np.array([
        [float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
        [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]
    ])
    
    # Simple Hadamard gate
    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Execute workflow
    result = framework.execute_full_workflow(
        state_id="test_state",
        initial_state=initial_state,
        gate=gate,
        ai_target_shape=(2, 2)  # Example AI weight shape
    )
    
    print("Compressed Size (bytes):", result["compressed_size_bytes"])
    print("Measurement Results:", result["measurement_results"])
    print("AI Weights:\n", result["ai_weights"])
