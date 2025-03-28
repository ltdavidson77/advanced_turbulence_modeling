import numpy as np
from typing import Dict, Any, Optional, Tuple
import requests  # For HTTP requests to external APIs
from precision_controls import PrecisionControls
from quantum_state import QuantumState
from enterprise_framework import QuantumEnterpriseFramework
from quantum_ml_pipeline import QuantumMLPipeline
from feature_mapper import FeatureMapper
from hardware_interface import HardwareInterface

# Global instances
PRECISION_CONFIG = PrecisionControls()
FRAMEWORK = QuantumEnterpriseFramework(compress_depth=3)
PIPELINE = QuantumMLPipeline(compress_depth=3, framework=FRAMEWORK)
FEATURE_MAPPER = FeatureMapper()
HARDWARE = HardwareInterface(use_gpu=True)  # Default to GPU if available

class APIClient:
    """Client for bridging quantum-classical data to external AI systems."""
    
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_endpoint (str): URL of the external AI system API.
            api_key (str, optional): Authentication key for the API.
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def prepare_data(self, data: np.ndarray, state_id: str, 
                    target_shape: Optional[Tuple[int, ...]] = None, 
                    gate: Optional[np.ndarray] = None, 
                    transform_id: Optional[str] = None) -> Dict[str, Any]:
        """Prepare data using the framework pipeline."""
        # Run through the ML pipeline
        pipeline_result = PIPELINE.run_pipeline(
            data=data,
            state_id=state_id,
            gate=gate,
            target_shape=target_shape,
            transform_id=transform_id
        )
        
        # Get the quantum state for further ops if needed
        qstate = FRAMEWORK.state_registry[state_id]
        
        return {
            "compressed_size": pipeline_result["compressed_size_bytes"],
            "features": pipeline_result.get("features", None),
            "qstate": qstate
        }
    
    def send_to_api(self, data: Dict[str, Any], endpoint_path: str = "/process") -> Dict[str, Any]:
        """Send prepared data to the external API."""
        # Prepare payload
        payload = {
            "compressed_size": data["compressed_size"],
            "features": data["features"].tolist() if data["features"] is not None else None
        }
        
        try:
            response = requests.post(
                f"{self.api_endpoint}{endpoint_path}",
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return {"error": str(e)}
    
    def bridge_to_ai(self, data: np.ndarray, state_id: str, 
                    target_shape: Optional[Tuple[int, ...]] = None, 
                    gate: Optional[np.ndarray] = None, 
                    transform_id: Optional[str] = None) -> Dict[str, Any]:
        """Full workflow: prepare data and send to external AI system."""
        # Prepare data using the pipeline
        prepared_data = self.prepare_data(data, state_id, target_shape, gate, transform_id)
        
        # Send to API
        api_response = self.send_to_api(prepared_data)
        
        return {
            "prepared_data": {
                "compressed_size": prepared_data["compressed_size"],
                "features_shape": prepared_data["features"].shape if prepared_data["features"] is not None else None
            },
            "api_response": api_response
        }

if __name__ == "__main__":
    # Test the API client
    client = APIClient(api_endpoint="http://example.com/api", api_key="test_key")
    
    # Test data: 3D tensor
    test_data = np.array([
        [[float(Decimal('10') ** 10 + Decimal('10') ** -20), float(Decimal('2') ** 32 + Decimal('2') ** -40)],
         [float(Decimal('10') ** 8 + Decimal('10') ** -15), float(Decimal('2') ** 16 + Decimal('2') ** -20)]],
        [[float(Decimal('10') ** 6), float(Decimal('2') ** 10)],
         [float(Decimal('10') ** 4), 0.0]]
    ])
    
    # Register a transform
    def scale_features(features: np.ndarray) -> np.ndarray:
        return features * 2.0
    FEATURE_MAPPER.register_transform("scale", scale_features)
    
    # Hadamard gate
    gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # Bridge to AI
    result = client.bridge_to_ai(
        data=test_data,
        state_id="api_test",
        target_shape=(4, 2),
        gate=gate,
        transform_id="scale"
    )
    
    print("Prepared Data:", result["prepared_data"])
    print("API Response:", result["api_response"])
