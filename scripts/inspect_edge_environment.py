"""Read device/runtime identity without changing Jetson power or clock settings."""
from pathlib import Path
import sys
import json
import importlib.metadata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.edge_benchmark import probe_system

if __name__ == "__main__":
    result = probe_system()
    result["packages"] = {}
    for name in ["numpy", "pandas", "scikit-learn", "torch", "xgboost", "psutil"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        import torch
        result["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                          "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    except Exception as exc:
        result["cuda"] = {"error": str(exc)}
    print(json.dumps(result, indent=2))
