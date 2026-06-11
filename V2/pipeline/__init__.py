"""V2 Pipeline package — Stage1 (Regime) → Stage2 (Signal) → Stage3 (Exit)."""

from .stage1_regime import RegimeDetector, RegimeType
from .stage2_signal import EnsembleSignal, SignalResult
from .stage3_exit   import AdaptiveExit, ExitAction
from .orchestrator  import PipelineOrchestrator

__all__ = [
    # Stage 1
    "RegimeDetector",
    "RegimeType",
    # Stage 2
    "EnsembleSignal",
    "SignalResult",
    # Stage 3
    "AdaptiveExit",
    "ExitAction",
    # Orchestrator
    "PipelineOrchestrator",
]

