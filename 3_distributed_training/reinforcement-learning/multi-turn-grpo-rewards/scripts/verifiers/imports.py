"""Central import handling for platform-specific dependencies."""
import platform

# Check if we're on macOS (Darwin)
IS_MACOS = platform.system() == 'Darwin'

# Use mock vLLM on macOS, real vLLM otherwise
if IS_MACOS:
    from .mock_vllm import LLM, SamplingParams
else:
    from vllm import LLM, SamplingParams  # type: ignore

__all__ = ['LLM', 'SamplingParams', 'IS_MACOS'] 