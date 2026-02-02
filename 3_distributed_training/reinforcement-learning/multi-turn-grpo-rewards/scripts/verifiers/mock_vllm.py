"""Mock vLLM module for macOS development and testing.
Only provides the minimal interface needed for imports to work.
"""

class SamplingParams:
    """Mock sampling parameters."""
    def __init__(self, **kwargs):
        self.temperature = kwargs.get('temperature', 1.0)
        self.top_p = kwargs.get('top_p', 1.0)
        self.max_tokens = kwargs.get('max_tokens', 100)

class LLM:
    """Mock LLM that raises NotImplementedError if actually used."""
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get('model', 'mock')
        self.dtype = kwargs.get('dtype', 'float16')
        
    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a mock vLLM for macOS development. "
            "Install the real vLLM package for actual model inference."
        ) 