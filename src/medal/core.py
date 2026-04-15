# Backward-compatibility shim for notebooks that still do:
#   from medal.core import AutoEncoder, MEDAL, GlobalEmbeddingNormalizer
# New code should import from medal.model and medal.normalizer directly.
from medal.model import AutoEncoder, MEDAL          # noqa: F401
from medal.normalizer import GlobalEmbeddingNormalizer  # noqa: F401
