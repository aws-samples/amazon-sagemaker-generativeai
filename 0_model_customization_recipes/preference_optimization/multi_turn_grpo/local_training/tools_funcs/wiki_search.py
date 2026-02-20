"""
Wikipedia Search Tool for MT-GRPO Training

Provides wiki_search tool using Pyserini for TriviaQA task.

Usage:
    Pass this file path to --tools_script when running training:
    ./sm_mt_grpo_train.sh --config <config.yaml> --tools_script tools_funcs/wiki_search.py
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_searcher = None
PYSERINI_LOADED = False


def _init_searcher():
    global _searcher, PYSERINI_LOADED
    if PYSERINI_LOADED:
        return
    try:
        import pyserini
        os.environ["JAVA_OPTS"] = "-Xms16g -Xmx32g -XX:MaxDirectMemorySize=16g"
        from pyserini.search.lucene import LuceneSearcher
        _searcher = LuceneSearcher.from_prebuilt_index("wikipedia-kilt-doc")
        PYSERINI_LOADED = True
        logger.info("Pyserini Wikipedia searcher initialized")
    except ImportError:
        logger.warning("Pyserini not available")
    except Exception as e:
        logger.warning(f"Failed to initialize Pyserini: {e}")


def wiki_search(query: str) -> str:
    """
    Searches Wikipedia and returns the top matching article content.

    Args:
        query: The search query string

    Returns:
        The Wikipedia article content or error message
    """
    _init_searcher()
    if not PYSERINI_LOADED:
        return "Error: Pyserini is not available."
    try:
        hits = _searcher.search(query, k=1)
        if not hits:
            return "No relevant Wikipedia content found."
        doc = _searcher.doc(hits[0].docid)
        return json.loads(doc.raw())["contents"]
    except Exception as e:
        return f"Error during search: {e}"


TOOL_FUNCTIONS = [wiki_search]
