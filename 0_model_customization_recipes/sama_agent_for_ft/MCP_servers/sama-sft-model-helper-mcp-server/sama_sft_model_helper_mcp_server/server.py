#!/usr/bin/env python3
"""
SAMA SFT Model Helper MCP Server
Provides tools for discovering, filtering, and selecting models from HuggingFace Hub
for Supervised Fine-Tuning workflows.
"""

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcp = FastMCP("sama-sft-model-helper-mcp-server")

# ---------------------------------------------------------------------------
# HuggingFace Hub helpers
# ---------------------------------------------------------------------------

def _query_hf_models(
    author: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Query HuggingFace Hub API for text-generation models.

    Compatible with huggingface_hub >= 1.0 (v1.7+ dropped 'direction' and 'author';
    uses 'filter' for org-scoped queries and 'sort' accepts 'downloads' directly).
    """
    from huggingface_hub import HfApi
    import inspect

    api = HfApi()

    # Build kwargs compatible with the installed huggingface_hub version
    sig_params = set(inspect.signature(api.list_models).parameters.keys())

    kwargs: Dict[str, Any] = {
        "pipeline_tag": "text-generation",
        "sort": "downloads",
        "limit": limit,
        "expand": ["safetensors"],
    }

    # 'direction' was removed in v1.x; only pass it if the API still accepts it
    if "direction" in sig_params:
        kwargs["direction"] = -1

    # 'author' was removed in v1.x; use 'filter' with "author:<name>" instead
    if author:
        if "author" in sig_params:
            kwargs["author"] = author
        else:
            # v1.7+ style: pass org as a filter string
            kwargs["filter"] = author

    if search:
        kwargs["search"] = search

    models = list(api.list_models(**kwargs))
    results = []
    for m in models:
        # Extract parameter count from safetensors metadata
        safetensors = getattr(m, "safetensors", None)
        param_count = None
        if safetensors is not None:
            # v1.7+ returns a SafeTensorsInfo object with a .total attribute
            if hasattr(safetensors, "total"):
                param_count = safetensors.total
            elif isinstance(safetensors, dict):
                param_count = safetensors.get("total")

        results.append(
            {
                "model_id": m.id,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "parameters": param_count,
                "last_modified": str(getattr(m, "last_modified", "")),
                "pipeline_tag": getattr(m, "pipeline_tag", ""),
                "tags": getattr(m, "tags", []),
            }
        )
    return results


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_models_by_family(
    model_family: str,
    limit: int = 30,
) -> Dict[str, Any]:
    """
    Search HuggingFace Hub for text-generation models by model family / organization.

    This is STEP 1 of the SFT workflow. Use this as the first step to discover
    available models. Provide an org name like 'meta-llama', 'Qwen', 'openai',
    'google', 'microsoft', 'deepseek-ai', or any HuggingFace organization.

    WORKFLOW: search/filter → get_model_details → generate_sft_recipe → generate_training_script

    Args:
        model_family: HuggingFace organization or author name (e.g. 'meta-llama').
        limit: Maximum number of models to return (default 30).

    Returns:
        Dictionary with status and a list of models including id, parameters,
        downloads, and likes.
    """
    try:
        logger.info(f"Searching models for family: {model_family}")
        models = _query_hf_models(author=model_family, limit=limit)
        if not models:
            return {
                "status": "success",
                "models": [],
                "message": f"No text-generation models found for family '{model_family}'.",
            }
        return {
            "status": "success",
            "model_family": model_family,
            "total_found": len(models),
            "models": models,
            "next_step": "Present this list to the user. Optionally use filter_models_by_size to narrow down, then ask the user to select a model and call get_model_details.",
        }
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        return {"status": "error", "message": f"HuggingFace Hub query failed: {e}"}


@mcp.tool()
def search_models_by_keyword(
    keyword: str,
    limit: int = 30,
) -> Dict[str, Any]:
    """
    Search HuggingFace Hub for text-generation models by keyword.

    This is STEP 1 (alternative) of the SFT workflow. Useful when the user
    describes a model broadly (e.g. 'llama 3', 'qwen 7b', 'code generation').

    WORKFLOW: search/filter → get_model_details → generate_sft_recipe → generate_training_script

    Args:
        keyword: Free-text search query.
        limit: Maximum number of models to return (default 30).

    Returns:
        Dictionary with status and a list of matching models.
    """
    try:
        logger.info(f"Searching models by keyword: {keyword}")
        models = _query_hf_models(search=keyword, limit=limit)
        if not models:
            return {
                "status": "success",
                "models": [],
                "message": f"No text-generation models found for keyword '{keyword}'.",
            }
        return {
            "status": "success",
            "keyword": keyword,
            "total_found": len(models),
            "models": models,
            "next_step": "Present this list to the user. Optionally use filter_models_by_size to narrow down, then ask the user to select a model and call get_model_details.",
        }
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        return {"status": "error", "message": f"HuggingFace Hub query failed: {e}"}


@mcp.tool()
def filter_models_by_size(
    models: List[Dict[str, Any]],
    min_params: Optional[int] = None,
    max_params: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Filter a previously retrieved model list by parameter count range.

    Pass the 'models' list from a prior search_models_by_family or
    search_models_by_keyword call, along with min/max parameter bounds.
    Parameter counts are in raw numbers (e.g. 3_000_000_000 for 3B).

    Args:
        models: List of model dicts (from a prior search call).
        min_params: Minimum parameter count (inclusive). None means no lower bound.
        max_params: Maximum parameter count (inclusive). None means no upper bound.

    Returns:
        Filtered list of models within the specified size range.
    """
    try:
        if min_params is not None and max_params is not None and max_params < min_params:
            return {
                "status": "error",
                "message": f"Invalid range: max_params ({max_params}) < min_params ({min_params}).",
            }

        filtered = []
        for m in models:
            p = m.get("parameters")
            if p is None:
                continue
            if min_params is not None and p < min_params:
                continue
            if max_params is not None and p > max_params:
                continue
            filtered.append(m)

        # Sort by parameters ascending
        filtered.sort(key=lambda x: x.get("parameters") or 0)

        return {
            "status": "success",
            "total_found": len(filtered),
            "filter": {"min_params": min_params, "max_params": max_params},
            "models": filtered,
            "next_step": "Present the filtered list to the user and ask them to select a model. Then call get_model_details with the chosen model_id.",
        }
    except Exception as e:
        logger.error(f"Failed to filter models: {e}")
        return {"status": "error", "message": f"Filter failed: {e}"}


@mcp.tool()
def get_model_details(
    model_id: str,
) -> Dict[str, Any]:
    """
    Get detailed metadata for a specific HuggingFace model.

    This is STEP 2 of the SFT workflow. Call this after the user selects a model
    from the search/filter results. Returns full metadata needed for recipe
    generation. You MUST store the 'parameters' count for use in later steps.

    WORKFLOW: search/filter → get_model_details → generate_sft_recipe → generate_training_script

    Args:
        model_id: Full HuggingFace model identifier (e.g. 'meta-llama/Llama-3.2-3B-Instruct').

    Returns:
        Detailed model information including architecture, license, tags, and parameters.
    """
    try:
        from huggingface_hub import HfApi

        logger.info(f"Getting details for model: {model_id}")
        api = HfApi()
        info = api.model_info(model_id)

        safetensors = getattr(info, "safetensors", None)
        param_count = None
        if safetensors is not None:
            if hasattr(safetensors, "total"):
                param_count = safetensors.total
            elif isinstance(safetensors, dict):
                param_count = safetensors.get("total")

        config = getattr(info, "config", None) or {}
        arch = config.get("architectures", [])

        return {
            "status": "success",
            "model_id": info.id,
            "parameters": param_count,
            "downloads": getattr(info, "downloads", 0),
            "likes": getattr(info, "likes", 0),
            "pipeline_tag": getattr(info, "pipeline_tag", ""),
            "license": getattr(info, "card_data", {}).get("license", "unknown") if hasattr(info, "card_data") and info.card_data else "unknown",
            "tags": getattr(info, "tags", []),
            "architectures": arch,
            "library_name": getattr(info, "library_name", ""),
            "last_modified": str(getattr(info, "last_modified", "")),
            "model_family": model_id.split("/")[0] if "/" in model_id else "unknown",
            "model_name": model_id.split("/")[1] if "/" in model_id else model_id,
            "next_step": "Confirm model selection with the user. Then proceed to generate_sft_recipe (sft_recipe_generator server) — ask for strategy (peft/spectrum/full) and dataset name.",
        }
    except Exception as e:
        logger.error(f"Failed to get model details: {e}")
        return {"status": "error", "message": f"Model not found or API error: {e}"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Main entry point for the MCP server."""
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
