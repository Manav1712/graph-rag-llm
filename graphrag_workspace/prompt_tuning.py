import os
from pathlib import Path
import asyncio

# Ensure these match your folder structure
PROJECT_ROOT = Path("./graphrag_workspace")
CONFIG_PATH = PROJECT_ROOT / "settings.yaml"



async def main():
    import graphrag.api as api
    from graphrag.config.load_config import load_config

    # Load config
    graphrag_config = load_config(PROJECT_ROOT)

    # Prompt tuning options (edit these as desired)
    prompt_tune_kwargs = {
        "config": graphrag_config,
        "root": PROJECT_ROOT,
        "domain": "physics",             # or "" to infer
        "selection_method": "random",    # or 'auto', 'top', 'all'
        "limit": 15,                     # number of text units
        "language": "English",           # or "" to auto-detect
        "max_tokens": 2048,
        "chunk_size": 256,
        "n_subset_max": 300,
        "k": 15,
        "min_examples_required": 3,
        "discover_entity_types": False,  # set to True to auto-discover
        "output": PROJECT_ROOT / "prompts"
    }

    # Run prompt tuning
    await api.prompt_tune(**prompt_tune_kwargs)
    print("Prompt tuning complete! Check prompts/ for outputs.")

if __name__ == "__main__":
    asyncio.run(main())