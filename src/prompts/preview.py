from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prompts.prompt_pack import PromptPackLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a rendered Prompt Pack prompt.")
    parser.add_argument("--node", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("config/prompts"))
    args = parser.parse_args()

    variables = json.loads(args.fixture.read_text(encoding="utf-8"))
    pack = PromptPackLoader(root=args.root).load_active()
    rendered = pack.render(node_id=args.node, lane=args.lane, variables=variables)
    print(rendered.text)
    print(f"\nprompt_hash={rendered.hash}")


if __name__ == "__main__":
    main()
