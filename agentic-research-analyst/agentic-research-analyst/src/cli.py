"""
CLI — Command-line interface for the research agent.

Supports interactive mode, single-brief execution, and PDF-augmented research.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.agents.research_agent import ResearchAgent
from src.config.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Research Analyst — Multi-tool LLM research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --brief "Analyze AI code assistant market in 2025"
  python -m src.cli --brief "Summarize this paper" --pdf ./papers/attention.pdf
  python -m src.cli --interactive
        """,
    )
    parser.add_argument("--brief", type=str, help="Research brief to execute")
    parser.add_argument("--pdf", type=str, help="PDF file to include as context")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--max-tools", type=int, default=10)
    parser.add_argument("--output", type=str, help="Save memo to file")

    args = parser.parse_args()

    agent = ResearchAgent(settings=Settings())

    if args.interactive:
        _interactive_mode(agent)
    elif args.brief:
        _single_brief(agent, args)
    else:
        parser.print_help()
        sys.exit(1)


def _single_brief(agent: ResearchAgent, args: argparse.Namespace) -> None:
    """Execute a single research brief."""
    print(f"\n{'='*60}")
    print(f"  AGENTIC RESEARCH ANALYST")
    print(f"{'='*60}")
    print(f"\n  Brief: {args.brief[:80]}...")
    print(f"  Max tools: {args.max_tools}")
    print(f"  Format: {args.format}\n")

    memo = agent.run(
        brief=args.brief,
        output_format=args.format,
        max_tools=args.max_tools,
    )

    if args.format == "json":
        output = json.dumps(memo.to_json(), indent=2)
    else:
        output = memo.to_markdown()

    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\n  Saved to {args.output}")


def _interactive_mode(agent: ResearchAgent) -> None:
    """Run the agent in interactive loop mode."""
    print(f"\n{'='*60}")
    print(f"  AGENTIC RESEARCH ANALYST — Interactive Mode")
    print(f"{'='*60}")
    print(f"  Type a research brief, or 'quit' to exit.\n")

    while True:
        try:
            brief = input("  Brief > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if brief.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break

        if not brief:
            continue

        memo = agent.run(brief=brief)
        print(f"\n{memo.to_markdown()}\n")


if __name__ == "__main__":
    main()
