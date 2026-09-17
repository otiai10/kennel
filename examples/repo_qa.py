"""Ask questions about a repository with the read-only tools.

Usage::

    python examples/repo_qa.py /path/to/repo "How is authentication implemented?"
"""

import asyncio
import sys

from kennel import Agent, EventType


async def main(workspace: str, question: str) -> None:
    agent = Agent(workspace, tools=["glob", "grep", "read"])
    agent.check_availability()
    agent.events.subscribe(lambda e: print(f"● {e.data['summary']}") if e.type == EventType.TOOL_STARTED else None)
    result = await agent.run(question)
    print()
    print(result.text)
    print(f"\n[{len(result.tool_calls)} tool calls, stop_reason={result.stop_reason}]")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python examples/repo_qa.py <workspace> <question>")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
