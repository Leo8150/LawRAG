"""Apply exponential forgetting and purge expired memory tombstones."""

import asyncio

from app.memory.long_term import long_term_memory


async def _run() -> None:
    result = await long_term_memory.maintain()
    print(
        "Memory maintenance complete: "
        f"scanned={result['scanned']}, "
        f"forgotten={result['forgotten']}, "
        f"purged={result['purged']}"
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
