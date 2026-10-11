"""Create LawRAG MySQL tables without invoking any model API."""

from app.config import settings
from app.db.repository import get_parent_child_repository
from app.memory.repository import get_long_term_memory_repository


def main() -> None:
    repository = get_parent_child_repository()
    repository.create_schema()
    get_long_term_memory_repository().ensure_lifecycle_schema()
    print(f"MySQL schema ready: {settings.MYSQL_URL.split('@')[-1]}")


if __name__ == "__main__":
    main()
