"""数据导入脚本

用法:
    cd backend
    python -m scripts.import_data

支持:
- data/laws/ 下的 .txt 文件（递归子目录）
- data/cases/ 下的 .json (JSONL) 文件（递归子目录）
"""

import os
import sys
import json

# 将 backend 目录加入 Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.db.repository import get_parent_child_repository
from app.services.ingestion_service import persist_and_index, prepare_document


def read_file(filepath: str) -> str:
    """尝试多种编码读取文件"""
    for enc in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码: {filepath}")


def collect_files(dir_path: str, extensions: set[str]) -> list[str]:
    """递归收集目录下指定后缀的文件"""
    results = []
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                results.append(os.path.join(root, fname))
    return results


def import_laws(repository, dir_path: str) -> int:
    """导入法律条文 (.txt 文件)"""
    files = collect_files(dir_path, {".txt", ".md"})
    if not files:
        print("  目录为空或不存在")
        return 0

    total_chunks = 0
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            text = read_file(fpath)
            tree = prepare_document(
                text,
                doc_type="law",
                source_file=fname,
                source_path=fpath,
                dataset_name="laws",
            )
            persist_and_index(
                tree,
                collection_name=settings.LAWS_COLLECTION,
                repository=repository,
            )
            total_chunks += len(tree.child_ids)
            print(f"  ✓ {fname}: {len(tree.document.parents)} 个父块 / {len(tree.child_ids)} 个子块")
        except Exception as e:
            print(f"  ✗ {fname}: {e}")

    return total_chunks


def _extract_case_text(obj: dict) -> tuple[str, dict]:
    """从 JSON 对象中提取案例文本和额外元数据

    支持多种格式：
    - CAIL2019-SCM: {"A": "...", "B": "...", "C": "...", "label": "..."}
    - CAIL2018:     {"fact": "...", "meta": {"accusation": [...], ...}}
    - 通用JSONL:    {"fact": "...", ...} 或 {"text": "...", ...} 或 {"content": "...", ...}
    """
    extra_meta = {}

    # CAIL2018 格式
    if "fact" in obj:
        text = obj["fact"]
        meta = obj.get("meta", {})
        if meta.get("accusation"):
            extra_meta["accusation"] = "；".join(meta["accusation"])
        if meta.get("relevant_articles"):
            extra_meta["relevant_articles"] = "；".join(
                str(a) for a in meta["relevant_articles"]
            )
        if meta.get("criminals"):
            extra_meta["criminals"] = "；".join(meta["criminals"])
        # 处理来自 prepare_datasets 转换后的格式
        if "accusation" in obj and not extra_meta.get("accusation"):
            extra_meta["accusation"] = obj["accusation"]
        if "sentence" in obj:
            extra_meta["sentence"] = obj["sentence"]
        return text, extra_meta

    # CAIL2019-SCM 格式
    if "A" in obj:
        return obj["A"], extra_meta

    # 通用格式
    for key in ("text", "content", "body"):
        if key in obj and obj[key]:
            return obj[key], extra_meta

    return "", extra_meta


def import_cases_jsonl(repository, dir_path: str, max_cases: int = 5000) -> int:
    """导入 JSONL 格式的案例文件

    自动识别多种格式（CAIL2019-SCM, CAIL2018, 通用JSONL）。
    """
    json_files = collect_files(dir_path, {".json", ".jsonl"})
    if not json_files:
        print("  无 JSON 文件")
        return 0

    total_chunks = 0
    case_count = 0

    for fpath in json_files:
        fname = os.path.basename(fpath)
        print(f"  处理 {fname}...")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if case_count >= max_cases:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text, extra_meta = _extract_case_text(obj)
                    if not text or len(text) < 50:
                        continue

                    tree = prepare_document(
                        text,
                        doc_type="case",
                        source_file=fname,
                        source_path=fpath,
                        dataset_name="cases",
                        source_record_id=str(line_no),
                    )
                    tree.document.document_metadata.update(extra_meta)
                    for parent in tree.document.parents:
                        parent.chunk_metadata.update(extra_meta)
                    persist_and_index(
                        tree,
                        collection_name=settings.CASES_COLLECTION,
                        repository=repository,
                    )
                    case_count += 1
                    total_chunks += len(tree.child_ids)
                    if case_count % 100 == 0:
                        print(f"    已导入 {case_count} 条案例 ({total_chunks} 个子块)")

                print(f"  ✓ {fname}: {case_count} 条案例, {total_chunks} 个分块")

        except Exception as e:
            print(f"  ✗ {fname}: {e}")

        if case_count >= max_cases:
            print(f"  已达上限 {max_cases} 条，停止导入")
            break

    return total_chunks


def main():
    print("=" * 55)
    print("LawRAG — 法律检索增强问答系统 — 数据导入")
    print("=" * 55)

    repository = get_parent_child_repository()
    repository.create_schema()

    # --- 法律条文 ---
    print(f"\n[1/2] 导入法律条文 ({settings.LAWS_DIR})")
    laws_count = import_laws(repository, settings.LAWS_DIR)

    # --- 案例 ---
    print(f"\n[2/2] 导入案例 ({settings.CASES_DIR})")
    cases_count = import_cases_jsonl(repository, settings.CASES_DIR, max_cases=5000)

    print(f"\n{'=' * 55}")
    print(f"导入完成: 法律子块 {laws_count} 块, 案例子块 {cases_count} 块")
    print(f"MySQL: {settings.MYSQL_URL.split('@')[-1]}")
    print(f"向量数据库: {settings.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
