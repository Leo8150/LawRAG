# LawRAG：法律检索增强问答系统

LawRAG 是一个面向中国法律咨询场景的检索增强生成（Retrieval-Augmented Generation，RAG）项目。系统将法律法规、裁判案例和犯罪知识整理为可检索知识库，在回答问题前先查找依据，再由大语言模型结合检索结果生成回答。

项目采用前后端分离架构：后端使用 FastAPI、LangChain 与 LangGraph 构建单一 Agentic RAG 状态图，LLM 在有界循环中选择本地法律检索、CrimeKG、来源回查以及外部法律 MCP 工具，并根据证据充分性动态补充检索；MySQL 保存原始文档及父子 Chunk，ChromaDB 仅保存 Child Chunk 向量索引。外部工具池连接国家法律法规数据库、人民法院案例库和 Tavily 官方站点搜索，DashScope 提供生成、向量、工具决策和可选重排模型，前端使用 React 展示答案、来源、Agent Trace 与评测指标。

> 一句话理解：LawRAG 不是让模型仅凭参数记忆回答法律问题，而是先从法律知识库寻找证据，再基于证据组织答案。

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构与技术选型](#2-系统架构与技术选型)
3. [数据准备与知识库构建](#3-数据准备与知识库构建)
4. [一次问答的完整执行流程](#4-一次问答的完整执行流程)
5. [系统实现与接口设计](#5-系统实现与接口设计)
6. [环境配置与项目运行](#6-环境配置与项目运行)
7. [测试评测与质量保障](#7-测试评测与质量保障)

---

## 1. 项目概述

**本章使用的核心技术**：RAG、领域知识库、法律文本结构化、混合检索。本章介绍项目目标与业务价值，其余章节展开具体技术实现。

### 1.1 为什么要做法律 RAG

通用大语言模型能够生成流畅的法律回答，但如果只依赖模型参数，仍然存在三个核心问题：

1. **知识不可控**：模型训练数据的时间范围不透明，可能不了解最新法规或仍引用失效条文；
2. **依据不可追溯**：答案看似合理，却无法说明结论来自哪部法律、哪一条规定或哪个案例；
3. **专业表达存在鸿沟**：用户常用“打人”“欠钱不还”“酒驾”等口语提问，而法律材料使用“故意伤害”“民间借贷”“危险驾驶”等正式术语。

RAG 将问题拆成“检索”和“生成”两部分：检索模块负责从受控知识库中找出证据，生成模块根据问题和证据组织回答。这样既使用了大模型的语言理解能力，也让答案尽量建立在明确资料之上。

### 1.2 项目目标

LawRAG 围绕法律文本特点实现以下能力：

- 以法律“编、章、节、条、款、项”结构进行分块，尽量保持法条语义完整；
- 对法律法规和裁判案例采用不同的分块策略；
- 使用 Document → Parent Chunk → Child Chunk 三级结构，兼顾召回精度和证据完整性；
- 以 MySQL 作为原文事实库，以 `child_chunk_id`、`parent_chunk_id` 关联 ChromaDB 派生索引；
- 结合 BM25 关键词检索与向量语义检索，提高精确匹配和语义召回能力；
- 使用 LangGraph 编排 Skill Loading、工具决策、证据评估、查询改写、生成和引用校验；
- 由 LLM 动态选择法律检索、CrimeKG 与原始来源回查工具，策略层强制至少执行一次权威知识检索；
- 证据不足时自动生成针对性查询并重新检索，所有循环均设置硬上限；
- 使用结构化犯罪知识补充罪名定义、构成要件和量刑信息；
- 在法律检索工具内部支持不重排、轻量重排和云端专用 Reranker；
- 使用带 TTL 的短期会话 Memory 理解“如果已经退赃呢”等连续追问；
- 根据问题领域按需加载刑事、劳动、合同、交通或通用法律 Skill；
- 通过 Budget、Snip、Micro、Summary 四阶段压缩 Tool Result 与最终证据上下文；
- 返回参考来源和各阶段耗时，便于分析系统行为；
- 提供知识库管理、问答配置、性能测试和报告导出页面。

### 1.3 数据规模

知识库采用三层统计口径：MySQL `documents` 记录原始文档，`parent_chunks` 记录进入生成阶段的完整法律证据，`child_chunks` 记录参与 BM25 和向量召回的细粒度片段。ChromaDB 的 `laws`、`cases` Collection 与 MySQL Child 一一对应，向量记录 ID 等于 `child_chunk_id`。`GET /api/knowledge/stats` 动态返回当前已导入的 Child 数量，不在文档中固化易失效的数据规模。

---

## 2. 系统架构与技术选型

**本章使用的核心技术**：React、FastAPI、Pydantic、LangChain、LangGraph、SQLAlchemy、MySQL、ChromaDB、BM25、DashScope。系统按表现层、接口层、Agentic 编排层、检索工具层和数据层分层，MySQL 是权威事实库，ChromaDB 是可重建的 Child 向量索引。

### 2.1 总体架构

```mermaid
flowchart TB
    USER[用户]

    subgraph FE[表现层 · frontend]
        CHATUI[智能问答页<br/>ChatPage.jsx]
        KBUI[知识库管理页<br/>KnowledgePage.jsx]
        PERFUI[性能监控页<br/>PerformancePage.jsx]
        CLIENT[Axios API Client<br/>services/api.js]
        CHATUI --> CLIENT
        KBUI --> CLIENT
        PERFUI --> CLIENT
    end

    subgraph API[接口层 · FastAPI]
        CHATAPI[问答接口<br/>api/chat.py]
        KBAPI[知识库接口<br/>api/knowledge.py]
        PERFAPI[评测接口<br/>api/performance.py]
        SCHEMA[Pydantic Schema<br/>models/schemas.py]
    end

    subgraph SERVICE[Agentic 编排层 · LangGraph]
        PIPE[AgenticRAGRunner<br/>agentic/runtime.py]
        STATE[AgenticRAGState<br/>agentic/state.py]
        ROUTER[LLM Tool Router]
        GRADER[Evidence Grader]
        REWRITE[Query Rewriter]
        VERIFY[Grounding Checker]
        MEMORY[短期会话 Memory<br/>memory/service.py]
        SKILL[Skill Catalog + load_skill Tool<br/>skills/loader.py / tool.py]
        COMPACT[四阶段 Context Compact<br/>context/compactor.py]
        PROMPT[Prompt 与生成<br/>prompts.py]
        QUALITY[RAG 评测与报告<br/>quality_service.py / report_service.py]
        PIPE --> STATE
        PIPE --> MEMORY
        PIPE --> SKILL
        PIPE --> ROUTER --> GRADER
        GRADER -->|证据不足| REWRITE --> ROUTER
        GRADER -->|证据充分| COMPACT --> PROMPT --> VERIFY
        VERIFY -->|未通过且未超限| PROMPT
        PIPE --> COMPACT
    end

    subgraph CORE[检索与模型能力层 · core]
        TOOLS[Agent Tool Registry<br/>检索 / CrimeKG / 来源回查]
        HYBRID[retrieve_legal_evidence]
        BM25[MySQL Child 语料<br/>BM25Okapi + jieba]
        VECTOR[Chroma Child 向量召回<br/>返回 child_chunk_id]
        RRF[Child ID 级 RRF 融合]
        HYDRATE[MySQL 批量回查 Child]
        CHILD_RERANK[Child Reranker]
        PARENT[按 parent_chunk_id 聚合<br/>回查 Parent]
        LLM[ChatOpenAI<br/>qwen-turbo]
        EMB[OpenAIEmbeddings<br/>text-embedding-v3]
        TOOLS --> HYBRID
        HYBRID --> BM25
        HYBRID --> VECTOR
        BM25 --> RRF
        VECTOR --> RRF
        RRF --> HYDRATE --> CHILD_RERANK --> PARENT
    end

    subgraph DATA[数据与存储层]
        RAW[法规 / 案例 / QA 原始数据]
        PREP[prepare_datasets.py<br/>清洗与格式转换]
        SPLIT[ParentChildChunker<br/>稳定 doc / parent / child ID]
        MYSQL_DOC[(MySQL documents<br/>原始全文)]
        MYSQL_PARENT[(MySQL parent_chunks<br/>完整证据)]
        MYSQL_CHILD[(MySQL child_chunks<br/>细粒度文本)]
        LAWS[(ChromaDB · laws<br/>Child vectors)]
        CASES[(ChromaDB · cases<br/>Child vectors)]
        KGDATA[(CrimeKG 结构化知识)]
        REPORTS[(评测报告 / 问答记录)]
        RAW --> PREP --> SPLIT
        SPLIT --> MYSQL_DOC
        SPLIT --> MYSQL_PARENT
        SPLIT --> MYSQL_CHILD
        MYSQL_CHILD --> EMB
        EMB --> LAWS
        EMB --> CASES
        PREP --> KGDATA
    end

    CLOUD[阿里云 DashScope]

    USER --> CHATUI
    USER --> KBUI
    USER --> PERFUI
    CLIENT -->|REST / JSON| CHATAPI
    CLIENT -->|REST / JSON| KBAPI
    CLIENT -->|REST / JSON| PERFAPI
    CHATAPI --> SCHEMA --> PIPE
    KBAPI --> SPLIT
    PERFAPI --> QUALITY
    ROUTER --> TOOLS
    PIPE --> QUALITY
    PARENT --> PIPE
    TOOLS --> KGDATA
    VECTOR --> LAWS
    VECTOR --> CASES
    PROMPT --> LLM
    LLM --> CLOUD
    EMB --> CLOUD
    QUALITY --> REPORTS
```

系统包含两条主链路：

- **离线建库链路**：原始数据 → Parent/Child 分块 → MySQL 持久化 → Child Embedding → ChromaDB；
- **在线问答链路**：问题 → Skill Loading → LLM 工具决策 → Child 混合召回与 Parent 回查 → 证据充分性判断 → 必要时改写并重试 → Tool Result 与最终上下文压缩 → 答案生成 → Grounding 校验。

离线建库只在首次导入或知识库变化时执行，在线问答则在每次用户提问时执行。

### 2.2 技术栈

| 层级 | 技术 | 作用 |
|---|---|---|
| 前端 | React 18、Vite、Axios、Recharts | 交互界面、接口调用、性能图表 |
| Web 后端 | FastAPI、Pydantic | REST API、参数校验、响应模型 |
| Agentic 编排 | LangGraph、LangChain | 有界状态图、工具决策、证据评估、检索纠错与 Grounding 校验 |
| 原文事实库 | MySQL 8、SQLAlchemy 2、PyMySQL | 保存 Documents、Parent Chunks、Child Chunks 和来源位置 |
| 上下文工程 | Budget、Snip、Micro、Summary | Token 预算、单文档裁剪、证据去重和抽取式摘要 |
| 会话记忆 | TTL Memory | 保存最近 6 轮问答、来源 ID 与当前 Skill，支持连续追问 |
| 领域能力 | YAML frontmatter、`load_skill`、ToolMessage | 技能目录进入 system prompt，完整 SKILL.md 按需进入 tool_result |
| 生成模型 | DashScope `qwen-turbo` | 查询改写、答案生成、可选评测 |
| 向量模型 | DashScope `text-embedding-v3` | 文本向量化和语义查询 |
| 向量数据库 | ChromaDB | 仅保存 Child Embedding、`child_chunk_id` 和过滤字段 |
| 云端重排 | DashScope `qwen3.7-text-rerank` | 对 MySQL 回查的 Child 文本精排，不重复重排 Parent |
| 稀疏检索 | `rank-bm25`、jieba | 中文分词和关键词匹配 |
| RAG 评测 | 人工相关文档标注、LLM Judge | Recall@5、MRR@10、P95 Latency、Faithfulness |
| 系统监控 | psutil | CPU 和内存使用率采集 |

### 2.3 为什么使用混合检索

法律问题同时需要“精确匹配”和“语义理解”。例如，“民法典第五百零九条”包含明确编号，BM25 更容易准确命中；“公司一直不给我发工资怎么办”与法条原文用词差异较大，向量检索更适合发现语义相关内容。

系统将 BM25 和向量检索的排名通过 RRF（Reciprocal Rank Fusion）合并。对排名为 `rank` 的文档，其贡献可简化表示为：

```text
RRF_score = weight / (60 + rank + 1)
```

同一文档如果被两种检索器同时命中，会累加两部分分数，从而获得更靠前的最终排名。默认 BM25 和向量检索权重均为 `0.5`。

---

## 3. 数据准备与知识库构建

**本章使用的核心技术**：Python 文件处理、PyArrow、JSON/JSONL、正则表达式、LangChain `RecursiveCharacterTextSplitter`、SQLAlchemy、MySQL、`OpenAIEmbeddings` 和 ChromaDB。目标是把多源原始数据转换为可追溯的 Document、Parent、Child 与 Child Vector。

### 3.1 数据来源

| 数据集 | 主要内容 | 在项目中的用途 |
|---|---|---|
| [Chinese-Laws](https://modelscope.cn/datasets/dengcao/Chinese-Laws) | 中国法律法规 TXT 文本 | 法律条文检索 |
| [Chinese Law and Regulations](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations) | 法律法规结构化数据 | 补充法规覆盖范围 |
| [CrimeKgAssitant](https://github.com/liuhuanyong/CrimeKgAssitant) | 罪名知识和法律问答 | 犯罪知识增强、参考答案 |
| [CAIL](https://github.com/thunlp/CAIL) | 中文法律案例数据 | 案例检索与测试 |
| CAIL2018 | 刑事案件事实与标签 | 构建案例 Collection |

原始数据统一存放在 `backend/data/` 下：

```text
backend/data/
├── laws/
│   ├── Chinese-Laws/
│   ├── HF_Chinese_Laws/
│   └── CrimeKG/
├── cases/
│   ├── CAIL2018/
│   └── CAIL2019-SCM/
├── qa/
├── reference/
├── reports/
└── chat_records/
```

项目数据分为两类：Chinese-Laws、HF 法规、CrimeKgAssitant 和 CAIL2018 用于构建业务知识库；CAIL2019 阅读理解与 `rag_eval_dataset.json` 用于构造检索评测问题和相关文档标注。

#### 3.1.1 Chinese-Laws：法律法规 TXT

每部法律对应一个 TXT 文件，正文按法条顺序排列。文件名用于识别法律名称，条文编号由分块器提取：

```text
文件：中华人民共和国反家庭暴力法.txt

《中华人民共和国反家庭暴力法》第一条规定，为了预防和制止家庭暴力……
《中华人民共和国反家庭暴力法》第二条规定，本法所称家庭暴力，是指……
```

导入后形成 `law_name`、`article_number`、`source_file` 等元数据；原文和父子块写入 MySQL，Child 向量写入 ChromaDB `laws` Collection。

#### 3.1.2 Chinese Law and Regulations：法规 Parquet

Parquet 中每行是一部法规。项目筛选状态为“有效/已修改”的国家级文件，再转换为独立 TXT：

```json
{
  "title": "中华人民共和国民法典",
  "publish_date": "2020-05-28",
  "effective_date": "2021-01-01",
  "status": "有效",
  "office": "全国人民代表大会",
  "office_level": "全国人民代表大会",
  "content": "第一条 为了保护民事主体的合法权益……"
}
```

其中 `title` 和 `content` 构成检索正文，其余字段用于过滤、溯源和元数据展示。

#### 3.1.3 CrimeKgAssitant：罪名知识与法律问答

`kg_crime.json` 采用 JSONL 格式，每行描述一个罪名。下面以“盗窃罪”为例展示核心结构：

```json
{
  "crime_big": "侵犯财产罪",
  "crime_small": "盗窃罪",
  "gainian": [
    "盗窃罪的概念和定义……"
  ],
  "tezheng": [
    "盗窃罪的犯罪构成特征……"
  ],
  "rending": [
    "盗窃罪与其他罪名的认定和区分……"
  ],
  "chufa": [
    "根据盗窃数额和犯罪情节确定处罚……"
  ],
  "fatiao": [
    "《中华人民共和国刑法》第二百六十四条……"
  ],
  "jieshi": [
    "关于办理盗窃刑事案件的相关司法解释……"
  ]
}
```

主要字段含义如下：

| 字段 | 含义 | 项目用途 |
|---|---|---|
| `crime_big` | 刑法罪名大类 | 分类与展示 |
| `crime_small` | 具体罪名 | 罪名识别和精确查找键 |
| `gainian` | 概念与定义 | 回答罪名含义 |
| `tezheng` | 犯罪构成特征 | 分析主体、客体、主观和客观要件 |
| `rending` | 罪名认定与区分 | 处理相近罪名边界 |
| `chufa` | 处罚规则 | 生成量刑说明 |
| `fatiao` | 关联刑法条文 | 提供法律依据 |
| `jieshi` | 司法解释 | 补充具体适用规则 |

预处理程序将数组字段展开为结构化罪名文本，输出到 `data/laws/CrimeKG/犯罪知识图谱.txt`。该文本一方面进入 `laws` Collection 参与混合检索，另一方面由 `kg_service.py` 加载为罪名索引，用于刑事问题的精确知识增强。

CrimeKgAssitant 的法律问答 `qa_corpus.json` 同样采用 JSONL：

```json
{
  "question": "交通事故发生后伤者住院，应如何处理？",
  "answers": ["治疗结束后可以整理证据并依法起诉。"],
  "category": "交通事故"
}
```

#### 3.1.4 CAIL2018：刑事案例

CAIL2018 每行是一份刑事案件及其裁判标签：

```json
{
  "fact": "被告人以非法占有为目的，秘密窃取他人财物……",
  "meta": {
    "accusation": ["盗窃"],
    "relevant_articles": ["264"],
    "criminals": ["张某"],
    "term_of_imprisonment": {
      "death_penalty": false,
      "life_imprisonment": false,
      "imprisonment": 12
    }
  }
}
```

预处理后保留案件事实、罪名、相关法条、被告人和刑期；案例 Parent / Child 写入 MySQL，Child 向量写入 ChromaDB `cases` Collection。

#### 3.1.5 CAIL2019 与 RAG 评测标注

CAIL2019-SCM 的核心数据是案件三元组，`A` 为查询案件，`B`、`C` 为候选案件：

```json
{
  "A": "查询案件的事实描述……",
  "B": "与 A 更相似的候选案件……",
  "C": "与 A 相似度较低的候选案件……"
}
```

CAIL2019 阅读理解子集采用类似 SQuAD 2.0 的结构，包含 `caseid`、案件原文、问题和标准答案，可转换为 `corpus + queries + qrels`。项目内部的检索标注文件使用更轻量的格式：

```json
{
  "id": "intentional_homicide_001",
  "category": "刑法",
  "question": "故意杀人罪的量刑标准是什么？",
  "relevant_documents": [
    {"law_name": "中华人民共和国刑法"}
  ]
}
```

`question` 作为评测查询，`relevant_documents` 表示标准相关文档，用于计算 Recall@5 和 MRR@10；实际测评时进一步使用稳定的 `doc_id` 关联固定评测语料。

### 3.2 数据预处理

运行 `scripts.prepare_datasets` 后，系统会执行以下转换：

1. 将 Hugging Face Parquet 法规筛选并转换为独立 TXT 文件；
2. 解压并读取 CAIL2018 数据，将案件事实和标签整理为 JSONL；
3. 从 CrimeKgAssitant 中提取罪名定义、构成要件、量刑和相关法条；
4. 整理评测问题及相关法律文档标注，生成 RAG 检索评测集；
5. 生成后续导入脚本能够统一读取的目录和文件格式。

该阶段执行确定性的本地文件转换，并支持按需下载 DISC-Law-SFT；模型调用统一集中在向量导入和在线问答阶段。

完整建库流程如下。图中的前半部分由 `prepare_datasets.py` 完成，后半部分由 `import_data.py` 完成：

```mermaid
flowchart TD
    A[多源原始数据] --> A1[TXT 法律法规]
    A --> A2[Parquet 法规数据]
    A --> A3[CAIL JSON / ZIP]
    A --> A4[CrimeKG / QA JSON]

    A1 --> B[统一读取与格式转换]
    A2 --> B
    A3 --> B
    A4 --> B
    B --> C{文档类型判断}

    C -->|law| D1[生成 Parent<br/>完整法条或法律结构单元]
    C -->|case| D2[生成 Parent<br/>案情、裁判理由或结果]
    C -->|crime knowledge| D3[解析罪名、定义、构成要件<br/>量刑和相关法条]

    D1 --> E[正则提取元数据]
    D2 --> E
    D3 --> E
    E --> F[生成稳定 doc_id<br/>parent_chunk_id]
    F --> G[Parent 切分为 Child<br/>生成 child_chunk_id]
    G --> M1[(MySQL documents)]
    G --> M2[(MySQL parent_chunks)]
    G --> M3[(MySQL child_chunks)]
    M3 --> H[Child 注入上下文标头<br/>调用 text-embedding-v3]
    H --> I{目标 Collection}
    I -->|法规与犯罪知识| J[(ChromaDB · laws)]
    I -->|裁判案例| K[(ChromaDB · cases)]
    J --> L[只保存 Child 向量<br/>vector ID = child_chunk_id]
    K --> L
```

这条链路体现了四个技术层次：格式转换解决数据源不统一，Parent 保留法律证据完整性，Child 提高检索粒度，MySQL 与 ChromaDB 则分别承担事实存储和派生向量索引。

### 3.3 法律结构化分块

普通 RAG 常按固定字符或 Token 窗口切分文档。但如果在法条中间直接切开，适用条件与法律后果可能被分到两个 Chunk，检索后只能得到半条规则。

LawRAG 为两类文本分别设计 Parent，再将每个 Parent 切为更小的 Child：

| 文档类型 | Parent 单位 | Child 参数 |
|---|---|---|
| 法律法规 | 完整法条；超长法条按款项形成父级结构单元 | 320 字符，64 字符重叠 |
| 裁判案例 | 裁判要旨、基本案情、裁判理由、裁判结果 | 320 字符，64 字符重叠 |
| CrimeKG | 一个罪名知识条目 | 概念、构成、认定、处罚和法条片段 |

`doc_id`、`parent_chunk_id` 和 `child_chunk_id` 由源数据哈希、分块器版本和序号通过 UUIDv5 确定性生成。重复执行相同导入会得到相同 ID；升级 `CHUNKER_VERSION` 后生成新索引版本。Parent 保存法律名称、条号、案号和原文位置，Child 保存其所属 Parent ID 以及自身字符范围。

### 3.4 上下文标头注入

单独看一句“当事人应当按照约定全面履行自己的义务”，Embedding 模型并不知道它属于哪部法律。为减少 Chunk 脱离上下文的问题，系统会在向量化前加入结构化标头：

```text
[法律名称: 中华人民共和国民法典 | 章节: 第四章 合同的履行 | 条号: 第509条]
当事人应当按照约定全面履行自己的义务……
```

标头来自文档结构和正则提取结果，不需要额外调用 LLM。

### 3.5 MySQL持久化与Child向量入库

`scripts.import_data` 负责最终导入：

```text
读取本地文件
  → 尝试 UTF-8 / GBK / GB2312 编码
  → 判断法规或案例类型
  → 构建 Document / Parent / Child
  → 先写入 MySQL 事实库
  → 仅对 Child 批量请求 text-embedding-v3
  → 以 child_chunk_id 写入 laws 或 cases Collection
  → 将 Document 状态更新为 indexed
```

MySQL 写入与向量索引之间通过 `pending → chunked → indexed / failed` 状态衔接。向量阶段失败时，原始数据和父子块仍保留在 MySQL，可根据稳定 ID 幂等重试。ChromaDB 中即使保存了 Child 文本派生副本，在线检索仍根据返回的 `child_chunk_id` 批量回查 MySQL，不把向量库副本作为权威正文。

> 资源说明：`python -m scripts.import_data` 会为全部 Chunk 请求 Embedding。索引与 Embedding 模型版本绑定，数据或模型发生变化时执行重建。

---

## 4. 一次问答的完整执行流程

**本章使用的核心技术**：LangGraph `StateGraph`、LangChain Tool Calling、FastAPI、短期 Memory、Skill Loading、父子检索、证据充分性判断、四阶段 Context Compact、Grounding Check 和有界自纠错循环。

用户提交问题后，请求进入唯一的 `AgenticRAGRunner`。系统不提供传统 RAG 分支：每次问答都经过 LangGraph 状态图，由 LLM 选择高层工具，策略层强制至少执行一次权威知识检索，并根据证据评估结果决定生成答案或改写查询后再次检索。

```mermaid
flowchart TD
    A([用户提交问题]) --> B[POST /api/chat<br/>Pydantic 校验 ChatRequest]
    B --> B1[按 conversation_id 加载短期 Memory]
    B1 --> B2[识别追问并补全当前问题]
    B2 --> B3[system prompt 注入技能目录<br/>仅 name + description]
    B3 --> B4[LLM 调用 load_skill name]
    B4 --> B5[完整 SKILL.md 作为 tool_result<br/>追加到 messages]
    B5 --> C[LLM Tool Router]
    C --> C1{选择高层工具}
    C1 --> T1[retrieve_legal_evidence]
    C1 --> T2[lookup_crime_knowledge]
    C1 --> T3[get_original_source]
    C1 --> T4[国家法律法规数据库 MCP]
    C1 --> T5[人民法院案例库 MCP]
    C1 --> T6[Tavily 官方域名搜索 MCP]
    C1 -.未选择检索.-> GUARD[策略层补充强制检索]
    GUARD --> T1

    T1 --> E1[MySQL Child BM25]
    T1 --> E2[ChromaDB Child 向量召回]
    E1 --> G[child_chunk_id 级 RRF]
    E2 --> G
    G --> H[MySQL 回查 Child 权威文本]
    H --> K[Child Reranker]
    K --> PA[parent_chunk_id 聚合]
    PA --> PH[MySQL 回查完整 Parent]
    T2 --> TR[四阶段压缩 Tool Result]
    T3 --> TR
    T4 --> TR
    T5 --> TR
    T6 --> TR
    PH --> TR
    TR --> TM[ToolMessage 写回状态图]
    TM --> EG{Evidence Grader}
    EG -->|证据不足且未超限| RW[生成针对性补充查询]
    RW --> C
    EG -->|证据充分或达到上限| L0[Budget<br/>计算最终证据预算]
    L0 --> L1[Snip<br/>限制单篇文档占用]
    L1 --> L2[Micro<br/>去重、排序与证据合并]
    L2 --> L3[Summary<br/>案例与 KG 抽取式摘要]

    L3 --> M[LLM 生成法律回答]
    M --> N{Grounding 与引用检查}
    N -->|未通过且未超限| M
    N -->|通过或达到上限| O[形成最终答案]
    O --> O1[写入最近 6 轮 Memory<br/>刷新 24 小时 TTL]

    O1 --> P{是否开启 RAG 评测}
    P -->|是| P1["Recall@5 + MRR@10<br/>P95 Latency + Faithfulness"]
    P -->|否| Q[组装 ChatResponse]
    P1 --> Q
    Q --> R[返回答案、来源、Agent Trace<br/>工具与各节点耗时]
    R --> S([React 渲染回答与监控信息])
```

每次问答同时使用 MySQL 事实库与本地 ChromaDB 派生索引。`collection` 决定 Child 的数据范围，两个存储通过相同的 `child_chunk_id`、`parent_chunk_id` 和 `doc_id` 关联：

| `collection` 参数 | ChromaDB Child Collection | MySQL 权威数据范围 |
|---|---|---|
| `all` | `laws` + `cases` | 同时读取法规与案例 Child，聚合后回查对应 Parent |
| `laws` | `laws` | 法律法规、司法解释和犯罪结构化知识的 Child / Parent |
| `cases` | `cases` | 裁判案例与指导案例的 Child / Parent |

`retrieve_legal_evidence` 是一个高层检索工具。LLM 只决定检索 Query、`laws/cases/all` 范围和 Top K，工具内部固定执行 MySQL BM25、ChromaDB Child 向量召回、RRF、MySQL Child 回查、Child Reranker、Parent 聚合与 MySQL Parent 回查。底层步骤不会暴露给 LLM，避免其跳过来源恢复或任意拼装检索链路。

流程图中的每个阶段都可以定位到具体实现：

| 阶段 | 采用技术 | 主要源码 | 是否调用云模型 |
|---|---|---|---|
| 请求接收 | FastAPI、Pydantic | `api/chat.py`、`models/schemas.py` | 否 |
| Agent 状态图 | LangGraph、TypedDict、条件边 | `agentic/runtime.py`、`agentic/state.py` | 否 |
| 短期记忆 | TTL、最近轮次窗口、追问补全 | `memory/service.py` | 否 |
| Skill 路由与加载 | frontmatter 目录、Tool Calling、`load_skill`、ToolMessage | `skills/loader.py`、`skills/tool.py`、`backend/skills/` | `auto` 模式调用 1 次 |
| 工具决策 | LangChain `bind_tools`、工具白名单、参数约束 | `agentic/tools.py`、`agentic/runtime.py` | 每个工具轮次 1 次 |
| 证据充分性判断 | JSON 结构化判断、缺失信息提取 | `agentic/runtime.py`、`agentic/prompts.py` | 每轮检索后 1 次 |
| 自适应查询改写 | 根据缺失证据生成针对性补充查询 | `agentic/runtime.py` | 必要时 1 次 |
| BM25 召回 | MySQL Child 语料、jieba、`rank_bm25.BM25Okapi` | `core/retriever.py`、`db/repository.py` | 否 |
| 向量召回 | `OpenAIEmbeddings`；ChromaDB 返回 Child ID 与相似度 | `core/embeddings.py`、`core/vectorstore.py` | 每个向量查询文本需要一次 Embedding |
| Child 融合与回查 | 按 `child_chunk_id` 执行 RRF、去重，批量回查 MySQL Child | `core/retriever.py`、`db/repository.py` | 否 |
| 犯罪知识工具 | `lookup_crime_knowledge`、本地 CrimeKG 精确查询 | `agentic/tools.py`、`services/kg_service.py` | 工具执行本身不调用模型 |
| 外部法律 MCP | MCP Streamable HTTP、工具白名单、个人信息脱敏、官方域名限制 | `mcp/client.py`、`mcp/legal_tools.py` | 仅 LLM 选择外部工具时调用 |
| Child 轻量重排序 | Jaccard、jieba、法律元数据加权 | `services/reranker.py` | 否 |
| Child 云端重排序 | HTTPX、`qwen3.7-text-rerank` | `services/cloud_reranker.py` | 仅选择 `cloud` 时调用 |
| Parent 聚合与回查 | `parent_chunk_id` 分组、最高 Child 分数、命中数加成、MySQL 批量查询 | `services/pipeline.py`、`db/repository.py` | 否 |
| Tool Result 与最终上下文压缩 | Budget、Snip、Micro、Summary、来源格式化 | `context/compactor.py` | 否 |
| 答案生成 | LCEL `prompt \| llm`、Qwen | `services/prompts.py`、`core/llm.py` | 是 |
| Grounding 校验 | 证据支持判断、引用完整性检查、有界重新生成 | `agentic/runtime.py`、`agentic/prompts.py` | 每次生成后 1 次 |
| RAG 评测 | 相关文档标注、排名指标、LLM-as-a-Judge | `services/quality_service.py`、`services/perf_service.py` | 检索和延迟指标否，Faithfulness 是 |
| 响应展示 | Pydantic、Axios、React Markdown | `models/schemas.py`、`ChatPage.jsx` | 否 |

### 4.1 短期 Memory 与 Skill Loading

前端在会话开始时生成一个 `conversation_id`，后续问题复用同一标识。后端短期 Memory 保存最近 6 轮问题、答案摘要、来源 ID 和实际 Skill，并在每次访问时刷新 24 小时 TTL。系统发现“如果已经退赃呢”“那未成年人呢”等追问后，将上一轮核心问题与当前问题组合为可独立检索的 `resolved_question`；新会话不会读取其他会话内容。

```mermaid
flowchart LR
    subgraph START[服务启动]
        A[扫描 skills/*/SKILL.md] --> B[解析 YAML frontmatter]
        B --> C[name + description 技能目录]
        C --> D[注入 system prompt]
    end
    subgraph RUNTIME[请求运行]
        E[LLM 查看技能目录] --> F[调用 load_skill name]
        F --> G[读取选中的完整 SKILL.md]
        G --> H[以 tool_result 追加到 messages]
        H --> I[后续 RAG 与答案生成]
    end
    D -.每次请求使用目录.-> E
```

项目内置五个领域 Skill：

| Skill | 典型问题 | 加载后的能力 |
|---|---|---|
| `general_legal` | 未明确分类的一般法律问题 | 通用法律关系分析，法规优先 |
| `criminal_law` | 盗窃、诈骗、伤害、量刑 | 自动启用 CrimeKG 本地精确匹配，优先罪名结构与刑法依据 |
| `labor_law` | 工资、辞退、工伤、仲裁 | 强调劳动关系、仲裁时效和举证责任 |
| `contract_law` | 借款、买卖、租赁、违约 | 强调合同效力、履行、解除和损失范围 |
| `traffic_law` | 交通事故、酒驾、保险赔偿 | 区分行政、民事与刑事责任 |

每个 `SKILL.md` 使用 YAML frontmatter 声明 `name` 和 `description`。服务启动时 `SkillLoader.scan()` 只读取这两项元数据，完整正文保持未加载；`skill_name=auto` 时，模型根据 system prompt 中的目录调用 `load_skill(name)`，Loader 才读取对应的完整 `SKILL.md`，并以 `ToolMessage` 追加到消息序列。下一次模型调用同时看到技能目录、`tool_result`、检索证据和当前问题。手动指定 `skill_name` 时跳过自动选择调用，但仍通过相同的 `load_skill → tool_result → messages` 路径加载正文。

自动选择 Skill 会增加 1 次轻量 LLM 调用。完整 Skill 进入状态图后，Tool Router 结合领域规则决定是否调用 CrimeKG；CrimeKG 工具本身只执行本地精确查询，不产生嵌套模型调用。

### 4.2 工具决策与检索纠错

Agent 只看到稳定的高层工具，底层 BM25、向量检索、重排、数据库回查和第三方 MCP 原始工具不会直接暴露：

| 工具 | 输入 | 返回结果 |
|---|---|---|
| `retrieve_legal_evidence` | Query、`laws/cases/all`、Top K | Child 精排后恢复的完整 Parent 证据 |
| `lookup_crime_knowledge` | 罪名 | 概念、构成要件、认定、处罚和关联法条 |
| `get_original_source` | Parent 或 Child ID | MySQL 中的完整来源和原始位置 |
| `search_statutes` | Query、标题/全文范围、效力状态、Top K | 国家法律法规数据库中的法规列表与 `regulation_id` |
| `get_statute_articles` | `regulation_id`、关键词 | 法规中直接命中的具体条文 |
| `search_court_cases` | Query、Top K | 人民法院案例库中的指导性案例与参考案例 |
| `get_court_case` | `case_id`、章节列表 | 裁判要点、基本案情、裁判理由和关联法条 |
| `search_official_legal_web` | Query、Top K | 限定官方域名的最新法律网页结果 |
| `fetch_official_legal_page` | 官方网页 URL | 可作为回答证据的网页正文 |

LLM 使用 `tool_choice=auto` 选择工具和参数。若首轮没有选择 `retrieve_legal_evidence`，策略层自动补充该调用，从而保证法律答案一定先经过本地知识库检索。本地证据缺失、需要核验效力状态或问题涉及最新政策时，LLM 再选择相应的外部 MCP。所有外部结果以 `Document` 进入同一 Evidence Grader、四阶段压缩和 Grounding 流程。工具轮次最多 3 轮、法律检索最多 2 轮，达到上限后使用现有证据继续生成，不会无限循环。

### 4.3 混合检索

混合检索围绕 Child 展开：MySQL 保存可参与关键词匹配的 Child 权威正文，ChromaDB 保存同一批 Child 的向量及稳定 ID。收到查询后：

1. BM25 在 MySQL Child 语料构建的内存索引中返回关键词排名；
2. ChromaDB 返回向量相似度排名和 `child_chunk_id`；
3. RRF 按 `child_chunk_id` 累加两路排名分数；
4. 多查询结果仍按 `child_chunk_id` 合并，避免同一 Child 重复出现；
5. 根据候选 ID 一次批量回查 MySQL，得到用于重排序的 Child 原文与完整元数据。

因此 ChromaDB 可以随时由 MySQL 数据重建，在线答案不会依赖向量库中的正文副本。用户可以只检索法规、只检索案例，或者同时检索两个 Collection。

### 4.4 CrimeKG 工具

Tool Router 判断问题需要罪名结构化知识时调用 `lookup_crime_knowledge`，在 CrimeKG 中精确查找定义、构成要件、量刑和相关法条。命中结果作为高优先级证据并入 Agent 状态。

犯罪知识模块采用轻量结构化查找：将罪名映射到定义、构成要件、量刑和关联法条，以常数时间完成精确查询，省去独立图数据库的部署与维护成本。

### 4.5 重排序

初步召回强调“尽量找到”，重排序强调“从细粒度 Child 中选出与 Query 最相关的命中”：

- `none`：直接截取前 `top_k` 条；
- `simple`：根据词项重合度和法律名称、条号、案号等元数据加权；
- `cloud`：将 Query 与回查后的 Child 原文批量提交给 `qwen3.7-text-rerank`，根据 `relevance_score` 取 Top K。

三种策略分别承担实验基线、零云调用轻量排序和云端精排职责。管线先保留 20～50 个 Child 候选，默认精排到 15 个 Child。随后按 `parent_chunk_id` 分组，使用“组内最高 Child 重排分数 + 封顶的多 Child 命中加成”计算 Parent 分数：

```text
parent_score = max(child_rerank_score) + min(hit_count - 1, 3) × 0.02
```

系统按 Parent 分数选择 Top 5，并批量回查 MySQL `parent_chunks` 恢复完整法条、案件事实或知识段落。Parent 不再进入 Reranker：相关性判断由更聚焦的 Child 完成，Parent 只承担证据扩展与上下文完整性，避免长文本稀释排序信号和产生第二次云端费用。`cloud` 异常时自动切换到 `simple`，并通过 `rerank_fallback` 记录实际执行路径。

### 4.6 四阶段上下文压缩与答案生成

四阶段压缩执行两次：第一次把检索、CrimeKG 和来源回查产生的大型 Tool Result 压缩到默认 1,800 Token 后写入 `ToolMessage`；第二次将 Agent 累积的 Parent 与 KG 证据压缩到默认 4,000 Token，供答案生成使用。整个压缩过程使用确定性算法，不调用模型：

| 阶段 | 处理方式 | 法律场景约束 |
|---|---|---|
| Budget | 扣除问题、短期记忆和 Skill 指令占用，计算剩余证据预算 | 为最终答案预留上下文空间 |
| Snip | 为法条、案例和 KG 设置单文档上限 | 法条只做原文截取，不改写法律原意 |
| Micro | 内容归一化去重，按 Skill 的证据优先级和重排分数排序 | 防止同一法条或案例重复占用预算 |
| Summary | 超出预算时对案例和 KG 做抽取式摘要 | 摘要仅选取原句；法律条文仍保持抽取式裁剪 |

压缩后的每段证据保留 `[来源N]` 标签和原始元数据。接口返回 `tokens_before`、`tokens_after`、`token_budget` 和实际文档数，便于观察压缩效果。

| 生成策略 | 输出特点 |
|---|---|
| `standard` | 基于资料直接回答 |
| `structured_legal` | 输出法律结论、适用依据、分析和注意事项 |
| `self_reflect` | 保留兼容的生成风格；所有输出仍统一经过 Graph Grounding 校验 |

三种策略分别承担快速基线、法律场景结构化输出和高质量纠错职责。复杂法律分析统一由 `structured_legal` 输出结论、依据和详细分析，避免重复的生成路径。

答案生成后，Grounding Checker 判断法律结论是否得到证据支持、关键结论是否具有来源；未通过时最多重新生成一次。最终响应包括 `agent_trace`、工具轮次、检索轮次、Grounding 状态、当前 Skill、上下文压缩统计、来源文档和各节点耗时。

### 4.7 贯穿示例：入室盗窃如何认定和处罚

下面以用户问题“进入他人住宅盗窃财物，数额不大，会构成盗窃罪吗？”为例，将离线知识库构建与在线 RAG 问答连接起来：

```mermaid
flowchart LR
    subgraph OFFLINE[离线知识库构建]
        A1[刑法第二百六十四条 TXT] --> B1[法条结构化分块]
        A2[CrimeKG 盗窃罪 JSON] --> B2[展开概念、构成要件、处罚和法条]
        A3[CAIL2018 盗窃案例 JSONL] --> B3[案例事实与裁判标签分块]
        B1 --> PC[生成 Document、Parent、Child<br/>及稳定 ID]
        B2 --> PC
        B3 --> PC
        PC --> M1[(MySQL documents)]
        PC --> M2[(MySQL parent_chunks)]
        PC --> M3[(MySQL child_chunks)]
        PC --> C1[text-embedding-v3<br/>仅编码 Child]
        C1 --> D1[(ChromaDB laws<br/>Child ID + Vector)]
        C1 --> D2[(ChromaDB cases<br/>Child ID + Vector)]
        B2 --> D3[CrimeKG 罪名内存索引]
    end

    subgraph ONLINE[在线 Agentic RAG]
        Q[用户问题<br/>入室盗窃数额不大会构成犯罪吗] --> MEM[加载短期 Memory]
        MEM --> CAT[system prompt 提供 Skill 目录]
        CAT --> CALL[LLM 调用 load_skill criminal_law]
        CALL --> SK[完整 SKILL.md 进入 tool_result]
        SK --> T[LLM Tool Router]
        T --> RT[调用 retrieve_legal_evidence<br/>laws + cases]
        T --> KG[调用 lookup_crime_knowledge<br/>盗窃罪]
        RT --> V[ChromaDB 向量召回<br/>Child ID]
        RT --> BM[MySQL Child 语料<br/>BM25 检索]
        V --> R[按 child_chunk_id<br/>RRF 融合]
        BM --> R
        R --> CH[MySQL 批量回查 Child 原文]
        CH --> RR[仅重排序 Child]
        RR --> PA[按 parent_chunk_id 聚合]
        PA --> PH[MySQL 回查 Top Parent]
        PH --> TR[压缩为 ToolMessage]
        KG --> TR
        TR --> EG{证据是否充分}
        EG -->|否| RW[改写检索 Query]
        RW --> T
        EG -->|是| CTX[Budget → Snip → Micro → Summary]
        CTX --> GEN[Qwen 生成结构化法律回答]
        GEN --> GC{Grounding 与引用校验}
        GC -->|未通过且未超限| GEN
        GC -->|通过| SAVE[更新短期 Memory]
        SAVE --> OUT[返回结论、法条、分析<br/>来源和阶段指标]
    end

    D1 -.法规 Child 向量.-> V
    D2 -.案例 Child 向量.-> V
    M3 -.权威 Child 正文.-> BM
    M3 -.按 ID 回查.-> CH
    M2 -.恢复完整证据.-> PH
    D3 -.精确罪名知识.-> KG
```

具体执行过程如下：

1. **离线整理法规**：刑法第二百六十四条按法条边界形成 Parent，再切成适合召回的 Child，并保留 `law_name=中华人民共和国刑法`、`article_number=二百六十四` 等元数据；
2. **离线整理罪名知识**：CrimeKG 中“盗窃罪”的 `gainian`、`tezheng`、`chufa`、`fatiao` 等字段被展开为可检索文本，同时建立以“盗窃罪”为键的内存索引；
3. **离线整理案例**：CAIL2018 中罪名为盗窃、关联法条为264的案件被转换为案例文档，保留案件事实、罪名、刑期和来源信息；
4. **双库存储**：原文、Parent 和 Child 先写入 MySQL；仅 Child 调用 `text-embedding-v3`，法规与 CrimeKG Child 写入 ChromaDB `laws`，案例 Child 写入 `cases`，向量记录 ID 等于 `child_chunk_id`；
5. **恢复会话并加载 Skill**：系统按 `conversation_id` 读取最近问答；模型选择 `criminal_law` 并调用 `load_skill`，完整刑事规则以 `ToolMessage` 进入 Agent 状态；
6. **Agent 工具决策**：Tool Router 选择 `retrieve_legal_evidence(query, all, 5)`，并为罪名结构调用 `lookup_crime_knowledge(盗窃罪)`；策略层确保法律检索不会被跳过；
7. **检索工具执行**：问题 Embedding 查询 ChromaDB 得到 Child ID，BM25 在 MySQL Child 中匹配关键词，RRF 融合后回查 Child 原文、完成 Child 精排并恢复完整 Parent；
8. **证据评估与纠错检索**：大型工具结果先经过四阶段压缩再写回消息；Evidence Grader 判断法条、构成要件和案例是否足够，不足时生成针对性 Query 并在有界循环中补充检索；
9. **基于证据生成与校验**：最终证据再次压缩，生成模型组织结构化答案；Grounding Checker 检查结论和引用，未通过时最多重新生成一次；
10. **更新记忆与评测**：答案、问题和来源 ID 写回短期 Memory；接口返回 Agent Trace、工具与检索轮次、Grounding 状态、来源及阶段耗时，并计算 Recall@5、MRR@10、P95 Latency 和 Faithfulness。

该问题最终使用的证据上下文示意如下：

```text
[来源1] 中华人民共和国刑法 / 第二百六十四条
盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃……处三年以下有期徒刑……

[来源2] CrimeKG / 盗窃罪 / 犯罪构成与处罚
入户盗窃属于刑法列举的盗窃行为类型……

[来源3] CAIL2018 / 盗窃案例
案件事实、关联罪名、第二百六十四条及裁判结果……
```

模型据此说明：入户盗窃属于刑法明确列举的盗窃行为类型，是否构成犯罪不能只看普通盗窃的数额标准，还需要结合行为方式、证据和具体案情判断；回答中的法律结论和条文引用均可回溯到返回的来源文档。

---

## 5. 系统实现与接口设计

**本章使用的核心技术**：FastAPI Router、Pydantic BaseModel、REST/JSON、React Hooks、Axios、React Markdown 和 Recharts。后端负责管线与数据，前端负责配置、展示和操作，两者通过稳定的数据模型解耦。

### 5.1 后端分层

```text
backend/
├── app/
│   ├── agentic/             # LangGraph 状态、节点、Prompt 与高层工具注册表
│   ├── api/                 # 问答、知识库、来源追溯、性能接口
│   ├── context/             # Budget、Snip、Micro、Summary 四阶段压缩
│   ├── core/                # LLM、Embedding、ChromaDB、检索器
│   ├── db/                  # SQLAlchemy 表模型与 Parent/Child Repository
│   ├── memory/              # 带 TTL 的短期会话记忆
│   ├── models/              # Pydantic 请求与响应模型
│   ├── services/            # 导入、检索执行与生成能力
│   ├── skills/              # frontmatter 扫描器、按需加载器与 load_skill 工具
│   ├── utils/               # 法律分块、Parent/Child 分块和元数据工具
│   ├── config.py            # 环境变量与默认配置
│   └── main.py              # FastAPI 应用入口
├── skills/                  # 各法律领域的 SKILL.md 与 config.json
├── scripts/                 # 数据准备、导入和评测脚本
├── tests/                   # 单元测试
├── data/                    # 本地数据和运行产物
└── chroma_db/               # 本地向量数据库
```

`services/ingestion_service.py` 负责“先写 MySQL、再建 Child 向量索引”的离线链路；`agentic/runtime.py` 是唯一在线问答入口，使用 LangGraph 串联 Memory、Skill Loading、工具决策、证据评估、自适应改写、上下文压缩、生成与 Grounding 校验。`services/pipeline.py` 提供检索、重排、Parent 聚合和生成等底层能力，不再暴露传统问答分支。

### 5.2 主要 API

后端默认运行在 `http://127.0.0.1:8000`，接口统一使用 `/api` 前缀。启动后可访问 `/docs` 查看 Swagger 文档。

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/` | 项目基本信息 |
| `GET` | `/health` | 服务和模型配置状态 |
| `POST` | `/api/chat` | 执行可配置 RAG 问答 |
| `DELETE` | `/api/chat/memory/{conversation_id}` | 清除指定会话的短期记忆 |
| `POST` | `/api/chat/save-record` | 保存问答记录 |
| `GET` | `/api/chat/records` | 查询问答记录 |
| `POST` | `/api/knowledge/upload` | 上传法规或案例文件 |
| `GET` | `/api/knowledge/list` | 列出知识库文件 |
| `DELETE` | `/api/knowledge/{filename}` | 删除指定文件 |
| `POST` | `/api/knowledge/rebuild` | 重建向量索引 |
| `GET` | `/api/knowledge/stats` | 查询文件和 Chunk 数量 |
| `GET` | `/api/sources/chunks/{chunk_id}` | 根据 Child 或 Parent ID 查询正文、位置和所属文档 |
| `GET` | `/api/sources/documents/{doc_id}` | 查询原始文档全文、元数据及父子块结构 |
| `GET` | `/api/performance/system` | 查询 CPU 和内存状态 |
| `POST` | `/api/performance/bench` | 执行性能或质量测试 |
| `POST` | `/api/performance/report` | 生成并保存测试报告 |
| `GET` | `/api/performance/reports` | 查询历史报告 |

基础问答请求示例：

```json
{
  "question": "公司拖欠工资三个月，员工应该如何维权？",
  "collection": "all",
  "rerank_strategy": "simple",
  "generation_strategy": "structured_legal",
  "conversation_id": "7ed4fef8aab44f20831b78495cebd8f2",
  "skill_name": "auto",
  "top_k": 5,
  "evaluate_quality": false
}
```

### 5.3 前端页面

- **智能问答**：复用同一 `conversation_id` 完成多轮追问，可自动或手动选择领域 Skill，并查看回答、来源及阶段耗时；
- **知识库管理**：查看文件与 Chunk 数量，上传 TXT、Markdown 或 JSON 文件，删除文件并重建索引；
- **性能监控**：查看 CPU/内存，执行基准测试，展示延迟分解和质量指标，导出历史报告。

Vite 开发服务器默认运行在 `http://127.0.0.1:5173`，并将 `/api` 请求代理到后端 `8000` 端口。

---

## 6. 环境配置与项目运行

**本章使用的核心技术**：Python `venv`、pip、Node.js、npm、Uvicorn、Vite、`pydantic-settings` 和环境变量。模型密钥与代码分离，后端与前端分别运行并通过开发代理通信。

### 6.1 环境要求

- Python 3.10 或更高版本；
- Node.js 18 或更高版本；
- MySQL 8.0 或更高版本，字符集使用 `utf8mb4`；
- 可访问 DashScope 的网络环境和 API Key；
- 8 GB 以上磁盘空间，用于保存数据、依赖和向量索引。

模型层使用 `qwen-turbo`、`text-embedding-v3`、`qwen3.7-text-rerank` 和 DashScope API。

### 6.2 安装后端

```powershell
cd "D:\LLM study\LawRAG\LawRAG\backend"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="sk-your-api-key"
```

也可以在 `backend/.env` 中配置：

```env
DASHSCOPE_API_KEY=sk-your-api-key
DASHSCOPE_WORKSPACE_ID=your-workspace-id
LLM_MODEL=qwen-turbo
EMBEDDING_MODEL=text-embedding-v3
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RERANKER_MODEL=qwen3.7-text-rerank
RERANKER_CANDIDATE_K=20
RERANKER_DOCUMENT_MAX_CHARS=1200
CONTEXT_MAX_TOKENS=4000
MEMORY_TTL_SECONDS=86400
MEMORY_MAX_TURNS=6
AGENT_MAX_TOOL_ROUNDS=3
AGENT_MAX_RETRIEVAL_ROUNDS=2
AGENT_MAX_GENERATION_ROUNDS=2
AGENT_TOOL_RESULT_MAX_TOKENS=1800
EXTERNAL_MCP_ENABLED=true
FLK_MCP_ENABLED=true
FLK_MCP_URL=http://127.0.0.1:18062/mcp
RMFYALK_MCP_ENABLED=true
RMFYALK_MCP_URL=http://127.0.0.1:18061/mcp
TAVILY_MCP_ENABLED=true
TAVILY_MCP_URL=https://mcp.tavily.com/mcp
TAVILY_API_KEY=tvly-your-api-key
TAVILY_LEGAL_DOMAINS=["flk.npc.gov.cn","court.gov.cn","spp.gov.cn","gov.cn","moj.gov.cn"]
MYSQL_URL=mysql+pymysql://root:your-password@127.0.0.1:3306/lawrag?charset=utf8mb4
MYSQL_ECHO=false
CHUNKER_VERSION=parent-child-v1
CHILD_CHUNK_SIZE=320
CHILD_CHUNK_OVERLAP=64
CHILD_RERANK_TOP_K=15
PARENT_TOP_K=5
```

复制 `backend/.env.example` 为 `backend/.env` 并填写环境变量。真实密钥仅保存在 `.env`，该文件已被 Git 忽略。云端 Reranker 使用带业务空间 ID 的独立文本排序 Endpoint，Chat 和 Embedding 使用 OpenAI 兼容地址。`TAVILY_API_KEY` 留空时，Tavily 的两个工具不会注册到 LLM 工具池，也不会产生 Tavily 调用。

### 6.3 安装并启动法律 MCP

项目通过官方 MCP Python SDK 连接 Streamable HTTP 服务。安装脚本把社区法律 MCP 下载到已忽略的 `backend/external/`，不会把第三方源码提交到本项目：

```powershell
cd "D:\LLM study\LawRAG\LawRAG\backend"
.\.venv\Scripts\Activate.ps1
.\scripts\setup_legal_mcps.ps1
.\scripts\start_legal_mcps.ps1
python -m scripts.check_mcp_servers
```

启动后使用以下端点：

| 服务 | MCP 地址 | 本项目开放的工具 | 认证 |
|---|---|---|---|
| 国家法律法规数据库 MCP | `http://127.0.0.1:18062/mcp` | `search_statutes`、`get_statute_articles` | 无 |
| 人民法院案例库 MCP | `http://127.0.0.1:18061/mcp` | `search_court_cases`、`get_court_case` | 首次使用需按上游说明配置 Token |
| Tavily MCP | `https://mcp.tavily.com/mcp` | `search_official_legal_web`、`fetch_official_legal_page` | `TAVILY_API_KEY` |

LawRAG 不暴露下载、导出、登录、全站爬取等 MCP 原始工具。适配层限制结果数，对查询中的手机号和身份证号进行脱敏，并拒绝 Tavily 提取白名单以外的 URL。服务启动只完成工具注册，不访问任何外部 MCP；实际调用只发生在 Tool Router 选择对应工具之后。

### 6.4 准备数据与建立索引

```powershell
cd "D:\LLM study\LawRAG\LawRAG\backend"
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS lawrag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
python -m scripts.init_mysql
python -m scripts.prepare_datasets
python -m scripts.import_data
```

`init_mysql` 创建 Documents、Parent Chunks 和 Child Chunks 三张表，不调用模型。`prepare_datasets` 完成本地格式转换；`import_data` 先把原文及父子块写入 MySQL，再批量调用 `text-embedding-v3` 生成 Child 向量并写入 ChromaDB。旧版 ChromaDB 记录不含稳定 Child ID，升级后需要重新执行一次 `import_data`。

### 6.5 启动后端和前端

后端：

```powershell
cd "D:\LLM study\LawRAG\LawRAG\backend"
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

前端使用另一个终端启动：

```powershell
cd "D:\LLM study\LawRAG\LawRAG\frontend"
npm install
npm run dev
```

访问地址：

- 前端界面：`http://127.0.0.1:5173`
- 后端文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

在线 Agentic RAG 的模型调用发生在 Skill 选择、工具决策、证据充分性判断、必要的查询改写、答案生成和 Grounding 校验阶段；每轮法律检索还会调用一次 Embedding，选择 `cloud` 时增加一次云端重排。工具与生成循环均有硬上限。Faithfulness 评测和索引构建也会独立调用模型；Recall@5、MRR@10 与 P95 Latency 的指标计算本身不额外调用模型。

---

## 7. 测试、评测与质量保障

**本章使用的核心技术**：pytest、人工相关文档标注、排名指标、LLM-as-a-Judge、psutil 和 JSON 报告。测试负责验证确定性代码，评测从检索、生成和端到端延迟三个层面衡量 RAG 系统。

### 7.1 单元测试

项目使用离线单元测试覆盖管线基础能力与 RAG 指标计算：

| 测试文件 | 覆盖内容 |
|---|---|
| `test_basic.py` | 配置、法律分块、案例分块、元数据格式化 |
| `test_pipeline.py` | 底层检索与生成配置、策略枚举、兼容参数映射 |
| `test_advanced_retrieval.py` | 法律术语规范化、上下文标头 |
| `test_kg.py` | 犯罪知识加载和罪名查找 |
| `test_cloud_reranker.py` | 云端排序请求格式、响应映射、配置检查和无网络降级 |
| `test_rag_evaluation.py` | Recall@5、MRR@10、指标聚合和 P95 计算 |
| `test_context_memory_skills.py` | 四阶段压缩、TTL Memory、frontmatter 扫描、按需正文加载和 `load_skill` Tool |
| `test_parent_child_storage.py` | 稳定父子 ID、MySQL 回查、Child 权威正文、Child 精排与 Parent 聚合 |
| `test_agentic_rag.py` | LangGraph 节点顺序、LLM 工具决策、强制检索策略、证据评估和 Grounding 校验 |
| `test_external_mcp.py` | MCP 工具注册、底层参数映射、查询脱敏、结果上限和官方域名白名单 |

```powershell
cd backend
pip install pytest
pytest tests -v
```

测试套件将模型和网络依赖替换为可控 Mock。云端重排测试通过 `httpx.MockTransport` 验证请求、响应、排序映射与降级逻辑，形成稳定的离线回归环境。

### 7.2 RAG 核心评测

项目将评测指标收敛为四项核心指标：

| 指标 | 评测对象 | 计算方式 |
|---|---|---|
| Recall@5 | 召回覆盖率 | Top 5 命中的标准相关文档数 / 标准相关文档总数 |
| MRR@10 | 检索排序 | Top 10 中第一个相关文档排名的倒数；未命中记为 0 |
| P95 Latency | 端到端性能 | 批量请求延迟升序排列后的第 95 百分位值 |
| Faithfulness | 回答忠实度 | LLM Judge 判断答案是否得到检索来源支持，分值范围 0～10 |

评测数据位于 `backend/data/rag_eval_dataset.json`。每条样本包含问题、法律领域和 `relevant_documents` 标注；标注通过 `law_name`、`article_number`、`source_file` 或 `content_contains` 与召回文档元数据匹配。Recall@5 和 MRR@10 直接使用这些人工标注离线计算。

```powershell
python -m scripts.run_integration_test
python -m scripts.run_quality_eval
```

`run_integration_test` 使用 40 个问题验证多种策略组合；`run_quality_eval` 读取标准评测集，统一输出四项核心指标并将完整结果保存到 `backend/data/reports/rag_eval_results.json`。其中 Recall@5、MRR@10 和 P95 Latency 不需要额外 Judge 调用，Faithfulness 在显式运行完整评测时调用 LLM Judge。

### 7.3 工程质量设计

项目通过以下机制保证管线稳定性和结果可分析性：

- **单一 Agentic 主链路**：所有问答统一进入 LangGraph，不保留传统 RAG 路由；
- **有界自主决策**：LLM 选择高层工具和补充查询，工具轮次、检索轮次与生成轮次均有硬上限；
- **强制知识检索**：策略层保证每个法律回答至少调用一次 `retrieve_legal_evidence`；
- **证据闭环**：Evidence Grader 控制补充检索，Grounding Checker 控制答案修正；
- **事实库与索引分离**：MySQL 是原文、Parent 和 Child 的唯一事实来源，ChromaDB 是可重建的 Child 向量索引；
- **父子索引**：短 Child 负责精确召回与重排，完整 Parent 负责生成证据，兼顾检索精度与语义完整性；
- **稳定 ID 关联**：`doc_id → parent_chunk_id → child_chunk_id` 使用确定性 ID 串联导入、检索、删除、重建和来源回查；
- **上下文有预算**：四阶段压缩限制输入规模，保留法条原文、来源标签和高相关证据；
- **多轮可隔离**：会话以 `conversation_id` 隔离，仅保存最近 6 轮并使用 TTL 自动过期；
- **能力按需加载**：启动时只有技能目录进入 system prompt，完整 `SKILL.md` 仅在 `load_skill` 调用后以 tool_result 注入；
- **云端调用隔离**：模型客户端按需初始化，服务启动与单元测试保持零外部调用；
- **重排自动降级**：云端排序异常时切换到轻量算法，并将降级状态写入响应指标；
- **请求可观测**：记录查询变换、检索、知识增强、重排、生成和总耗时；
- **效果可量化**：统一计算 Recall@5、MRR@10、P95 Latency 和 Faithfulness；
- **结果可追溯**：回答返回 Parent 来源、命中 Child ID、法律元数据和排序分数，并可通过 Source API 回查原始文档；
- **报告可沉淀**：性能测试、质量指标和问答记录均可保存并下载为 JSON；
- **数据与代码分离**：数据集、向量索引、密钥和运行产物通过目录规范独立管理。
