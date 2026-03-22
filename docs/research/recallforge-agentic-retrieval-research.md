# Agentic Retrieval Patterns for MCP Memory Systems

## Executive Summary

NVIDIA's research shows that wrapping a retriever in an agentic ReACT loop (think → retrieve → evaluate → refine) boosted ViDoRe v3 from 64.36 to 69.22 NDCG@10 (+4.86 absolute improvement). This document analyzes NVIDIA's implementation, other agentic retrieval patterns, and provides design recommendations for RecallForge v0.3.

---

## 1. NVIDIA's Agentic Retrieval Implementation

### 1.1 Open Source Status & License

**Repository:** https://github.com/NVIDIA/NeMo-Retriever/tree/main/retrieval-bench

**License:** Apache 2.0 (fully open source, commercially usable)

The agentic retrieval code is located in:
- `retrieval-bench/src/retrieval_bench/nemo_agentic/agent.py` - Main agent loop
- `retrieval-bench/src/retrieval_bench/nemo_agentic/tool_helpers.py` - Tool definitions
- `retrieval-bench/src/retrieval_bench/nemo_agentic/selection_agent/` - Final ranking agent

### 1.2 The ReACT Loop Architecture

NVIDIA's agent uses a **3-tool ReACT pattern**:

#### Tool 1: `think`
```python
class ThinkTool(BaseTool):
    """Tool that allows the LLM to think with output tokens."""
    # Description: "Use the tool to think about something. It will not obtain 
    # new information or make any changes, but just log the thought."
    # Use cases:
    # - Organize thoughts on complex queries
    # - Brainstorm sub-queries needed
    # - Think about clues when queries are vague
    # - Plan search strategies when initial attempts fail
```

#### Tool 2: `retrieve`
```python
class RetrieveToolBase(BaseTool):
    """Marker base class for retrieve-like tools."""
    # - Performs semantic search with configurable top_k
    # - Supports deduplication of seen documents
    # - Excludes specified document IDs
    # - Over-fetches to guarantee top_k new results
```

#### Tool 3: `final_results`
```python
class FinalResults(BaseTool):
    """Tool for logging selected document IDs and signaling end of interaction."""
    # Parameters:
    # - doc_ids: List[str] (ranked by relevance)
    # - message: str (explanation of exploration process)
    # - search_successful: "true" | "false" | "partial"
```

### 1.3 Agent Loop Flow

```python
async def run_for_input(self, query: str, ...):
    # 1. Initialize with system prompt + initial retrieval (optional)
    # 2. Loop until final_results called or max_steps reached:
    #    a. LLM completion with tool definitions
    #    b. If tool_calls: process each tool call
    #    c. If no tool_calls: append auto_user_msg to continue
    #    d. If final_results called successfully: break
    # 3. Run selection_agent to rerank final results
    # 4. Return artifacts: message_history, retrieval_log, rrf_scores
```

### 1.4 Key Design Patterns

1. **Query Rewriting**: Before retrieval, an LLM rewrites queries for better recall
2. **Image Explanation**: Multimodal documents get image descriptions via separate LLM call
3. **Document Deduplication**: Tracks `retrieved_docs` set to avoid returning duplicates
4. **Over-fetch Strategy**: Retrieves more than needed, filters to guarantee `top_k` new docs
5. **Selection Agent**: Separate LLM agent makes final top-k selection from all retrieved docs
6. **RRF Fallback**: If agent fails, uses Reciprocal Rank Fusion as backup ranking

### 1.5 The MCP Removal

From `tool_helpers.py` docstring:
> "This is an internalized subset of the original external agent code. All MCP and fastmcp integration has been removed..."

**Why NVIDIA removed MCP:**
- MCP introduces serialization overhead (JSON-RPC)
- Each tool call requires a round-trip through MCP protocol
- For high-throughput benchmarking, in-process singleton is faster
- MCP is still used in production NVIDIA NeMo Retriever microservices, but removed from benchmark code

**Implications for RecallForge:**
- We ARE an MCP server, so we cannot remove MCP
- Must design agentic retrieval that works with MCP latency constraints
- Key insight: Reduce round trips, batch operations, cache aggressively

---

## 2. Other Agentic Retrieval Implementations

### 2.1 LlamaIndex Agentic RAG

**Pattern:** `ReActAgent` with `QueryEngineTool`

```python
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool

# Tools wrap vector index query engines
query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=lyft_engine,
        name="lyft_10k",
        description="Provides information about Lyft financials..."
    ),
]

agent = ReActAgent(tools=query_engine_tools, llm=OpenAI(model="gpt-4o-mini"))
response = await agent.run("Compare revenue growth of Uber and Lyft in 2021")
```

**Key Characteristics:**
- ReACT loop embedded in agent class
- Tools are query engines (single retrieval per call)
- No multi-round refinement by default - agent decides when to stop
- State managed via `Context` object

### 2.2 LangChain Self-Query Retriever

**Pattern:** Single-shot structured query generation

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Uses LLM to parse query into filter + search
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief summary of movie plots",
    metadata_field_info=metadata_fields
)
docs = retriever.invoke("What are some movies about dinosaurs")
```

**Key Characteristics:**
- LLM generates structured queries (filters + search terms)
- NOT a multi-turn agent - single query transformation
- Good for metadata filtering, not iterative refinement

### 2.3 DSPy Retrieval Modules

**Pattern:** Declarative programs with retriever modules

```python
# Basic retrieval
retriever = dspy.Retrieve(k=3)
results = retriever("What causes climate change?")

# Multi-hop retrieval (agentic)
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes, titles')
    
    def forward(self, claim: str) -> list[str]:
        notes = []
        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
        return dspy.Prediction(notes=notes)
```

**Key Characteristics:**
- `dspy.Retrieve` wraps any retriever (ColBERTv2, etc.)
- Multi-hop pattern is truly agentic: query → retrieve → reason → new query
- Composable with `dspy.ReAct` for tool-using agents
- Can optimize the entire pipeline end-to-end with DSPy teleprompters

### 2.4 MCP-Native Memory Servers

**Existing implementations:**
- **Recall** (recallmcp.com): Cross-session memory with semantic search
- **Muninn**: Local-first persistent memory with 5-signal hybrid search
- **MemOS**: Memory OS with multimodal, tool memory support

**None implement agentic retrieval patterns natively.** They provide:
- `search` / `retrieve` - single-shot semantic search
- `store` / `save` - memory persistence
- `list` / `get` - direct access by ID

---

## 3. Design Recommendations for RecallForge v0.3

### 3.1 Architectural Decision: Client-Side vs Server-Side Agentic Loop

**Option A: Server-Side (Inside RecallForge)**
```
LLM Client ──calls `agentic_search`──► RecallForge MCP Server
                                            │
                                            ▼
                                        Internal Agent Loop
                                        (think/retrieve/evaluate)
                                            │
                                            ▼
                                    Final Ranked Results
```

**Pros:**
- Single MCP call = minimal latency
- Client complexity hidden
- Can use specialized small models for evaluation

**Cons:**
- Requires RecallForge to host its own LLM (or call external API)
- More complex server implementation
- Less transparent to client (can't see reasoning)

**Option B: Client-Side (Expose Tools)**
```
LLM Client (Agent) ◄──MCP tools──► RecallForge MCP Server
      │                                      │
      ▼                                      │
   Agent Loop                               ▼
(think/retrieve/final_results)         Simple search
```

**Pros:**
- RecallForge stays simple (just tools)
- Client LLM is already the agent (no duplication)
- Transparent reasoning visible to user

**Cons:**
- Multiple MCP round trips
- Latency accumulation per retrieval

**Recommendation:** Hybrid Approach

```
LLM Client ──► RecallForge MCP Server
                   │
                   ├─► `search` (existing, single-shot)
                   │
                   ├─► `search_iterative` (new)
                   │       - query: str
                   │       - max_iterations: int (default 3)
                   │       - strategy: "refine" | "expand" | "multi_query"
                   │       Returns: final ranked results + reasoning trace
                   │
                   └─► `explain_results` (new)
                           - query: str
                           - doc_ids: List[str]
                           Returns: relevance explanations
```

### 3.2 Recommended MCP Tools

#### Tool 1: `search` (Existing)
Keep as-is for single-shot retrieval.

#### Tool 2: `search_iterative` (New)
```json
{
  "name": "search_iterative",
  "description": "Perform multi-round retrieval with automatic query refinement",
  "parameters": {
    "query": "string - Initial search query",
    "max_iterations": "int (default 3) - Maximum retrieval rounds",
    "strategy": "string - One of: refine, expand, multi_query",
    "return_trace": "bool - Include reasoning trace in response"
  },
  "returns": {
    "documents": ["Ranked list of documents"],
    "reasoning_trace": ["Optional: Query iterations and rationale"]
  }
}
```

**Strategy Implementations:**

1. **`refine`**: Single query refined each iteration based on results
   - Iteration 1: Initial query → results
   - Iteration 2: "Previous results lack X. New query: Y"
   - Continue until max_iterations or confidence threshold

2. **`expand`**: Generate multiple query variants, merge results with RRF
   - Parallel: Original query + 2-3 rewritten variants
   - Merge with Reciprocal Rank Fusion
   - Return top-k from merged set

3. **`multi_query`**: Decompose complex query into sub-queries
   - LLM decomposes: "Compare A and B" → ["A details", "B details", "A vs B comparison"]
   - Parallel retrieval, merged results

#### Tool 3: `explain_results` (New)
```json
{
  "name": "explain_results",
  "description": "Explain why specific documents were retrieved/ranked",
  "parameters": {
    "query": "string - Original query",
    "doc_ids": ["List of document IDs to explain"],
    "top_k": "int - Include top-k similar documents in explanation"
  },
  "returns": {
    "explanations": ["Per-document relevance rationale"],
    "similarity_matrix": "Optional: inter-document similarity"
  }
}
```

### 3.3 Working WITH MCP Latency

**Problem:** MCP round trips add latency (typically 50-200ms per call)

**NVIDIA's solution:** Remove MCP entirely, use in-process singletons.

**Our constraints:** We ARE an MCP server. Cannot remove MCP.

**Solutions:**

#### 1. Batch Operations
```json
// Instead of multiple search calls:
search({query: "A"})
search({query: "B"})
search({query: "C"})

// Single batch call:
search_batch({queries: ["A", "B", "C"]})
```

#### 2. Pre-computed Query Expansions
```python
# At index time, generate query expansions for common entities
cache = {
    "NVIDIA": ["NVIDIA Corporation", "NVDA", "Nvidia GPU company"],
    "MCP": ["Model Context Protocol", "MCP server", "Anthropic MCP"],
}

# At query time, expand automatically
def search_with_expansion(query):
    expansions = cache.get(query, generate_expansions(query))
    results = parallel_search([query] + expansions)
    return merge_with_rrf(results)
```

#### 3. Result Caching with Semantic Similarity
```python
# Cache query embeddings, not just exact queries
def semantic_cache_lookup(query_embedding, threshold=0.95):
    for cached_query, cached_embedding in cache:
        if cosine_similarity(query_embedding, cached_embedding) > threshold:
            return cache[cached_query]
    return None
```

#### 4. Streaming Results
```python
# Stream results as they arrive (MCP supports streaming)
async def search_stream(query):
    # First result arrives fast (top-5)
    yield {"batch": 1, "docs": top_5_results}
    
    # Second batch arrives later (top-20)
    yield {"batch": 2, "docs": top_6_to_20}
    
    # Agentic refinement happens in background
    refined = await refine_and_retrieve(query)
    yield {"batch": 3, "docs": refined}
```

### 3.4 Implementation Priorities

**Phase 1: Low-Hanging Fruit**
1. Add `search_batch` tool for parallel queries (reduces round trips)
2. Implement RRF merging for multi-query results
3. Add query expansion cache (semantic similarity lookup)

**Phase 2: Iterative Retrieval**
1. Implement `search_iterative` with `refine` strategy
2. Add `explain_results` for transparency
3. Integrate query rewriting LLM call

**Phase 3: Advanced Patterns**
1. `multi_query` strategy with query decomposition
2. Selection agent for final reranking (like NVIDIA)
3. Multimodal image explanation (if RecallForge supports images)

---

## 4. Key Questions Answered

### Q: Should the agentic loop be inside RecallForge or expose tools for client?

**A: Hybrid.** Keep simple `search` for single-shot, add `search_iterative` for multi-round, expose `explain_results` for transparency. Client can choose complexity level.

### Q: What about NVIDIA's MCP removal for performance?

**A: That's a benchmarking optimization.** In production, NVIDIA still uses MCP. We can mitigate latency with:
- Batch APIs
- Streaming results
- Semantic caching
- Pre-computed expansions

### Q: What's the minimum viable agentic retrieval for RecallForge v0.3?

**A: Three additions:**
1. `search_batch(queries[])` - parallel queries, RRF merge
2. `search_iterative(query, max_iterations)` - refine loop with local LLM
3. `explain_results(query, doc_ids)` - relevance explanations

### Q: How does this compare to just using Claude/GPT-4 with the existing search tool?

**A: The client LLM IS the agent.** The question is whether to:
1. Have the client orchestrate iterations (many MCP calls)
2. Have RecallForge orchestrate internally (single call, less transparent)
3. Hybrid: client decides strategy, RecallForge executes (balanced)

The hybrid approach preserves MCP architecture while reducing round trips.

---

## 5. References

1. **NVIDIA NeMo Retriever Agentic Retrieval**
   - Blog: https://huggingface.co/blog/nvidia/nemo-retriever-agentic-retrieval
   - Code: https://github.com/NVIDIA/NeMo-Retriever/tree/main/retrieval-bench
   - License: Apache 2.0

2. **LlamaIndex ReAct Agent**
   - Docs: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

3. **DSPy Retrieval Modules**
   - Docs: https://dspy.ai/learn/programming/modules/
   - Multi-hop: https://dspy.ai/tutorials/multihop_search/

4. **LangChain Self-Query Retriever**
   - Docs: https://python.langchain.com/docs/integrations/retrievers/self_query/

5. **MCP Memory Servers**
   - Recall: https://recallmcp.com/
   - Muninn: https://glama.ai/mcp/servers/wjohns989/Muninn

---

## 6. Appendix: NVIDIA Agent Prompts

### System Prompt Template (02_v1.j2)
```
You are a helpful assistant tasked with finding the most relevant documents for a given user query.

You have access to the following tools:
- `think`: Use this to organize your thoughts before searching
- `retrieve`: Search for documents using semantic similarity
- `final_results`: Call this when you have found all relevant documents

Guidelines:
1. Start by thinking about what information you need
2. Use multiple retrieval queries if needed
3. Documents may be retrieved multiple times - check for duplicates
4. Rank results by relevance in final_results
5. Set search_successful to "true", "false", or "partial" appropriately
```

### Think Tool Description
```
Use the tool to think about something. It will not obtain new information 
or make any changes, but just log the thought. Use it when complex 
reasoning or brainstorming is needed.

Common use cases:
- When processing a complex query, use this tool to organize your thoughts
- If a query is vague, use this to think about clues to narrow down the search
- When you fail to find results, think about alternative search strategies
```

### Final Results Tool Description
```
Signals the completion of the search process for the current query.

Use this tool when:
- You have found all the relevant documents to the query
- Despite several attempts, you cannot find good documents

Parameters:
- doc_ids: List of document IDs sorted by relevance (most relevant first)
- message: Summary of your exploration process
- search_successful: "true" | "false" | "partial"
```

---

*Research completed: 2026-03-15*
*Author: Research subagent for RecallForge v0.3 planning*