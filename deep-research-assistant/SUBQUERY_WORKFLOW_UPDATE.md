# Subquery-Based Workflow Update

## Overview
Updated the research workflow to process each subquery individually through semantic search and web scraping, ensuring comprehensive analysis of all query aspects.

## Key Changes

### 1. State Schema Updates (`langgraph_agents.py`)
Added new state fields to track subquery-specific results:
- `subquery_retrieval_results`: Maps each subquery to its retrieval results
- `subquery_scraped_documents`: Maps each subquery to its scraped documents

### 2. ResearchCoordinatorAgent (`agents.py`)
**Before**: Performed semantic search once on the entire query
**After**: 
- Loops through each subquery
- Performs semantic search for each subquery individually
- Tracks results per subquery in `subquery_retrieval_results`
- Aggregates and deduplicates results across all subqueries
- Calculates average confidence score across all subqueries
- Emits progress events for each subquery retrieval

### 3. WebScraperRetrievalAgent (`agents.py`)
**Before**: Generated URLs and scraped based on the entire query
**After**:
- Processes each subquery separately
- Generates intelligent URLs for each subquery
- Scrapes documents for each subquery individually
- Tracks scraped documents per subquery in `subquery_scraped_documents`
- After upserting all documents, re-runs semantic search for each subquery
- Aggregates and deduplicates results from all subqueries
- Emits progress events for each subquery scraping

### 4. DeepAnalysisAgent (`agents.py`)
**Before**: Analyzed aggregated results without subquery context
**After**:
- Receives results organized by subquery
- Creates `subquery_context_map` linking each subquery to its relevant documents
- Passes subquery organization to comprehensive analysis
- Ensures each subquery is addressed in the analysis

### 5. Comprehensive Analysis Tool (`mcp_tools.py`)
**Before**: Received flat list of documents
**After**:
- Accepts `subqueries` and `subquery_context_map` parameters
- Organizes source materials by sub-question in the prompt
- Instructs GPT-4o to address each sub-question thoroughly
- Synthesizes information across all sub-questions

## Workflow Flow

```
1. Query Decomposition
   └─> Breaks query into N subqueries

2. Semantic Search (ResearchCoordinatorAgent)
   ├─> For each subquery:
   │   ├─> Perform semantic search
   │   ├─> Store results in subquery_retrieval_results[subquery]
   │   └─> Emit progress event
   └─> Aggregate and deduplicate all results

3. Web Scraping (WebScraperRetrievalAgent) [if confidence low]
   ├─> For each subquery:
   │   ├─> Generate intelligent URLs
   │   ├─> Scrape documents
   │   ├─> Store in subquery_scraped_documents[subquery]
   │   └─> Emit progress event
   ├─> Upsert all documents to vector store
   └─> Re-run semantic search for each subquery

4. Deep Analysis (DeepAnalysisAgent)
   ├─> Organize documents by subquery
   ├─> Pass subquery context map to comprehensive analysis
   └─> Generate analysis addressing all subqueries

5. Fact Checking & Output Formatting
   └─> Process aggregated results as before
```

## Benefits

1. **Comprehensive Coverage**: Each aspect of the query is analyzed separately
2. **Better Retrieval**: Focused searches per subquery improve relevance
3. **Targeted Scraping**: Web scraping is more focused on specific aspects
4. **Organized Analysis**: GPT-4o receives well-organized context by topic
5. **Traceability**: Can track which sources address which subqueries
6. **Progress Visibility**: Users see progress for each subquery

## Example

**Query**: "What is the impact of AI on healthcare and education?"

**Subqueries**:
1. "What is the impact of AI on healthcare?"
2. "What is the impact of AI on education?"

**Processing**:
- Semantic search runs twice (once per subquery)
- Web scraping generates URLs for healthcare AI and education AI separately
- Analysis receives sources organized by these two topics
- Final report addresses both aspects comprehensively

## Testing

To test the updated workflow:
1. Submit a complex query that naturally breaks into multiple aspects
2. Monitor progress events to see subquery processing
3. Verify that the final analysis addresses all subqueries
4. Check that sources are relevant to their respective subqueries
