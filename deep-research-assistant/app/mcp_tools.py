"""
MCP Tools implementation for Deep Research Assistant.
All tools decorated with @mcp.tool() for agent use.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

from app.mcp import mcp_tool

logger = logging.getLogger(__name__)

# Global progress tracking queue for SSE events
progress_queue: asyncio.Queue = asyncio.Queue()


# ============================================================================
# Research Coordinator Agent Tools
# ============================================================================

@mcp_tool(
    name="query_decomposer",
    description="Decomposes complex queries into prioritized subqueries for systematic research",
    allowed_agents=["ResearchCoordinatorAgent"]
)
async def query_decomposer_tool(tool_input: dict, context: dict) -> dict:
    """
    Decomposes complex queries into subqueries and returns prioritized list.
    
    Args:
        tool_input: {"query": str, "max_subqueries": int}
        context: Additional context information
    """
    try:
        query = tool_input.get("query", "")
        max_subqueries = tool_input.get("max_subqueries", 5)
        
        if not query:
            return {"status": "error", "error": "Query is required", "meta": {}}
        
        # Simple query decomposition logic (can be enhanced with LLM)
        subqueries = []
        
        # Look for key topics and relationships
        # This is a simplified implementation - in production, use LLM for better decomposition
        
        # Extract key entities and concepts
        key_patterns = [
            r"COVID-19|coronavirus|pandemic",
            r"renewable energy|solar|wind|clean energy",
            r"investment|funding|finance|economic",
            r"developing countries|emerging markets",
            r"regulatory|policy|government|legislation",
            r"regional|geographic|country|nation"
        ]
        
        found_concepts = []
        for pattern in key_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                found_concepts.append(pattern.split("|")[0])  # Take first term as representative
        
        # Generate subqueries based on found concepts
        if "COVID-19" in str(found_concepts):
            subqueries.append({
                "subquery": "COVID-19 economic impacts on developing countries",
                "priority": 1,
                "rationale": "Understanding the baseline economic impact is crucial"
            })
        
        if "renewable energy" in str(found_concepts):
            subqueries.append({
                "subquery": "Renewable energy investment trends 2019-2023",
                "priority": 2,
                "rationale": "Historical context for investment patterns"
            })
        
        if "regulatory" in str(found_concepts):
            subqueries.append({
                "subquery": "Regional regulatory frameworks for renewable energy",
                "priority": 3,
                "rationale": "Policy context affects investment decisions"
            })
        
        # Add correlation analysis if multiple concepts found
        if len(found_concepts) > 1:
            subqueries.append({
                "subquery": "Correlation between economic disruption and policy changes",
                "priority": 4,
                "rationale": "Understanding causal relationships"
            })
            
            subqueries.append({
                "subquery": "Geographic variation in policy responses",
                "priority": 5,
                "rationale": "Regional differences in implementation"
            })
        
        # Limit to max_subqueries
        subqueries = subqueries[:max_subqueries]
        
        # Sort by priority
        subqueries.sort(key=lambda x: x["priority"])
        
        return {
            "status": "ok",
            "result": {
                "original_query": query,
                "subqueries": subqueries,
                "total_subqueries": len(subqueries)
            },
            "meta": {
                "concepts_found": found_concepts,
                "decomposition_method": "pattern_based"
            }
        }
        
    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        return {
            "status": "error",
            "error": f"Query decomposition failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="task_prioritizer",
    description="Ranks tasks by estimated cost and importance for optimal execution order",
    allowed_agents=["ResearchCoordinatorAgent"]
)
async def task_prioritizer_tool(tool_input: dict, context: dict) -> dict:
    """
    Ranks tasks by estimated cost/importance.
    
    Args:
        tool_input: {"tasks": List[dict], "criteria": dict}
        context: Additional context information
    """
    try:
        tasks = tool_input.get("tasks", [])
        criteria = tool_input.get("criteria", {})
        
        if not tasks:
            return {"status": "error", "error": "Tasks list is required", "meta": {}}
        
        # Default scoring criteria
        default_criteria = {
            "importance_weight": 0.6,
            "cost_weight": 0.4,
            "urgency_weight": 0.3
        }
        criteria = {**default_criteria, **criteria}
        
        prioritized_tasks = []
        
        for i, task in enumerate(tasks):
            # Calculate priority score based on multiple factors
            importance = task.get("importance", 5)  # 1-10 scale
            estimated_cost = task.get("estimated_cost", 5)  # 1-10 scale (lower is better)
            urgency = task.get("urgency", 5)  # 1-10 scale
            
            # Calculate composite score (higher is better)
            priority_score = (
                importance * criteria["importance_weight"] +
                (11 - estimated_cost) * criteria["cost_weight"] +  # Invert cost (lower cost = higher score)
                urgency * criteria["urgency_weight"]
            )
            
            prioritized_tasks.append({
                "task_id": task.get("id", i),
                "task": task,
                "priority_score": round(priority_score, 2),
                "rank": 0  # Will be set after sorting
            })
        
        # Sort by priority score (descending)
        prioritized_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
        
        # Assign ranks
        for i, task in enumerate(prioritized_tasks):
            task["rank"] = i + 1
        
        return {
            "status": "ok",
            "result": {
                "prioritized_tasks": prioritized_tasks,
                "total_tasks": len(prioritized_tasks),
                "criteria_used": criteria
            },
            "meta": {
                "prioritization_method": "weighted_scoring"
            }
        }
        
    except Exception as e:
        logger.error(f"Task prioritization failed: {e}")
        return {
            "status": "error",
            "error": f"Task prioritization failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="progress_tracker",
    description="Emits progress updates to SSE queue for real-time client updates",
    allowed_agents=["ResearchCoordinatorAgent", "WebScraperRetrievalAgent", "DeepAnalysisAgent", "FactCheckingAgent", "OutputFormattingAgent"]
)
async def progress_tracker_tool(tool_input: dict, context: dict) -> dict:
    """
    Emits progress updates (to SSE queue).
    
    Args:
        tool_input: {"event": str, "step": str, "payload": dict, "task_id": str}
        context: Additional context information
    """
    try:
        event_type = tool_input.get("event", "progress")
        step = tool_input.get("step", "unknown")
        payload = tool_input.get("payload", {})
        task_id = tool_input.get("task_id", context.get("task_id", "unknown"))
        
        # Create progress event
        progress_event = {
            "event": event_type,
            "step": step,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "task_id": task_id
        }
        
        # Add to progress queue for SSE streaming
        await progress_queue.put(progress_event)
        
        logger.info(f"Progress event emitted: {event_type}/{step}")
        
        return {
            "status": "ok",
            "result": {
                "event_emitted": True,
                "event": progress_event
            },
            "meta": {
                "queue_size": progress_queue.qsize()
            }
        }
        
    except Exception as e:
        logger.error(f"Progress tracking failed: {e}")
        return {
            "status": "error",
            "error": f"Progress tracking failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="result_synthesis",
    description="Combines agent outputs into a single answer with intermediate reasoning steps",
    allowed_agents=["ResearchCoordinatorAgent"]
)
async def result_synthesis_tool(tool_input: dict, context: dict) -> dict:
    """
    Combines agent outputs into a single answer with intermediate reasoning steps.
    
    Args:
        tool_input: {"agent_results": List[dict], "synthesis_method": str}
        context: Additional context information
    """
    try:
        agent_results = tool_input.get("agent_results", [])
        synthesis_method = tool_input.get("synthesis_method", "hierarchical")
        
        if not agent_results:
            return {"status": "error", "error": "Agent results are required", "meta": {}}
        
        # Organize results by agent type
        results_by_agent = {}
        for result in agent_results:
            agent_name = result.get("agent_name", "unknown")
            if agent_name not in results_by_agent:
                results_by_agent[agent_name] = []
            results_by_agent[agent_name].append(result)
        
        # Synthesis logic based on method
        if synthesis_method == "hierarchical":
            synthesized_result = await _hierarchical_synthesis(results_by_agent)
        elif synthesis_method == "weighted":
            synthesized_result = await _weighted_synthesis(results_by_agent)
        else:
            synthesized_result = await _simple_synthesis(results_by_agent)
        
        return {
            "status": "ok",
            "result": synthesized_result,
            "meta": {
                "synthesis_method": synthesis_method,
                "agents_involved": list(results_by_agent.keys()),
                "total_results": len(agent_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Result synthesis failed: {e}")
        return {
            "status": "error",
            "error": f"Result synthesis failed: {str(e)}",
            "meta": {}
        }


# Helper functions for result synthesis
async def _hierarchical_synthesis(results_by_agent: Dict[str, List[dict]]) -> dict:
    """Hierarchical synthesis prioritizing analysis over raw data."""
    synthesis = {
        "executive_summary": "",
        "key_findings": [],
        "supporting_evidence": [],
        "methodology": [],
        "confidence_score": 0.0,
        "reasoning_chain": []
    }
    
    # Process results in order of analytical depth
    agent_priority = [
        "OutputFormattingAgent",
        "FactCheckingAgent", 
        "DeepAnalysisAgent",
        "WebScraperRetrievalAgent"
    ]
    
    total_confidence = 0.0
    confidence_count = 0
    
    for agent in agent_priority:
        if agent in results_by_agent:
            for result in results_by_agent[agent]:
                # Extract key information
                if "findings" in result:
                    synthesis["key_findings"].extend(result["findings"])
                
                if "evidence" in result:
                    synthesis["supporting_evidence"].extend(result["evidence"])
                
                if "confidence" in result:
                    total_confidence += result["confidence"]
                    confidence_count += 1
                
                # Add to reasoning chain
                synthesis["reasoning_chain"].append({
                    "agent": agent,
                    "step": result.get("step", "unknown"),
                    "contribution": result.get("summary", "No summary provided")
                })
    
    # Calculate overall confidence
    if confidence_count > 0:
        synthesis["confidence_score"] = total_confidence / confidence_count
    
    return synthesis


async def _weighted_synthesis(results_by_agent: Dict[str, List[dict]]) -> dict:
    """Weighted synthesis based on agent reliability."""
    # Agent weights based on reliability for different types of information
    agent_weights = {
        "FactCheckingAgent": 0.9,
        "DeepAnalysisAgent": 0.8,
        "OutputFormattingAgent": 0.7,
        "WebScraperRetrievalAgent": 0.6
    }
    
    # Implementation would weight results based on agent reliability
    # For now, return simple synthesis
    return await _simple_synthesis(results_by_agent)


async def _simple_synthesis(results_by_agent: Dict[str, List[dict]]) -> dict:
    """Simple concatenation synthesis."""
    synthesis = {
        "combined_results": [],
        "agent_contributions": {},
        "total_results": 0
    }
    
    for agent, results in results_by_agent.items():
        synthesis["agent_contributions"][agent] = len(results)
        synthesis["combined_results"].extend(results)
        synthesis["total_results"] += len(results)
    
    return synthesis

# =====
#=======================================================================
# Web Scraper Agent Tools
# ============================================================================

@mcp_tool(
    name="keyword_search",
    description="Uses Postgres full-text search to find candidate pages or seed URLs",
    allowed_agents=["WebScraperRetrievalAgent"]
)
async def keyword_search_tool(tool_input: dict, context: dict) -> dict:
    """
    Use Postgres full-text search to find candidate pages or seed URLs.
    
    Args:
        tool_input: {"keywords": str, "limit": int, "min_score": float}
        context: Additional context information
    """
    try:
        from app.db.database import db_manager
        from app.db.models import Document
        from sqlalchemy import text
        
        keywords = tool_input.get("keywords", "")
        limit = tool_input.get("limit", 10)
        min_score = tool_input.get("min_score", 0.1)
        
        if not keywords:
            return {"status": "error", "error": "Keywords are required", "meta": {}}
        
        # Perform full-text search using PostgreSQL
        async with db_manager.get_session() as session:
            # Use PostgreSQL full-text search with ranking
            query = text("""
                SELECT d.id, d.url, d.title, d.cleaned_text,
                       ts_rank(to_tsvector('english', d.title || ' ' || d.cleaned_text), 
                              plainto_tsquery('english', :keywords)) as rank
                FROM documents d
                WHERE to_tsvector('english', d.title || ' ' || d.cleaned_text) 
                      @@ plainto_tsquery('english', :keywords)
                      AND d.language = 'en'
                ORDER BY rank DESC
                LIMIT :limit
            """)
            
            result = await session.execute(query, {
                "keywords": keywords,
                "limit": limit
            })
            
            documents = []
            for row in result:
                if row.rank >= min_score:
                    documents.append({
                        "id": row.id,
                        "url": row.url,
                        "title": row.title,
                        "relevance_score": float(row.rank),
                        "snippet": row.cleaned_text[:200] + "..." if len(row.cleaned_text) > 200 else row.cleaned_text
                    })
        
        return {
            "status": "ok",
            "result": {
                "documents": documents,
                "total_found": len(documents),
                "search_keywords": keywords
            },
            "meta": {
                "search_method": "postgresql_fulltext",
                "min_score_threshold": min_score
            }
        }
        
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return {
            "status": "error",
            "error": f"Keyword search failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="web_scraper",
    description="Fetches pages with aiohttp and Playwright fallback, extracts text and metadata",
    allowed_agents=["WebScraperRetrievalAgent"]
)
async def web_scraper_tool(tool_input: dict, context: dict) -> dict:
    """
    Given a URL or search-term seed, fetches pages, extracts text, metadata.
    
    Args:
        tool_input: {"urls": List[str], "search_terms": List[str], "max_pages": int}
        context: Additional context information
    """
    try:
        from app.workers.scraper_worker import ScrapingWorker
        
        urls = tool_input.get("urls", [])
        search_terms = tool_input.get("search_terms", [])
        max_pages = tool_input.get("max_pages", 10)
        
        if not urls and not search_terms:
            return {"status": "error", "error": "URLs or search terms are required", "meta": {}}
        
        # Initialize scraping worker
        scraper = ScrapingWorker()
        
        # If search terms provided, convert to URLs (simplified - would use search API in production)
        if search_terms and not urls:
            # For demo purposes, generate some example URLs
            # In production, this would use a search API like Google Custom Search
            example_urls = [
                f"https://example.com/search?q={'+'.join(term.split())}"
                for term in search_terms[:3]
            ]
            urls.extend(example_urls)
        
        # Limit URLs to max_pages
        urls = urls[:max_pages]
        
        # Scrape URLs
        scraped_documents = await scraper.scrape_urls(urls)
        
        return {
            "status": "ok",
            "result": {
                "scraped_documents": scraped_documents,
                "total_scraped": len(scraped_documents),
                "requested_urls": urls
            },
            "meta": {
                "scraper_config": {
                    "max_concurrent": scraper.max_concurrent,
                    "rate_limit": scraper.rate_limit
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Web scraping failed: {e}")
        return {
            "status": "error",
            "error": f"Web scraping failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="semantic_scraping",
    description="Specialized extraction using schema.org and readability heuristics",
    allowed_agents=["WebScraperRetrievalAgent"]
)
async def semantic_scraping_tool(tool_input: dict, context: dict) -> dict:
    """
    Specialized extraction using schema.org and readability heuristics.
    
    Args:
        tool_input: {"html_content": str, "url": str, "extract_schema": bool}
        context: Additional context information
    """
    try:
        from app.utils.text_cleaner import extract_semantic_content
        
        html_content = tool_input.get("html_content", "")
        url = tool_input.get("url", "")
        extract_schema = tool_input.get("extract_schema", True)
        
        if not html_content:
            return {"status": "error", "error": "HTML content is required", "meta": {}}
        
        # Extract semantic content
        extracted_content = await extract_semantic_content(
            html_content, 
            url, 
            extract_schema=extract_schema
        )
        
        return {
            "status": "ok",
            "result": extracted_content,
            "meta": {
                "extraction_method": "semantic_readability",
                "schema_extraction": extract_schema
            }
        }
        
    except Exception as e:
        logger.error(f"Semantic scraping failed: {e}")
        return {
            "status": "error",
            "error": f"Semantic scraping failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="rag_upsert",
    description="Chunks, embeds, and upserts content to pgvector for RAG retrieval",
    allowed_agents=["WebScraperRetrievalAgent"]
)
async def rag_upsert_tool(tool_input: dict, context: dict) -> dict:
    """
    Chunk -> embed -> upsert to pgvector.
    
    Args:
        tool_input: {"documents": List[dict], "chunk_strategy": str}
        context: Additional context information
    """
    try:
        from app.utils.chunker import SemanticChunker
        from app.workers.embedding import EmbeddingGenerator
        from app.db.database import db_manager
        from app.db.models import Document, Chunk
        from sqlalchemy import select
        
        documents = tool_input.get("documents", [])
        chunk_strategy = tool_input.get("chunk_strategy", "semantic")
        
        if not documents:
            return {"status": "error", "error": "Documents are required", "meta": {}}
        
        # Initialize components
        chunker = SemanticChunker()
        embedder = EmbeddingGenerator()
        
        processed_documents = []
        total_chunks = 0
        
        async with db_manager.get_session() as session:
            for doc_data in documents:
                try:
                    # Check if document already exists
                    existing_doc = await session.execute(
                        select(Document).where(Document.url == doc_data["url"])
                    )
                    existing_doc = existing_doc.scalar_one_or_none()
                    
                    if existing_doc:
                        logger.info(f"Document already exists: {doc_data['url']}")
                        continue
                    
                    # Create document record
                    document = Document(
                        url=doc_data["url"],
                        title=doc_data.get("title", ""),
                        raw_html=doc_data.get("raw_html", ""),
                        cleaned_text=doc_data.get("cleaned_text", ""),
                        language=doc_data.get("language", "en"),
                        source_trust_score=doc_data.get("source_trust_score", 0.5),
                        publish_date=doc_data.get("publish_date"),
                        license=doc_data.get("license")
                    )
                    
                    session.add(document)
                    await session.flush()  # Get the document ID
                    
                    # Chunk the document
                    chunks = await chunker.chunk_document(
                        document.cleaned_text,
                        {"document_id": document.id, "url": document.url}
                    )
                    
                    # Generate embeddings and create chunk records
                    chunk_texts = [chunk.chunk_text for chunk in chunks]
                    embeddings = await embedder.generate_embeddings(chunk_texts)
                    
                    for chunk_data, embedding in zip(chunks, embeddings):
                        chunk = Chunk(
                            document_id=document.id,
                            chunk_text=chunk_data.chunk_text,
                            token_count=chunk_data.token_count,
                            chunk_meta=chunk_data.chunk_meta,
                            embedding=embedding
                        )
                        session.add(chunk)
                    
                    total_chunks += len(chunks)
                    processed_documents.append({
                        "document_id": document.id,
                        "url": document.url,
                        "chunks_created": len(chunks)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc_data.get('url', 'unknown')}: {e}")
                    continue
        
        return {
            "status": "ok",
            "result": {
                "processed_documents": processed_documents,
                "total_documents": len(processed_documents),
                "total_chunks": total_chunks
            },
            "meta": {
                "chunk_strategy": chunk_strategy,
                "embedding_model": "text-embedding-3-small"
            }
        }
        
    except Exception as e:
        logger.error(f"RAG upsert failed: {e}")
        return {
            "status": "error",
            "error": f"RAG upsert failed: {str(e)}",
            "meta": {}
        }
# ===
#=========================================================================
# Deep Analysis Agent Tools
# ============================================================================

@mcp_tool(
    name="comparative_analysis",
    description="Performs cross-document comparative analysis to identify patterns and differences",
    allowed_agents=["DeepAnalysisAgent"]
)
async def comparative_analysis_tool(tool_input: dict, context: dict) -> dict:
    """
    Create Comparative Analysis Tool for cross-document analysis.
    
    Args:
        tool_input: {"documents": List[dict], "comparison_dimensions": List[str]}
        context: Additional context information
    """
    try:
        documents = tool_input.get("documents", [])
        comparison_dimensions = tool_input.get("comparison_dimensions", ["content", "methodology", "conclusions"])
        
        if len(documents) < 2:
            return {"status": "error", "error": "At least 2 documents required for comparison", "meta": {}}
        
        comparisons = []
        
        # Compare documents pairwise
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1, doc2 = documents[i], documents[j]
                
                comparison = {
                    "document_pair": [doc1.get("id", i), doc2.get("id", j)],
                    "titles": [doc1.get("title", ""), doc2.get("title", "")],
                    "similarities": [],
                    "differences": [],
                    "confidence": 0.0
                }
                
                # Analyze each dimension
                for dimension in comparison_dimensions:
                    similarity_score = await _calculate_text_similarity(
                        doc1.get("content", ""), 
                        doc2.get("content", "")
                    )
                    
                    if similarity_score > 0.7:
                        comparison["similarities"].append({
                            "dimension": dimension,
                            "score": similarity_score,
                            "description": f"High similarity in {dimension}"
                        })
                    elif similarity_score < 0.3:
                        comparison["differences"].append({
                            "dimension": dimension,
                            "score": similarity_score,
                            "description": f"Significant difference in {dimension}"
                        })
                
                # Calculate overall confidence
                comparison["confidence"] = sum(s["score"] for s in comparison["similarities"]) / len(comparison_dimensions) if comparison_dimensions else 0.0
                comparisons.append(comparison)
        
        # Generate insights
        insights = await _generate_comparative_insights(comparisons)
        
        return {
            "status": "ok",
            "result": {
                "comparisons": comparisons,
                "insights": insights,
                "total_comparisons": len(comparisons)
            },
            "meta": {
                "analysis_method": "pairwise_comparison",
                "dimensions_analyzed": comparison_dimensions
            }
        }
        
    except Exception as e:
        logger.error(f"Comparative analysis failed: {e}")
        return {
            "status": "error",
            "error": f"Comparative analysis failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="trend_analysis",
    description="Analyzes temporal patterns and trends across documents and data points",
    allowed_agents=["DeepAnalysisAgent"]
)
async def trend_analysis_tool(tool_input: dict, context: dict) -> dict:
    """
    Implement Trend Analysis Tool for temporal pattern detection.
    
    Args:
        tool_input: {"data_points": List[dict], "time_dimension": str, "metrics": List[str]}
        context: Additional context information
    """
    try:
        data_points = tool_input.get("data_points", [])
        time_dimension = tool_input.get("time_dimension", "date")
        metrics = tool_input.get("metrics", ["value"])
        
        if not data_points:
            return {"status": "error", "error": "Data points are required", "meta": {}}
        
        # Sort data points by time dimension
        try:
            sorted_data = sorted(data_points, key=lambda x: x.get(time_dimension, ""))
        except:
            sorted_data = data_points  # Fallback if sorting fails
        
        trends = {}
        
        for metric in metrics:
            metric_values = []
            time_points = []
            
            for point in sorted_data:
                if metric in point and time_dimension in point:
                    metric_values.append(float(point[metric]))
                    time_points.append(point[time_dimension])
            
            if len(metric_values) < 2:
                continue
            
            # Calculate trend statistics
            trend_analysis = {
                "metric": metric,
                "data_points": len(metric_values),
                "trend_direction": "stable",
                "trend_strength": 0.0,
                "change_rate": 0.0,
                "volatility": 0.0,
                "key_periods": []
            }
            
            # Simple trend calculation
            if len(metric_values) >= 2:
                start_value = metric_values[0]
                end_value = metric_values[-1]
                
                if end_value > start_value * 1.1:
                    trend_analysis["trend_direction"] = "increasing"
                elif end_value < start_value * 0.9:
                    trend_analysis["trend_direction"] = "decreasing"
                
                trend_analysis["change_rate"] = (end_value - start_value) / start_value if start_value != 0 else 0
                
                # Calculate volatility (standard deviation)
                mean_value = sum(metric_values) / len(metric_values)
                variance = sum((x - mean_value) ** 2 for x in metric_values) / len(metric_values)
                trend_analysis["volatility"] = variance ** 0.5
                
                # Identify significant changes
                for i in range(1, len(metric_values)):
                    change = abs(metric_values[i] - metric_values[i-1]) / metric_values[i-1] if metric_values[i-1] != 0 else 0
                    if change > 0.2:  # 20% change threshold
                        trend_analysis["key_periods"].append({
                            "period": f"{time_points[i-1]} to {time_points[i]}",
                            "change": change,
                            "description": f"Significant {'increase' if metric_values[i] > metric_values[i-1] else 'decrease'}"
                        })
            
            trends[metric] = trend_analysis
        
        return {
            "status": "ok",
            "result": {
                "trends": trends,
                "time_range": {
                    "start": time_points[0] if time_points else None,
                    "end": time_points[-1] if time_points else None
                },
                "total_data_points": len(sorted_data)
            },
            "meta": {
                "analysis_method": "statistical_trend",
                "time_dimension": time_dimension
            }
        }
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        return {
            "status": "error",
            "error": f"Trend analysis failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="causal_reasoning",
    description="Analyzes cause-effect relationships and causal chains in the data",
    allowed_agents=["DeepAnalysisAgent"]
)
async def causal_reasoning_tool(tool_input: dict, context: dict) -> dict:
    """
    Build Causal Reasoning Tool for cause-effect relationships.
    
    Args:
        tool_input: {"events": List[dict], "potential_causes": List[str], "effects": List[str]}
        context: Additional context information
    """
    try:
        events = tool_input.get("events", [])
        potential_causes = tool_input.get("potential_causes", [])
        effects = tool_input.get("effects", [])
        
        if not events:
            return {"status": "error", "error": "Events are required for causal analysis", "meta": {}}
        
        causal_relationships = []
        
        # Analyze temporal relationships
        for cause in potential_causes:
            for effect in effects:
                relationship = await _analyze_causal_relationship(events, cause, effect)
                if relationship["strength"] > 0.3:  # Threshold for significant relationships
                    causal_relationships.append(relationship)
        
        # Build causal chains
        causal_chains = await _build_causal_chains(causal_relationships)
        
        return {
            "status": "ok",
            "result": {
                "causal_relationships": causal_relationships,
                "causal_chains": causal_chains,
                "total_relationships": len(causal_relationships)
            },
            "meta": {
                "analysis_method": "temporal_correlation",
                "confidence_threshold": 0.3
            }
        }
        
    except Exception as e:
        logger.error(f"Causal reasoning failed: {e}")
        return {
            "status": "error",
            "error": f"Causal reasoning failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="statistical_analysis",
    description="Performs quantitative statistical analysis on numerical data",
    allowed_agents=["DeepAnalysisAgent"]
)
async def statistical_analysis_tool(tool_input: dict, context: dict) -> dict:
    """
    Develop Statistical Analysis Tool for quantitative insights.
    
    Args:
        tool_input: {"datasets": List[dict], "analysis_types": List[str]}
        context: Additional context information
    """
    try:
        datasets = tool_input.get("datasets", [])
        analysis_types = tool_input.get("analysis_types", ["descriptive", "correlation"])
        
        if not datasets:
            return {"status": "error", "error": "Datasets are required", "meta": {}}
        
        statistical_results = {}
        
        for dataset_name, data in datasets.items() if isinstance(datasets, dict) else enumerate(datasets):
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                stats = {}
                
                if "descriptive" in analysis_types:
                    stats["descriptive"] = {
                        "count": len(data),
                        "mean": sum(data) / len(data) if data else 0,
                        "median": sorted(data)[len(data)//2] if data else 0,
                        "min": min(data) if data else 0,
                        "max": max(data) if data else 0,
                        "std_dev": (sum((x - sum(data)/len(data))**2 for x in data) / len(data))**0.5 if len(data) > 1 else 0
                    }
                
                if "distribution" in analysis_types:
                    stats["distribution"] = await _analyze_distribution(data)
                
                statistical_results[str(dataset_name)] = stats
        
        # Cross-dataset correlations if multiple datasets
        correlations = {}
        if len(datasets) > 1 and "correlation" in analysis_types:
            correlations = await _calculate_correlations(datasets)
        
        return {
            "status": "ok",
            "result": {
                "statistical_results": statistical_results,
                "correlations": correlations,
                "datasets_analyzed": len(datasets)
            },
            "meta": {
                "analysis_types": analysis_types,
                "statistical_method": "basic_statistics"
            }
        }
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        return {
            "status": "error",
            "error": f"Statistical analysis failed: {str(e)}",
            "meta": {}
        }


# Helper functions for Deep Analysis tools
async def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    # Simple word overlap similarity (can be enhanced with embeddings)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


async def _generate_comparative_insights(comparisons: List[dict]) -> List[dict]:
    """Generate insights from comparative analysis."""
    insights = []
    
    # Find most similar documents
    if comparisons:
        most_similar = max(comparisons, key=lambda x: x["confidence"])
        insights.append({
            "type": "similarity",
            "description": f"Highest similarity found between documents {most_similar['document_pair']}",
            "confidence": most_similar["confidence"]
        })
    
    return insights


async def _analyze_causal_relationship(events: List[dict], cause: str, effect: str) -> dict:
    """Analyze potential causal relationship between cause and effect."""
    # Simplified causal analysis based on temporal ordering and co-occurrence
    cause_events = [e for e in events if cause.lower() in str(e).lower()]
    effect_events = [e for e in events if effect.lower() in str(e).lower()]
    
    # Calculate temporal correlation (simplified)
    strength = 0.0
    if cause_events and effect_events:
        # Simple heuristic: if cause events generally precede effect events
        strength = min(len(cause_events), len(effect_events)) / max(len(cause_events), len(effect_events))
    
    return {
        "cause": cause,
        "effect": effect,
        "strength": strength,
        "evidence_count": len(cause_events) + len(effect_events),
        "confidence": strength * 0.8  # Reduce confidence for simplified analysis
    }


async def _build_causal_chains(relationships: List[dict]) -> List[dict]:
    """Build causal chains from individual relationships."""
    chains = []
    
    # Simple chain building (can be enhanced with graph algorithms)
    for rel in relationships:
        chains.append({
            "chain": [rel["cause"], rel["effect"]],
            "strength": rel["strength"],
            "length": 2
        })
    
    return chains


async def _analyze_distribution(data: List[float]) -> dict:
    """Analyze data distribution characteristics."""
    if not data:
        return {}
    
    sorted_data = sorted(data)
    n = len(data)
    
    return {
        "quartiles": {
            "q1": sorted_data[n//4] if n > 4 else sorted_data[0],
            "q2": sorted_data[n//2],
            "q3": sorted_data[3*n//4] if n > 4 else sorted_data[-1]
        },
        "outliers": [x for x in data if abs(x - sum(data)/len(data)) > 2 * ((sum((x - sum(data)/len(data))**2 for x in data) / len(data))**0.5)],
        "skewness": "normal"  # Simplified - would calculate actual skewness
    }


async def _calculate_correlations(datasets: dict) -> dict:
    """Calculate correlations between datasets."""
    correlations = {}
    
    dataset_items = list(datasets.items()) if isinstance(datasets, dict) else list(enumerate(datasets))
    
    for i, (name1, data1) in enumerate(dataset_items):
        for j, (name2, data2) in enumerate(dataset_items[i+1:], i+1):
            if isinstance(data1, list) and isinstance(data2, list) and all(isinstance(x, (int, float)) for x in data1 + data2):
                # Simple correlation calculation
                min_len = min(len(data1), len(data2))
                if min_len > 1:
                    corr_data1 = data1[:min_len]
                    corr_data2 = data2[:min_len]
                    
                    mean1 = sum(corr_data1) / len(corr_data1)
                    mean2 = sum(corr_data2) / len(corr_data2)
                    
                    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(corr_data1, corr_data2))
                    denominator = (sum((x - mean1)**2 for x in corr_data1) * sum((y - mean2)**2 for y in corr_data2))**0.5
                    
                    correlation = numerator / denominator if denominator != 0 else 0
                    
                    correlations[f"{name1}_vs_{name2}"] = {
                        "correlation": correlation,
                        "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
                    }
    
    return correlations# ==
#==========================================================================
# Fact-Checking Agent Tools
# ============================================================================

@mcp_tool(
    name="source_credibility_checker",
    description="Evaluates source credibility and trustworthiness based on multiple factors",
    allowed_agents=["FactCheckingAgent"]
)
async def source_credibility_checker_tool(tool_input: dict, context: dict) -> dict:
    """
    Create Source Credibility Checker for trust scoring.
    
    Args:
        tool_input: {"sources": List[dict], "criteria": List[str]}
        context: Additional context information
    """
    try:
        sources = tool_input.get("sources", [])
        criteria = tool_input.get("criteria", ["domain_authority", "publication_date", "author_credentials", "citation_count"])
        
        if not sources:
            return {"status": "error", "error": "Sources are required", "meta": {}}
        
        credibility_scores = []
        
        for source in sources:
            score_components = {}
            total_score = 0.0
            max_score = 0.0
            
            # Domain authority scoring
            if "domain_authority" in criteria:
                domain_score = await _evaluate_domain_authority(source.get("url", ""))
                score_components["domain_authority"] = domain_score
                total_score += domain_score * 0.3
                max_score += 0.3
            
            # Publication date recency
            if "publication_date" in criteria:
                date_score = await _evaluate_publication_recency(source.get("publish_date"))
                score_components["publication_recency"] = date_score
                total_score += date_score * 0.2
                max_score += 0.2
            
            # Author credentials (simplified)
            if "author_credentials" in criteria:
                author_score = await _evaluate_author_credentials(source.get("author", ""))
                score_components["author_credentials"] = author_score
                total_score += author_score * 0.2
                max_score += 0.2
            
            # Citation/reference count
            if "citation_count" in criteria:
                citation_score = await _evaluate_citations(source.get("citations", 0))
                score_components["citation_score"] = citation_score
                total_score += citation_score * 0.3
                max_score += 0.3
            
            # Normalize score
            final_score = total_score / max_score if max_score > 0 else 0.0
            
            credibility_scores.append({
                "source_id": source.get("id", "unknown"),
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "credibility_score": round(final_score, 3),
                "score_components": score_components,
                "trust_level": _get_trust_level(final_score),
                "recommendations": await _generate_credibility_recommendations(final_score, score_components)
            })
        
        return {
            "status": "ok",
            "result": {
                "credibility_scores": credibility_scores,
                "average_credibility": sum(s["credibility_score"] for s in credibility_scores) / len(credibility_scores),
                "high_credibility_sources": [s for s in credibility_scores if s["credibility_score"] > 0.7]
            },
            "meta": {
                "evaluation_criteria": criteria,
                "total_sources": len(sources)
            }
        }
        
    except Exception as e:
        logger.error(f"Source credibility check failed: {e}")
        return {
            "status": "error",
            "error": f"Source credibility check failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="cross_reference_validator",
    description="Validates information by cross-referencing multiple sources",
    allowed_agents=["FactCheckingAgent"]
)
async def cross_reference_validator_tool(tool_input: dict, context: dict) -> dict:
    """
    Implement Cross-Reference Validator for information verification.
    
    Args:
        tool_input: {"claims": List[dict], "sources": List[dict], "validation_threshold": float}
        context: Additional context information
    """
    try:
        claims = tool_input.get("claims", [])
        sources = tool_input.get("sources", [])
        validation_threshold = tool_input.get("validation_threshold", 0.6)
        
        if not claims:
            return {"status": "error", "error": "Claims are required for validation", "meta": {}}
        
        validation_results = []
        
        for claim in claims:
            claim_text = claim.get("text", "")
            claim_id = claim.get("id", "unknown")
            
            # Find supporting and contradicting sources
            supporting_sources = []
            contradicting_sources = []
            neutral_sources = []
            
            for source in sources:
                source_content = source.get("content", "")
                support_score = await _calculate_claim_support(claim_text, source_content)
                
                if support_score > 0.7:
                    supporting_sources.append({
                        "source": source,
                        "support_score": support_score
                    })
                elif support_score < 0.3:
                    contradicting_sources.append({
                        "source": source,
                        "contradiction_score": 1.0 - support_score
                    })
                else:
                    neutral_sources.append({
                        "source": source,
                        "relevance_score": support_score
                    })
            
            # Calculate validation confidence
            total_sources = len(supporting_sources) + len(contradicting_sources) + len(neutral_sources)
            support_ratio = len(supporting_sources) / total_sources if total_sources > 0 else 0
            
            validation_confidence = support_ratio
            validation_status = "validated" if validation_confidence >= validation_threshold else "unvalidated"
            
            if len(contradicting_sources) > len(supporting_sources):
                validation_status = "contradicted"
            
            validation_results.append({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "validation_status": validation_status,
                "validation_confidence": round(validation_confidence, 3),
                "supporting_sources": supporting_sources,
                "contradicting_sources": contradicting_sources,
                "neutral_sources": neutral_sources,
                "evidence_summary": {
                    "total_sources": total_sources,
                    "supporting": len(supporting_sources),
                    "contradicting": len(contradicting_sources),
                    "neutral": len(neutral_sources)
                }
            })
        
        return {
            "status": "ok",
            "result": {
                "validation_results": validation_results,
                "overall_validation_rate": sum(1 for r in validation_results if r["validation_status"] == "validated") / len(validation_results),
                "contradicted_claims": [r for r in validation_results if r["validation_status"] == "contradicted"]
            },
            "meta": {
                "validation_threshold": validation_threshold,
                "total_claims": len(claims)
            }
        }
        
    except Exception as e:
        logger.error(f"Cross-reference validation failed: {e}")
        return {
            "status": "error",
            "error": f"Cross-reference validation failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="contradiction_detector",
    description="Detects contradictions and inconsistencies within and across sources",
    allowed_agents=["FactCheckingAgent"]
)
async def contradiction_detector_tool(tool_input: dict, context: dict) -> dict:
    """
    Build Contradiction Detector for inconsistency identification.
    
    Args:
        tool_input: {"statements": List[dict], "detection_method": str}
        context: Additional context information
    """
    try:
        statements = tool_input.get("statements", [])
        detection_method = tool_input.get("detection_method", "semantic")
        
        if len(statements) < 2:
            return {"status": "error", "error": "At least 2 statements required for contradiction detection", "meta": {}}
        
        contradictions = []
        
        # Compare statements pairwise
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                stmt1, stmt2 = statements[i], statements[j]
                
                contradiction_score = await _detect_contradiction(
                    stmt1.get("text", ""), 
                    stmt2.get("text", ""),
                    method=detection_method
                )
                
                if contradiction_score > 0.5:  # Threshold for contradiction
                    contradictions.append({
                        "statement_pair": [stmt1.get("id", i), stmt2.get("id", j)],
                        "statements": [stmt1.get("text", ""), stmt2.get("text", "")],
                        "contradiction_score": contradiction_score,
                        "contradiction_type": await _classify_contradiction_type(stmt1.get("text", ""), stmt2.get("text", "")),
                        "sources": [stmt1.get("source", ""), stmt2.get("source", "")],
                        "confidence": contradiction_score * 0.9  # Slightly reduce confidence
                    })
        
        # Analyze contradiction patterns
        patterns = await _analyze_contradiction_patterns(contradictions)
        
        return {
            "status": "ok",
            "result": {
                "contradictions": contradictions,
                "contradiction_patterns": patterns,
                "total_contradictions": len(contradictions),
                "contradiction_rate": len(contradictions) / (len(statements) * (len(statements) - 1) / 2)
            },
            "meta": {
                "detection_method": detection_method,
                "statements_analyzed": len(statements)
            }
        }
        
    except Exception as e:
        logger.error(f"Contradiction detection failed: {e}")
        return {
            "status": "error",
            "error": f"Contradiction detection failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="confidence_scorer",
    description="Calculates confidence scores for analysis results and claims",
    allowed_agents=["FactCheckingAgent"]
)
async def confidence_scorer_tool(tool_input: dict, context: dict) -> dict:
    """
    Develop Confidence Scorer for result reliability assessment.
    
    Args:
        tool_input: {"results": List[dict], "scoring_factors": List[str]}
        context: Additional context information
    """
    try:
        results = tool_input.get("results", [])
        scoring_factors = tool_input.get("scoring_factors", ["source_quality", "evidence_strength", "consistency"])
        
        if not results:
            return {"status": "error", "error": "Results are required for confidence scoring", "meta": {}}
        
        confidence_scores = []
        
        for result in results:
            factor_scores = {}
            total_confidence = 0.0
            max_confidence = 0.0
            
            # Source quality factor
            if "source_quality" in scoring_factors:
                source_quality = await _evaluate_source_quality(result.get("sources", []))
                factor_scores["source_quality"] = source_quality
                total_confidence += source_quality * 0.4
                max_confidence += 0.4
            
            # Evidence strength factor
            if "evidence_strength" in scoring_factors:
                evidence_strength = await _evaluate_evidence_strength(result.get("evidence", []))
                factor_scores["evidence_strength"] = evidence_strength
                total_confidence += evidence_strength * 0.3
                max_confidence += 0.3
            
            # Consistency factor
            if "consistency" in scoring_factors:
                consistency = await _evaluate_consistency(result.get("claims", []))
                factor_scores["consistency"] = consistency
                total_confidence += consistency * 0.3
                max_confidence += 0.3
            
            # Normalize confidence score
            final_confidence = total_confidence / max_confidence if max_confidence > 0 else 0.0
            
            confidence_scores.append({
                "result_id": result.get("id", "unknown"),
                "confidence_score": round(final_confidence, 3),
                "confidence_level": _get_confidence_level(final_confidence),
                "factor_scores": factor_scores,
                "reliability_assessment": await _generate_reliability_assessment(final_confidence, factor_scores)
            })
        
        return {
            "status": "ok",
            "result": {
                "confidence_scores": confidence_scores,
                "average_confidence": sum(s["confidence_score"] for s in confidence_scores) / len(confidence_scores),
                "high_confidence_results": [s for s in confidence_scores if s["confidence_score"] > 0.8]
            },
            "meta": {
                "scoring_factors": scoring_factors,
                "total_results": len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"Confidence scoring failed: {e}")
        return {
            "status": "error",
            "error": f"Confidence scoring failed: {str(e)}",
            "meta": {}
        }


# Helper functions for Fact-Checking tools
async def _evaluate_domain_authority(url: str) -> float:
    """Evaluate domain authority based on URL characteristics."""
    if not url:
        return 0.0
    
    # Simple domain authority heuristics
    domain_scores = {
        ".edu": 0.9,
        ".gov": 0.95,
        ".org": 0.7,
        "wikipedia.org": 0.8,
        "reuters.com": 0.85,
        "bbc.com": 0.85,
        "nature.com": 0.9,
        "science.org": 0.9
    }
    
    for domain, score in domain_scores.items():
        if domain in url.lower():
            return score
    
    # Default score for unknown domains
    return 0.5


async def _evaluate_publication_recency(publish_date) -> float:
    """Evaluate publication recency score."""
    if not publish_date:
        return 0.3  # Low score for unknown date
    
    try:
        from datetime import datetime, timedelta
        
        if isinstance(publish_date, str):
            # Simple date parsing (would use proper date parser in production)
            pub_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
        else:
            pub_date = publish_date
        
        days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
        
        # Score based on recency (higher for more recent)
        if days_old < 30:
            return 1.0
        elif days_old < 365:
            return 0.8
        elif days_old < 1825:  # 5 years
            return 0.6
        else:
            return 0.3
            
    except:
        return 0.3


async def _evaluate_author_credentials(author: str) -> float:
    """Evaluate author credentials (simplified)."""
    if not author:
        return 0.3
    
    # Simple heuristics for author credibility
    if "dr." in author.lower() or "prof." in author.lower():
        return 0.8
    elif "phd" in author.lower():
        return 0.8
    elif len(author.split()) >= 2:  # Has first and last name
        return 0.6
    else:
        return 0.4


async def _evaluate_citations(citation_count: int) -> float:
    """Evaluate citation count score."""
    if citation_count == 0:
        return 0.2
    elif citation_count < 5:
        return 0.4
    elif citation_count < 20:
        return 0.6
    elif citation_count < 100:
        return 0.8
    else:
        return 1.0


def _get_trust_level(score: float) -> str:
    """Convert numerical score to trust level."""
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    elif score >= 0.4:
        return "low"
    else:
        return "very_low"


async def _generate_credibility_recommendations(score: float, components: dict) -> List[str]:
    """Generate recommendations based on credibility score."""
    recommendations = []
    
    if score < 0.5:
        recommendations.append("Consider finding additional sources to verify information")
    
    if components.get("domain_authority", 0) < 0.5:
        recommendations.append("Source domain has low authority - verify with established sources")
    
    if components.get("publication_recency", 0) < 0.5:
        recommendations.append("Information may be outdated - check for more recent sources")
    
    return recommendations


async def _calculate_claim_support(claim: str, source_content: str) -> float:
    """Calculate how well a source supports a claim."""
    # Simple keyword overlap method (would use semantic similarity in production)
    claim_words = set(claim.lower().split())
    content_words = set(source_content.lower().split())
    
    if not claim_words or not content_words:
        return 0.0
    
    overlap = claim_words.intersection(content_words)
    return len(overlap) / len(claim_words)


async def _detect_contradiction(stmt1: str, stmt2: str, method: str = "semantic") -> float:
    """Detect contradiction between two statements."""
    # Simple contradiction detection based on negation patterns
    negation_patterns = ["not", "no", "never", "none", "neither", "without"]
    
    stmt1_words = set(stmt1.lower().split())
    stmt2_words = set(stmt2.lower().split())
    
    # Check for negation patterns
    stmt1_negated = any(neg in stmt1_words for neg in negation_patterns)
    stmt2_negated = any(neg in stmt2_words for neg in negation_patterns)
    
    # Simple heuristic: if one is negated and they share keywords, likely contradiction
    shared_words = stmt1_words.intersection(stmt2_words)
    
    if stmt1_negated != stmt2_negated and len(shared_words) > 2:
        return 0.8
    elif stmt1_negated and stmt2_negated and len(shared_words) > 2:
        return 0.3  # Both negated, less likely to contradict
    else:
        return 0.2


async def _classify_contradiction_type(stmt1: str, stmt2: str) -> str:
    """Classify the type of contradiction."""
    # Simple classification based on content patterns
    if "increase" in stmt1.lower() and "decrease" in stmt2.lower():
        return "directional"
    elif any(word in stmt1.lower() for word in ["yes", "true", "correct"]) and any(word in stmt2.lower() for word in ["no", "false", "incorrect"]):
        return "boolean"
    else:
        return "semantic"


async def _analyze_contradiction_patterns(contradictions: List[dict]) -> dict:
    """Analyze patterns in contradictions."""
    patterns = {
        "most_common_type": "semantic",
        "source_conflicts": {},
        "topic_conflicts": []
    }
    
    # Count contradiction types
    type_counts = {}
    for contradiction in contradictions:
        cont_type = contradiction.get("contradiction_type", "unknown")
        type_counts[cont_type] = type_counts.get(cont_type, 0) + 1
    
    if type_counts:
        patterns["most_common_type"] = max(type_counts, key=type_counts.get)
    
    return patterns


async def _evaluate_source_quality(sources: List[dict]) -> float:
    """Evaluate overall quality of sources."""
    if not sources:
        return 0.0
    
    quality_scores = []
    for source in sources:
        # Simple quality evaluation
        score = 0.5  # Base score
        
        if source.get("credibility_score"):
            score = source["credibility_score"]
        elif "url" in source:
            score = await _evaluate_domain_authority(source["url"])
        
        quality_scores.append(score)
    
    return sum(quality_scores) / len(quality_scores)


async def _evaluate_evidence_strength(evidence: List[dict]) -> float:
    """Evaluate strength of evidence."""
    if not evidence:
        return 0.0
    
    # Simple evidence strength based on quantity and type
    strength = min(len(evidence) / 5.0, 1.0)  # More evidence = stronger (up to 5 pieces)
    
    # Bonus for diverse evidence types
    evidence_types = set(e.get("type", "unknown") for e in evidence)
    diversity_bonus = min(len(evidence_types) / 3.0, 0.2)  # Up to 20% bonus
    
    return min(strength + diversity_bonus, 1.0)


async def _evaluate_consistency(claims: List[dict]) -> float:
    """Evaluate consistency across claims."""
    if len(claims) < 2:
        return 1.0  # Single claim is consistent by definition
    
    # Simple consistency check based on contradiction detection
    contradictions = 0
    total_pairs = 0
    
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            total_pairs += 1
            contradiction_score = await _detect_contradiction(
                claims[i].get("text", ""),
                claims[j].get("text", "")
            )
            if contradiction_score > 0.5:
                contradictions += 1
    
    if total_pairs == 0:
        return 1.0
    
    return 1.0 - (contradictions / total_pairs)


def _get_confidence_level(score: float) -> str:
    """Convert confidence score to level."""
    if score >= 0.9:
        return "very_high"
    elif score >= 0.7:
        return "high"
    elif score >= 0.5:
        return "medium"
    elif score >= 0.3:
        return "low"
    else:
        return "very_low"


async def _generate_reliability_assessment(confidence: float, factors: dict) -> dict:
    """Generate reliability assessment based on confidence and factors."""
    assessment = {
        "overall_reliability": _get_confidence_level(confidence),
        "strengths": [],
        "weaknesses": [],
        "recommendations": []
    }
    
    # Identify strengths and weaknesses
    for factor, score in factors.items():
        if score > 0.7:
            assessment["strengths"].append(f"Strong {factor.replace('_', ' ')}")
        elif score < 0.4:
            assessment["weaknesses"].append(f"Weak {factor.replace('_', ' ')}")
    
    # Generate recommendations
    if confidence < 0.6:
        assessment["recommendations"].append("Seek additional verification")
    if factors.get("source_quality", 0) < 0.5:
        assessment["recommendations"].append("Use higher quality sources")
    
    return assessment# ===
#=========================================================================
# Output Formatting Agent Tools
# ============================================================================

@mcp_tool(
    name="citation_formatter",
    description="Formats citations in IEEE style with proper numbering and bibliography",
    allowed_agents=["OutputFormattingAgent"]
)
async def citation_formatter_tool(tool_input: dict, context: dict) -> dict:
    """
    Create Citation Formatter for IEEE-style citations.
    
    Args:
        tool_input: {"sources": List[dict], "citation_style": str}
        context: Additional context information
    """
    try:
        sources = tool_input.get("sources", [])
        citation_style = tool_input.get("citation_style", "ieee")
        
        if not sources:
            return {"status": "error", "error": "Sources are required for citation formatting", "meta": {}}
        
        formatted_citations = []
        bibliography = []
        citation_map = {}
        
        for i, source in enumerate(sources, 1):
            citation_number = i
            citation_map[source.get("id", str(i))] = citation_number
            
            # Format IEEE citation
            ieee_citation = await _format_ieee_citation(source, citation_number)
            
            formatted_citations.append({
                "citation_number": citation_number,
                "source_id": source.get("id", str(i)),
                "inline_citation": f"[{citation_number}]",
                "full_citation": ieee_citation,
                "url": source.get("url", ""),
                "title": source.get("title", "")
            })
            
            bibliography.append(ieee_citation)
        
        return {
            "status": "ok",
            "result": {
                "formatted_citations": formatted_citations,
                "bibliography": bibliography,
                "citation_map": citation_map,
                "total_citations": len(formatted_citations)
            },
            "meta": {
                "citation_style": citation_style,
                "formatting_standard": "IEEE"
            }
        }
        
    except Exception as e:
        logger.error(f"Citation formatting failed: {e}")
        return {
            "status": "error",
            "error": f"Citation formatting failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="report_structuring",
    description="Creates hierarchical outline and structures comprehensive reports",
    allowed_agents=["OutputFormattingAgent"]
)
async def report_structuring_tool(tool_input: dict, context: dict) -> dict:
    """
    Implement Report Structuring Tool for hierarchical organization.
    
    Args:
        tool_input: {"content": dict, "structure_type": str, "sections": List[str]}
        context: Additional context information
    """
    try:
        content = tool_input.get("content", {})
        structure_type = tool_input.get("structure_type", "research_report")
        custom_sections = tool_input.get("sections", [])
        
        if not content:
            return {"status": "error", "error": "Content is required for report structuring", "meta": {}}
        
        # Define report structure based on type
        if structure_type == "research_report":
            sections = custom_sections or [
                "Executive Summary",
                "Introduction", 
                "Methodology",
                "Key Findings",
                "Analysis",
                "Discussion",
                "Conclusions",
                "References"
            ]
        elif structure_type == "analysis_brief":
            sections = custom_sections or [
                "Summary",
                "Key Points",
                "Supporting Evidence",
                "Implications",
                "Sources"
            ]
        else:
            sections = custom_sections or ["Introduction", "Main Content", "Conclusion", "References"]
        
        structured_report = {
            "title": content.get("title", "Research Report"),
            "sections": [],
            "metadata": {
                "structure_type": structure_type,
                "total_sections": len(sections),
                "word_count": 0
            }
        }
        
        # Organize content into sections
        for section_name in sections:
            section_content = await _extract_section_content(content, section_name)
            
            section = {
                "section_number": len(structured_report["sections"]) + 1,
                "title": section_name,
                "content": section_content,
                "subsections": await _identify_subsections(section_content),
                "word_count": len(section_content.split()) if section_content else 0
            }
            
            structured_report["sections"].append(section)
            structured_report["metadata"]["word_count"] += section["word_count"]
        
        # Generate table of contents
        toc = await _generate_table_of_contents(structured_report["sections"])
        
        return {
            "status": "ok",
            "result": {
                "structured_report": structured_report,
                "table_of_contents": toc,
                "formatting_guidelines": await _get_formatting_guidelines(structure_type)
            },
            "meta": {
                "structure_type": structure_type,
                "total_word_count": structured_report["metadata"]["word_count"]
            }
        }
        
    except Exception as e:
        logger.error(f"Report structuring failed: {e}")
        return {
            "status": "error",
            "error": f"Report structuring failed: {str(e)}",
            "meta": {}
        }


@mcp_tool(
    name="executive_summary_generator",
    description="Generates concise executive summaries from detailed analysis results",
    allowed_agents=["OutputFormattingAgent"]
)
async def executive_summary_generator_tool(tool_input: dict, context: dict) -> dict:
    """
    Build Executive Summary Generator for result summarization.
    
    Args:
        tool_input: {"analysis_results": dict, "summary_length": str, "key_points": List[str]}
        context: Additional context information
    """
    try:
        analysis_results = tool_input.get("analysis_results", {})
        summary_length = tool_input.get("summary_length", "medium")  # short, medium, long
        key_points = tool_input.get("key_points", [])
        
        if not analysis_results:
            return {"status": "error", "error": "Analysis results are required", "meta": {}}
        
        # Define summary parameters based on length
        length_params = {
            "short": {"max_sentences": 3, "max_words": 150},
            "medium": {"max_sentences": 6, "max_words": 300},
            "long": {"max_sentences": 10, "max_words": 500}
        }
        
        params = length_params.get(summary_length, length_params["medium"])
        
        # Extract key information from analysis results
        summary_components = await _extract_summary_components(analysis_results, key_points)
        
        # Generate executive summary
        executive_summary = {
            "title": "Executive Summary",
            "overview": await _generate_overview(summary_components, params),
            "key_findings": await _generate_key_findings(summary_components, params),
            "recommendations": await _generate_recommendations(summary_components),
            "confidence_assessment": await _generate_confidence_assessment(analysis_results),
            "metadata": {
                "summary_length": summary_length,
                "word_count": 0,
                "confidence_score": analysis_results.get("confidence_score", 0.0)
            }
        }
        
        # Calculate total word count
        total_words = (
            len(executive_summary["overview"].split()) +
            sum(len(finding.split()) for finding in executive_summary["key_findings"]) +
            sum(len(rec.split()) for rec in executive_summary["recommendations"])
        )
        executive_summary["metadata"]["word_count"] = total_words
        
        # Generate formatted summary text
        formatted_summary = await _format_executive_summary(executive_summary)
        
        return {
            "status": "ok",
            "result": {
                "executive_summary": executive_summary,
                "formatted_summary": formatted_summary,
                "summary_statistics": {
                    "word_count": total_words,
                    "key_findings_count": len(executive_summary["key_findings"]),
                    "recommendations_count": len(executive_summary["recommendations"])
                }
            },
            "meta": {
                "summary_length": summary_length,
                "generation_method": "structured_extraction"
            }
        }
        
    except Exception as e:
        logger.error(f"Executive summary generation failed: {e}")
        return {
            "status": "error",
            "error": f"Executive summary generation failed: {str(e)}",
            "meta": {}
        }


# Helper functions for Output Formatting tools
async def _format_ieee_citation(source: dict, citation_number: int) -> str:
    """Format a source in IEEE citation style."""
    # Extract source information
    authors = source.get("authors", [])
    title = source.get("title", "")
    url = source.get("url", "")
    publish_date = source.get("publish_date", "")
    publisher = source.get("publisher", "")
    
    # Format authors
    if authors:
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) <= 3:
            author_str = ", ".join(authors[:-1]) + " and " + authors[-1]
        else:
            author_str = authors[0] + " et al."
    else:
        author_str = "Anonymous"
    
    # Format date
    if publish_date:
        try:
            if isinstance(publish_date, str):
                year = publish_date[:4] if len(publish_date) >= 4 else publish_date
            else:
                year = str(publish_date.year)
        except:
            year = "n.d."
    else:
        year = "n.d."
    
    # Build IEEE citation
    citation_parts = [f"[{citation_number}]"]
    
    if author_str:
        citation_parts.append(f'{author_str},')
    
    if title:
        citation_parts.append(f'"{title},"')
    
    if publisher:
        citation_parts.append(f'{publisher},')
    
    citation_parts.append(f'{year}.')
    
    if url:
        citation_parts.append(f'[Online]. Available: {url}')
    
    return " ".join(citation_parts)


async def _extract_section_content(content: dict, section_name: str) -> str:
    """Extract content for a specific section."""
    section_key = section_name.lower().replace(" ", "_")
    
    # Try to find content by section name
    if section_key in content:
        return str(content[section_key])
    
    # Map common section names to content keys
    section_mapping = {
        "executive_summary": ["summary", "overview", "abstract"],
        "introduction": ["intro", "background", "context"],
        "methodology": ["method", "approach", "process"],
        "key_findings": ["findings", "results", "discoveries"],
        "analysis": ["analysis", "interpretation", "discussion"],
        "conclusions": ["conclusion", "summary", "final_thoughts"],
        "references": ["sources", "citations", "bibliography"]
    }
    
    if section_key in section_mapping:
        for key in section_mapping[section_key]:
            if key in content:
                return str(content[key])
    
    # Default content if no specific match found
    if section_name == "Executive Summary":
        return await _generate_default_summary(content)
    elif section_name == "Key Findings":
        return await _generate_default_findings(content)
    else:
        return f"Content for {section_name} section."


async def _identify_subsections(content: str) -> List[dict]:
    """Identify potential subsections within content."""
    if not content or len(content) < 200:
        return []
    
    # Simple subsection identification based on length
    paragraphs = content.split('\n\n')
    subsections = []
    
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph) > 100:  # Substantial paragraph
            subsections.append({
                "subsection_number": i + 1,
                "title": f"Subsection {i + 1}",
                "content": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph
            })
    
    return subsections[:3]  # Limit to 3 subsections


async def _generate_table_of_contents(sections: List[dict]) -> List[dict]:
    """Generate table of contents from sections."""
    toc = []
    
    for section in sections:
        toc_entry = {
            "section_number": section["section_number"],
            "title": section["title"],
            "page_number": section["section_number"],  # Simplified page numbering
            "subsections": []
        }
        
        for subsection in section.get("subsections", []):
            toc_entry["subsections"].append({
                "title": subsection["title"],
                "page_number": f"{section['section_number']}.{subsection['subsection_number']}"
            })
        
        toc.append(toc_entry)
    
    return toc


async def _get_formatting_guidelines(structure_type: str) -> dict:
    """Get formatting guidelines for different report types."""
    guidelines = {
        "research_report": {
            "font": "Times New Roman, 12pt",
            "spacing": "Double-spaced",
            "margins": "1 inch all sides",
            "citation_style": "IEEE",
            "page_numbering": "Bottom center"
        },
        "analysis_brief": {
            "font": "Arial, 11pt",
            "spacing": "Single-spaced",
            "margins": "0.75 inch all sides",
            "citation_style": "IEEE",
            "page_numbering": "Top right"
        }
    }
    
    return guidelines.get(structure_type, guidelines["research_report"])


async def _extract_summary_components(analysis_results: dict, key_points: List[str]) -> dict:
    """Extract components for executive summary."""
    components = {
        "main_findings": [],
        "key_metrics": {},
        "conclusions": [],
        "evidence": [],
        "confidence_indicators": {}
    }
    
    # Extract findings
    if "findings" in analysis_results:
        components["main_findings"] = analysis_results["findings"][:5]  # Top 5 findings
    
    # Extract key metrics
    if "statistics" in analysis_results:
        components["key_metrics"] = analysis_results["statistics"]
    
    # Extract conclusions
    if "conclusions" in analysis_results:
        components["conclusions"] = analysis_results["conclusions"]
    
    # Extract evidence
    if "evidence" in analysis_results:
        components["evidence"] = analysis_results["evidence"][:3]  # Top 3 pieces of evidence
    
    # Extract confidence indicators
    components["confidence_indicators"] = {
        "overall_confidence": analysis_results.get("confidence_score", 0.0),
        "source_quality": analysis_results.get("source_quality", 0.0),
        "evidence_strength": analysis_results.get("evidence_strength", 0.0)
    }
    
    return components


async def _generate_overview(components: dict, params: dict) -> str:
    """Generate overview section of executive summary."""
    findings = components.get("main_findings", [])
    confidence = components.get("confidence_indicators", {}).get("overall_confidence", 0.0)
    
    if not findings:
        return "This analysis examines the available data and provides insights based on the research conducted."
    
    # Create overview based on top findings
    overview_parts = []
    
    if confidence > 0.7:
        overview_parts.append("This comprehensive analysis reveals several key insights.")
    elif confidence > 0.5:
        overview_parts.append("This analysis provides important findings with moderate confidence.")
    else:
        overview_parts.append("This preliminary analysis offers initial insights that require further validation.")
    
    # Add top finding
    if findings:
        top_finding = findings[0] if isinstance(findings[0], str) else str(findings[0])
        overview_parts.append(f"The primary finding indicates that {top_finding.lower()}")
    
    overview = " ".join(overview_parts)
    
    # Truncate if too long
    words = overview.split()
    if len(words) > params["max_words"] // 2:
        overview = " ".join(words[:params["max_words"] // 2]) + "..."
    
    return overview


async def _generate_key_findings(components: dict, params: dict) -> List[str]:
    """Generate key findings list."""
    findings = components.get("main_findings", [])
    max_findings = min(params["max_sentences"] - 1, len(findings), 5)
    
    key_findings = []
    for i, finding in enumerate(findings[:max_findings]):
        finding_text = finding if isinstance(finding, str) else str(finding)
        key_findings.append(f"{finding_text}")
    
    if not key_findings:
        key_findings = ["No specific findings were identified in the analysis."]
    
    return key_findings


async def _generate_recommendations(components: dict) -> List[str]:
    """Generate recommendations based on analysis."""
    confidence = components.get("confidence_indicators", {}).get("overall_confidence", 0.0)
    findings = components.get("main_findings", [])
    
    recommendations = []
    
    if confidence < 0.5:
        recommendations.append("Conduct additional research to validate preliminary findings")
    
    if len(findings) > 3:
        recommendations.append("Focus on the most significant findings for immediate action")
    
    if not recommendations:
        recommendations.append("Continue monitoring developments in this area")
    
    return recommendations[:3]  # Limit to 3 recommendations


async def _generate_confidence_assessment(analysis_results: dict) -> dict:
    """Generate confidence assessment for the summary."""
    confidence_score = analysis_results.get("confidence_score", 0.0)
    
    assessment = {
        "overall_confidence": confidence_score,
        "confidence_level": _get_confidence_level(confidence_score),
        "reliability_factors": [],
        "limitations": []
    }
    
    # Add reliability factors
    if confidence_score > 0.7:
        assessment["reliability_factors"].append("High-quality sources")
        assessment["reliability_factors"].append("Consistent findings across sources")
    elif confidence_score > 0.5:
        assessment["reliability_factors"].append("Moderate source quality")
    
    # Add limitations
    if confidence_score < 0.6:
        assessment["limitations"].append("Limited source diversity")
    if confidence_score < 0.4:
        assessment["limitations"].append("Insufficient evidence for strong conclusions")
    
    return assessment


async def _format_executive_summary(summary: dict) -> str:
    """Format executive summary as readable text."""
    formatted_parts = []
    
    # Title
    formatted_parts.append("# Executive Summary\n")
    
    # Overview
    formatted_parts.append("## Overview")
    formatted_parts.append(summary["overview"])
    formatted_parts.append("")
    
    # Key Findings
    formatted_parts.append("## Key Findings")
    for i, finding in enumerate(summary["key_findings"], 1):
        formatted_parts.append(f"{i}. {finding}")
    formatted_parts.append("")
    
    # Recommendations
    if summary["recommendations"]:
        formatted_parts.append("## Recommendations")
        for i, rec in enumerate(summary["recommendations"], 1):
            formatted_parts.append(f"{i}. {rec}")
        formatted_parts.append("")
    
    # Confidence Assessment
    confidence = summary["confidence_assessment"]
    formatted_parts.append("## Confidence Assessment")
    formatted_parts.append(f"Overall Confidence: {confidence['confidence_level'].title()} ({confidence['overall_confidence']:.2f})")
    
    return "\n".join(formatted_parts)


async def _generate_default_summary(content: dict) -> str:
    """Generate default summary when no specific summary is provided."""
    return "This section provides a summary of the key points and findings from the analysis."


async def _generate_default_findings(content: dict) -> str:
    """Generate default findings when no specific findings are provided."""
    return "Key findings will be presented based on the analysis of available data and sources."