"""
Individual agent implementations for the multi-agent research system.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

from app.mcp import mcp_registry
from app.workers.vector_store import retrieve_with_confidence
from app.config import config
from app.mcp_tools import semantic_search_tool

logger = logging.getLogger(__name__)


class ResearchCoordinatorAgent:
    """Research Coordinator Agent - orchestrates the research workflow."""
    
    def __init__(self):
        self.name = "ResearchCoordinatorAgent"
        self.allowed_tools = [
            "query_decomposer",
            "task_prioritizer", 
            "progress_tracker",
            "result_synthesis"
        ]
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Research Coordinator Agent."""
        try:
            logger.info(f"ResearchCoordinator processing query: {state['query'][:50]}...")
            
            # Step 1: Decompose query if not already done
            if not state.get("subqueries"):
                decomposition_result = await mcp_registry.execute_tool(
                    "query_decomposer",
                    self.name,
                    {"query": state["query"], "max_subqueries": 5},
                    {"task_id": state["task_id"]}
                )
                
                if decomposition_result["status"] == "ok":
                    subqueries_data = decomposition_result["result"]["subqueries"]
                    state["subqueries"] = [sq["subquery"] for sq in subqueries_data]
                    
                    # Prioritize the subqueries using GPT-4o
                    prioritization_result = await mcp_registry.execute_tool(
                        "task_prioritizer",
                        self.name,
                        {
                            "tasks": subqueries_data,
                            "main_query": state["query"]
                        },
                        {"task_id": state["task_id"]}
                    )
                    
                    if prioritization_result["status"] == "ok":
                        # Update subqueries with prioritized order
                        prioritized_tasks = prioritization_result["result"]["prioritized_tasks"]
                        state["subqueries"] = [task["task"]["subquery"] for task in prioritized_tasks]
                
                if decomposition_result["status"] == "ok":
                    state["subqueries"] = [
                        sq["subquery"] for sq in decomposition_result["result"]["subqueries"]
                    ]
                else:
                    state["subqueries"] = [state["query"]]
            
            # Step 2: Initial RAG retrieval
            if not state.get("retrieval_results"):
                retrieval_result = await semantic_search_tool(
                    tool_input = state["query"],
                    context = {}
                )
                
                state["retrieval_results"] = {
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "document_id": r.document_id,
                            "chunk_text": r.chunk_text,
                            "similarity_score": r.similarity_score,
                            "document_title": r.document_title,
                            "document_url": r.document_url,
                            "chunk_meta": r.chunk_meta,
                            "token_count": r.token_count
                        }
                        for r in retrieval_result.results
                    ],
                    "confidence_score": retrieval_result.confidence_score,
                    "total_results": retrieval_result.total_results
                }
                
                state["confidence_score"] = retrieval_result.confidence_score
            
            # Add coordinator message
            decomp_method = decomposition_result.get("meta", {}).get("decomposition_method", "unknown") if decomposition_result["status"] == "ok" else "failed"
            state["messages"].append({
                "role": "system",
                "content": f"Query intelligently decomposed using {decomp_method} into {len(state['subqueries'])} prioritized subqueries. Initial retrieval confidence: {state['confidence_score']:.2f}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"ResearchCoordinator failed: {e}")
            state["error"] = f"Coordinator error: {str(e)}"
            return state


class WebScraperRetrievalAgent:
    """Web Scraper & Document Retrieval Agent."""
    
    def __init__(self):
        self.name = "WebScraperRetrievalAgent"
        self.allowed_tools = [
            "intelligent_url_generator",
            "semantic_search",
            "web_scraper",
            "rag_upsert"
        ]
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Web Scraper Agent."""
        try:
            logger.info(f"WebScraperAgent processing query: {state['query'][:50]}...")
            
            # Generate intelligent URLs using LLM
            url_generation_result = await mcp_registry.execute_tool(
                "intelligent_url_generator",
                self.name,
                {
                    "query": state["query"],
                    "subqueries": state.get("subqueries", []),
                    "max_urls": 8,
                    "domains": []  # Let LLM choose the best domains
                },
                {"task_id": state["task_id"]}
            )
            
            # Extract URLs from the generation result
            if url_generation_result["status"] == "ok":
                generated_url_data = url_generation_result["result"]["generated_urls"]
                search_urls = [url_data["url"] for url_data in generated_url_data]
                
                logger.info(f"Generated {len(search_urls)} intelligent URLs using {url_generation_result['meta'].get('generation_method', 'unknown')} method")
                
                # Log some example URLs for debugging
                for i, url_data in enumerate(generated_url_data[:3]):
                    logger.info(f"  URL {i+1}: {url_data['url']} ({url_data['content_type']}) - {url_data['description'][:100]}...")
            else:
                # Fallback to simple URL generation if LLM fails
                logger.warning(f"Intelligent URL generation failed: {url_generation_result.get('error', 'Unknown error')}")
                search_urls = self._generate_search_urls(state.get("subqueries", [state["query"]]))
            
            # Scrape URLs
            scraping_result = await mcp_registry.execute_tool(
                "web_scraper",
                self.name,
                {"urls": search_urls[:5], "max_pages": 5},
                {"task_id": state["task_id"]}
            )
            
            if scraping_result["status"] == "ok":
                scraped_docs = scraping_result["result"]["scraped_documents"]
                state["scraped_documents"] = scraped_docs
                
                # Upsert to RAG if we have documents
                if scraped_docs:
                    upsert_result = await mcp_registry.execute_tool(
                        "rag_upsert",
                        self.name,
                        {"documents": scraped_docs},
                        {"task_id": state["task_id"]}
                    )
                    
                    if upsert_result["status"] == "ok":
                        # Re-run retrieval with new content
                        retrieval_result = await semantic_search_tool(
                            tool_input = state["query"],
                            context = {}
                        )
                
                        
                        # Update retrieval results
                        state["retrieval_results"] = {
                            "results": [
                                {
                                    "chunk_id": r.chunk_id,
                                    "document_id": r.document_id,
                                    "chunk_text": r.chunk_text,
                                    "similarity_score": r.similarity_score,
                                    "document_title": r.document_title,
                                    "document_url": r.document_url,
                                    "chunk_meta": r.chunk_meta,
                                    "token_count": r.token_count
                                }
                                for r in retrieval_result.results
                            ],
                            "confidence_score": retrieval_result.confidence_score,
                            "total_results": retrieval_result.total_results,
                            "method": "post_scraping"
                        }
                        
                        state["confidence_score"] = retrieval_result.confidence_score
            
            # Add scraper message
            url_gen_method = url_generation_result.get("meta", {}).get("generation_method", "fallback") if url_generation_result["status"] == "ok" else "fallback"
            state["messages"].append({
                "role": "system",
                "content": f"Generated {len(search_urls)} URLs using {url_gen_method}. "
                          f"Scraped {len(state.get('scraped_documents', []))} documents. "
                          f"Updated confidence: {state['confidence_score']:.2f}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"WebScraperAgent failed: {e}")
            state["error"] = f"Scraper error: {str(e)}"
            return state
    
    def _generate_search_urls(self, subqueries: List[str]) -> List[str]:
        """Generate search URLs from subqueries."""
        search_urls = []
        
        for query in subqueries[:3]:
            search_terms = query.replace(" ", "+")
            # In production, use actual search APIs
            example_urls = [
                f"https://example.com/search?q={search_terms}",
                f"https://research.example.org/articles/{search_terms}",
            ]
            search_urls.extend(example_urls)
        
        return search_urls


class DeepAnalysisAgent:
    """Deep Analysis Agent - performs complex analysis."""
    
    def __init__(self):
        self.name = "DeepAnalysisAgent"
        self.allowed_tools = [
            "comprehensive_analysis",
            "semantic_search"
        ]
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Deep Analysis Agent."""
        try:
            logger.info(f"DeepAnalysisAgent processing analysis...")
            
            retrieval_results = state.get("retrieval_results", {}).get("results", [])
            
            if not retrieval_results:
                state["analysis_results"] = {"error": "No retrieval results to analyze"}
                return state
            
            # Prepare documents for comprehensive analysis
            context_documents = []
            for result in retrieval_results:
                context_documents.append({
                    "chunk_id": result["chunk_id"],
                    "document_title": result["document_title"],
                    "content": result["chunk_text"],
                    "document_url": result["document_url"],
                    "similarity_score": result["similarity_score"],
                    "token_count": result.get("token_count", 0)
                })
            
            # Perform comprehensive analysis using GPT-4o
            analysis_result = await mcp_registry.execute_tool(
                "comprehensive_analysis",
                self.name,
                {
                    "query": state["query"],
                    "context_documents": context_documents,
                    "analysis_type": "comprehensive_research"
                },
                {"task_id": state["task_id"]}
            )
            
            if analysis_result["status"] == "ok":
                state["analysis_results"] = {
                    "comprehensive": analysis_result["result"],
                    "source_documents": len(context_documents),
                    "analysis_confidence": 0.8  # High confidence for GPT-4o analysis
                }
            else:
                state["analysis_results"] = {
                    "error": analysis_result.get("error", "Analysis failed"),
                    "source_documents": len(context_documents)
                }
            
            # Add analysis message
            analysis_results = state.get("analysis_results", {})
            word_count = analysis_results.get("comprehensive", {}).get("word_count", 0)
            
            state["messages"].append({
                "role": "system",
                "content": f"Generated comprehensive analysis ({word_count} words) using GPT-4o with {len(context_documents)} source documents.",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"DeepAnalysisAgent failed: {e}")
            state["error"] = f"Analysis error: {str(e)}"
            return state
    
    def _extract_data_points(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract data points for trend analysis."""
        data_points = []
        
        for doc in documents:
            years = re.findall(r'\b(20\d{2})\b', doc["content"])
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:billion|million|%)\b', doc["content"])
            
            for year in years[:2]:
                for number in numbers[:2]:
                    try:
                        data_points.append({
                            "date": year,
                            "value": float(number),
                            "source": doc["title"]
                        })
                    except ValueError:
                        continue
        
        return data_points[:10]
    
    def _extract_events(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Extract events for causal analysis."""
        events = []
        
        for doc in documents:
            event_keywords = ["impact", "effect", "result", "consequence"]
            
            for keyword in event_keywords:
                if keyword in doc["content"].lower():
                    events.append({
                        "description": f"Event related to {keyword} in {doc['title']}",
                        "source": doc["title"],
                        "relevance": doc["similarity"]
                    })
        
        return events[:5]
    
    def _extract_causes(self, query: str) -> List[str]:
        """Extract potential causes from query."""
        causes = []
        
        if "covid" in query.lower():
            causes.extend(["COVID-19 pandemic", "economic disruption"])
        
        if "climate" in query.lower():
            causes.extend(["climate change", "environmental regulations"])
        
        return causes[:3]
    
    def _extract_effects(self, query: str) -> List[str]:
        """Extract potential effects from query."""
        effects = []
        
        if "investment" in query.lower():
            effects.extend(["investment changes", "funding shifts"])
        
        if "policy" in query.lower():
            effects.extend(["policy changes", "regulatory updates"])
        
        return effects[:3]


class FactCheckingAgent:
    """Fact-Checking & Validation Agent."""
    
    def __init__(self):
        self.name = "FactCheckingAgent"
        self.allowed_tools = [
            "source_credibility_checker",
            "cross_reference_validator",
            "contradiction_detector",
            "confidence_scorer"
        ]
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Fact-Checking Agent."""
        try:
            logger.info("FactCheckingAgent processing analysis results...")
            
            analysis_results = state.get("analysis_results", {})
            retrieval_results = state.get("retrieval_results", {}).get("results", [])
            
            if not analysis_results or not retrieval_results:
                state["fact_check_results"] = {"error": "No analysis results to fact-check"}
                return state
            
            fact_check_results = {}
            
            # Check source credibility
            sources = []
            for result in retrieval_results:
                sources.append({
                    "id": result["chunk_id"],
                    "url": result["document_url"],
                    "title": result["document_title"],
                    "similarity_score": result["similarity_score"]
                })
            
            credibility_result = await mcp_registry.execute_tool(
                "source_credibility_checker",
                self.name,
                {
                    "sources": sources,
                    "query_context": state["query"]
                },
                {"task_id": state["task_id"]}
            )
            
            if credibility_result["status"] == "ok":
                fact_check_results["credibility"] = credibility_result["result"]
            
            # Cross-reference validation
            claims = self._extract_claims(analysis_results)
            
            if claims:
                validation_result = await mcp_registry.execute_tool(
                    "cross_reference_validator",
                    self.name,
                    {
                        "claims": claims,
                        "sources": [{"content": r["chunk_text"], **r} for r in retrieval_results]
                    },
                    {"task_id": state["task_id"]}
                )
                
                if validation_result["status"] == "ok":
                    fact_check_results["validation"] = validation_result["result"]
            
            # Contradiction detection
            statements = self._extract_statements(retrieval_results)
            
            if len(statements) > 1:
                contradiction_result = await mcp_registry.execute_tool(
                    "contradiction_detector",
                    self.name,
                    {"statements": statements},
                    {"task_id": state["task_id"]}
                )
                
                if contradiction_result["status"] == "ok":
                    fact_check_results["contradictions"] = contradiction_result["result"]
            
            # Overall confidence scoring
            confidence_result = await mcp_registry.execute_tool(
                "confidence_scorer",
                self.name,
                {
                    "results": [
                        {
                            "id": "analysis",
                            "sources": sources,
                            "evidence": retrieval_results,
                            "claims": claims
                        }
                    ]
                },
                {"task_id": state["task_id"]}
            )
            
            if confidence_result["status"] == "ok":
                fact_check_results["confidence"] = confidence_result["result"]
                
                # Update overall confidence score
                confidence_scores = confidence_result["result"]["confidence_scores"]
                if confidence_scores:
                    state["confidence_score"] = confidence_scores[0]["confidence_score"]
            
            state["fact_check_results"] = fact_check_results
            
            # Add fact-check message
            contradictions = len(fact_check_results.get("contradictions", {}).get("contradictions", []))
            
            state["messages"].append({
                "role": "system",
                "content": f"Fact-checking complete. Final confidence: {state['confidence_score']:.2f}. "
                          f"Found {contradictions} contradictions.",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"FactCheckingAgent failed: {e}")
            state["error"] = f"Fact-check error: {str(e)}"
            return state
    
    def _extract_claims(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract claims from analysis results."""
        claims = []
        
        # Extract from comparative analysis
        comparative = analysis_results.get("comparative", {})
        if "insights" in comparative:
            for i, insight in enumerate(comparative["insights"]):
                claims.append({
                    "id": f"comparative_{i}",
                    "text": insight.get("description", ""),
                    "type": "comparative"
                })
        
        return claims[:5]
    
    def _extract_statements(self, retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract statements for contradiction detection."""
        statements = []
        
        for i, result in enumerate(retrieval_results[:3]):
            sentences = result["chunk_text"].split('. ')
            
            for j, sentence in enumerate(sentences[:2]):
                if len(sentence.strip()) > 20:
                    statements.append({
                        "id": f"stmt_{i}_{j}",
                        "text": sentence.strip(),
                        "source": result["document_title"]
                    })
        
        return statements


class OutputFormattingAgent:
    """Output Formatting Agent - creates final formatted output."""
    
    def __init__(self):
        self.name = "OutputFormattingAgent"
        self.allowed_tools = [
            "report_structuring",
            "citation_formatter",
            "executive_summary_generator"
        ]
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Output Formatting Agent."""
        try:
            logger.info("OutputFormattingAgent creating final output...")
            
            # Collect all results
            analysis_results = state.get("analysis_results", {})
            fact_check_results = state.get("fact_check_results", {})
            retrieval_results = state.get("retrieval_results", {}).get("results", [])
            
            # Format citations
            sources = []
            for result in retrieval_results:
                sources.append({
                    "id": result["chunk_id"],
                    "title": result["document_title"],
                    "url": result["document_url"],
                    "authors": [],
                    "publish_date": None
                })
            
            citation_result = await mcp_registry.execute_tool(
                "citation_formatter",
                self.name,
                {"sources": sources},
                {"task_id": state["task_id"]}
            )
            
            citations = []
            if citation_result["status"] == "ok":
                citations = citation_result["result"]["bibliography"]
            
            # Create executive summary
            summary_result = await mcp_registry.execute_tool(
                "executive_summary_generator",
                self.name,
                {
                    "analysis_results": {
                        **analysis_results,
                        "confidence_score": state["confidence_score"],
                        "fact_check": fact_check_results
                    }
                },
                {"task_id": state["task_id"]}
            )
            
            executive_summary = ""
            if summary_result["status"] == "ok":
                executive_summary = summary_result["result"]["formatted_summary"]
            
            # Create final result
            answer_markdown = self._create_final_markdown(
                state["query"],
                executive_summary,
                analysis_results,
                citations,
                retrieval_results
            )
            
            # Create provenance records
            provenance = []
            for i, result in enumerate(retrieval_results):
                provenance.append({
                    "claim": f"Information from {result['document_title']}",
                    "chunk_id": result["chunk_id"],
                    "document_title": result["document_title"],
                    "document_url": result["document_url"],
                    "similarity_score": result["similarity_score"],
                    "citation_number": i + 1
                })
            
            state["final_result"] = {
                "answer_markdown": answer_markdown,
                "citations": citations,
                "provenance": provenance,
                "confidence_score": state["confidence_score"],
                "processing_time": 0.0,
                "total_sources": len(retrieval_results),
                "search_method": state.get("retrieval_results", {}).get("method", "semantic")
            }
            
            # Add final message
            state["messages"].append({
                "role": "system",
                "content": f"Final report generated with {len(citations)} citations.",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"OutputFormattingAgent failed: {e}")
            state["error"] = f"Formatting error: {str(e)}"
            return state
    
    def _create_final_markdown(
        self,
        query: str,
        executive_summary: str,
        analysis_results: Dict[str, Any],
        citations: List[str],
        retrieval_results: List[Dict[str, Any]]
    ) -> str:
        """Create final markdown response."""
        
        markdown_parts = []
        
        # Title
        markdown_parts.append(f"# Research Analysis: {query}\n")
        
        # Executive Summary
        if executive_summary:
            markdown_parts.append("## Executive Summary\n")
            markdown_parts.append(executive_summary)
            markdown_parts.append("\n")
        
        # Key Findings
        markdown_parts.append("## Key Findings\n")
        
        # Add findings from different analysis types
        if "comparative" in analysis_results:
            comparative = analysis_results["comparative"]
            if "insights" in comparative:
                for insight in comparative["insights"][:3]:
                    markdown_parts.append(f"- {insight.get('description', 'N/A')} [1]\n")
        
        if "trend" in analysis_results:
            trend = analysis_results["trend"]
            if "trends" in trend:
                for metric, trend_data in list(trend["trends"].items())[:2]:
                    direction = trend_data.get("trend_direction", "stable")
                    markdown_parts.append(f"- {metric.title()} shows {direction} trend [2]\n")
        
        markdown_parts.append("\n")
        
        # Supporting Evidence
        markdown_parts.append("## Supporting Evidence\n")
        
        for i, result in enumerate(retrieval_results[:3], 1):
            title = result["document_title"]
            snippet = result["chunk_text"][:200] + "..." if len(result["chunk_text"]) > 200 else result["chunk_text"]
            markdown_parts.append(f"**Source {i}: {title}**\n")
            markdown_parts.append(f"{snippet} [{i}]\n\n")
        
        # References
        if citations:
            markdown_parts.append("## References\n")
            for i, citation in enumerate(citations, 1):
                markdown_parts.append(f"[{i}] {citation}\n")
        
        return "".join(markdown_parts)