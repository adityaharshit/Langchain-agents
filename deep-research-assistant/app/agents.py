"""
Individual agent implementations for the multi-agent research system.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import re

from app.mcp import mcp_registry
from app.config import config

logger = logging.getLogger(__name__)


class ResearchCoordinatorAgent:
    """Research Coordinator Agent - orchestrates the research workflow."""

    def __init__(self):
        self.name = "ResearchCoordinatorAgent"
        self.allowed_tools = [
            "query_decomposer",
            "task_prioritizer",
            "progress_tracker",
            "result_synthesis",
            "semantic_search",
        ]

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Research Coordinator Agent."""
        try:
            logger.info(
                f"ResearchCoordinator processing query: {state['query'][:50]}..."
            )

            # Step 1: Decompose query if not already done
            if not state.get("subqueries"):
                decomposition_result = await mcp_registry.execute_tool(
                    "query_decomposer",
                    self.name,
                    {"query": state["query"], "max_subqueries": 5},
                    {"task_id": state["task_id"]},
                )

                if decomposition_result["status"] == "ok":
                    subqueries_data = decomposition_result["result"]["subqueries"]
                    state["subqueries"] = [sq["subquery"] for sq in subqueries_data]

                    # Emit progress event with sub-queries for streaming
                    await mcp_registry.execute_tool(
                        "progress_tracker",
                        self.name,
                        {
                            "event": "progress",
                            "step": "query_decomposition",
                            "payload": {
                                "message": f"Query decomposed into {len(subqueries_data)} sub-queries",
                                "original_query": state["query"],
                                "subqueries": [
                                    {
                                        "subquery": sq["subquery"],
                                        "priority": sq["priority"],
                                        "rationale": sq["rationale"],
                                        "expected_info_type": sq["expected_info_type"],
                                    }
                                    for sq in subqueries_data
                                ],
                                "total_subqueries": len(subqueries_data),
                            },
                            "task_id": state["task_id"],
                        },
                        {"task_id": state["task_id"]},
                    )

                    # Prioritize the subqueries using GPT-4o
                    prioritization_result = await mcp_registry.execute_tool(
                        "task_prioritizer",
                        self.name,
                        {"tasks": subqueries_data, "main_query": state["query"]},
                        {"task_id": state["task_id"]},
                    )

                    if prioritization_result["status"] == "ok":
                        # Update subqueries with prioritized order
                        prioritized_tasks = prioritization_result["result"][
                            "prioritized_tasks"
                        ]
                        state["subqueries"] = [
                            task["task"]["subquery"] for task in prioritized_tasks
                        ]
                else:
                    state["subqueries"] = [state["query"]]

            # Step 2: Perform RAG retrieval for each subquery
            if not state.get("subquery_retrieval_results"):
                state["subquery_retrieval_results"] = {}
                all_results = []
                total_confidence = 0.0

                for subquery in state["subqueries"]:
                    retrieval_result = await mcp_registry.execute_tool(
                        "semantic_search",
                        self.name,
                        {"query": subquery, "k": config.RETRIEVAL_K},
                        {"task_id": state["task_id"]},
                    )

                    if retrieval_result["status"] == "ok":
                        # Transform semantic_search_tool results to match expected structure
                        search_results = retrieval_result["result"]["results"]
                        subquery_results = [
                            {
                                "chunk_id": r.get("metadata", {}).get("chunk_id", 0),
                                "document_id": r.get("metadata", {}).get(
                                    "document_id", 0
                                ),
                                "chunk_text": r["content"],
                                "similarity_score": r["similarity_score"],
                                "document_title": r["title"],
                                "document_url": r["url"],
                                "chunk_meta": r.get("metadata", {}),
                                "token_count": r.get("metadata", {}).get(
                                    "token_count", 0
                                ),
                                "subquery": subquery,  # Track which subquery this came from
                            }
                            for r in search_results
                        ]

                        state["subquery_retrieval_results"][subquery] = subquery_results
                        all_results.extend(subquery_results)
                        total_confidence += retrieval_result["result"][
                            "confidence_score"
                        ]

                        # Emit progress for this subquery
                        await mcp_registry.execute_tool(
                            "progress_tracker",
                            self.name,
                            {
                                "event": "progress",
                                "step": "subquery_retrieval",
                                "payload": {
                                    "message": f"Retrieved {len(subquery_results)} results for subquery",
                                    "subquery": subquery,
                                    "results_count": len(subquery_results),
                                    "confidence": retrieval_result["result"][
                                        "confidence_score"
                                    ],
                                },
                                "task_id": state["task_id"],
                            },
                            {"task_id": state["task_id"]},
                        )
                    else:
                        state["subquery_retrieval_results"][subquery] = []

                # Calculate average confidence across all subqueries
                avg_confidence = (
                    total_confidence / len(state["subqueries"])
                    if state["subqueries"]
                    else 0.0
                )

                # Deduplicate results based on chunk_id and keep highest similarity
                unique_results = {}
                for result in all_results:
                    chunk_id = result["chunk_id"]
                    if (
                        chunk_id not in unique_results
                        or result["similarity_score"]
                        > unique_results[chunk_id]["similarity_score"]
                    ):
                        unique_results[chunk_id] = result

                deduplicated_results = list(unique_results.values())

                # Store aggregated results
                state["retrieval_results"] = {
                    "results": deduplicated_results,
                    "confidence_score": avg_confidence,
                    "total_results": len(deduplicated_results),
                    "subquery_count": len(state["subqueries"]),
                }

                state["confidence_score"] = avg_confidence

            # Add coordinator message
            decomp_method = (
                decomposition_result.get("meta", {}).get(
                    "decomposition_method", "unknown"
                )
                if decomposition_result["status"] == "ok"
                else "failed"
            )
            total_retrieved = sum(
                len(results)
                for results in state.get("subquery_retrieval_results", {}).values()
            )
            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Query intelligently decomposed using {decomp_method} into {len(state['subqueries'])} prioritized subqueries. "
                    f"Retrieved {total_retrieved} results across all subqueries. "
                    f"Average retrieval confidence: {state['confidence_score']:.2f}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

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
            "rag_upsert",
        ]

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Web Scraper Agent."""
        try:
            logger.info(f"WebScraperAgent processing query: {state['query'][:50]}...")

            # Initialize subquery tracking if not exists
            if not state.get("subquery_scraped_documents"):
                state["subquery_scraped_documents"] = {}

            all_scraped_docs = []

            # Process each subquery separately
            for subquery in state.get("subqueries", [state["query"]]):
                # Generate intelligent URLs for this subquery
                url_generation_result = await mcp_registry.execute_tool(
                    "intelligent_url_generator",
                    self.name,
                    {
                        "query": subquery,
                        "subqueries": [subquery],
                        "max_urls": 5,
                        "domains": [],
                    },
                    {"task_id": state["task_id"]},
                )

                # Extract URLs from the generation result
                if url_generation_result["status"] == "ok":
                    generated_url_data = url_generation_result["result"][
                        "generated_urls"
                    ]
                    search_urls = [url_data["url"] for url_data in generated_url_data]

                    logger.info(
                        f"Generated {len(search_urls)} URLs for subquery: {subquery[:50]}"
                    )
                else:
                    logger.warning(
                        f"URL generation failed for subquery: {subquery[:50]}"
                    )
                    search_urls = self._generate_search_urls([subquery])

                # Scrape URLs for this subquery
                scraping_result = await mcp_registry.execute_tool(
                    "web_scraper",
                    self.name,
                    {"urls": search_urls[:3], "max_pages": 3},
                    {"task_id": state["task_id"]},
                )

                if scraping_result["status"] == "ok":
                    scraped_docs = scraping_result["result"]["scraped_documents"]
                    state["subquery_scraped_documents"][subquery] = scraped_docs
                    all_scraped_docs.extend(scraped_docs)

                    # Emit progress for this subquery
                    await mcp_registry.execute_tool(
                        "progress_tracker",
                        self.name,
                        {
                            "event": "progress",
                            "step": "subquery_scraping",
                            "payload": {
                                "message": f"Scraped {len(scraped_docs)} documents for subquery",
                                "subquery": subquery,
                                "documents_count": len(scraped_docs),
                            },
                            "task_id": state["task_id"],
                        },
                        {"task_id": state["task_id"]},
                    )
                else:
                    state["subquery_scraped_documents"][subquery] = []

            state["scraped_documents"] = all_scraped_docs

            # Upsert all scraped documents to RAG
            if all_scraped_docs:
                upsert_result = await mcp_registry.execute_tool(
                    "rag_upsert",
                    self.name,
                    {"documents": all_scraped_docs},
                    {"task_id": state["task_id"]},
                )

                if upsert_result["status"] == "ok":
                    # Re-run retrieval for each subquery with new content
                    state["subquery_retrieval_results"] = state.get(
                        "subquery_retrieval_results", {}
                    )
                    all_results = []
                    total_confidence = 0.0

                    for subquery in state.get("subqueries", [state["query"]]):
                        retrieval_result = await mcp_registry.execute_tool(
                            "semantic_search",
                            self.name,
                            {"query": subquery, "k": config.RETRIEVAL_K},
                            {"task_id": state["task_id"]},
                        )

                        if retrieval_result["status"] == "ok":
                            search_results = retrieval_result["result"]["results"]
                            subquery_results = [
                                {
                                    "chunk_id": r.get("metadata", {}).get(
                                        "chunk_id", 0
                                    ),
                                    "document_id": r.get("metadata", {}).get(
                                        "document_id", 0
                                    ),
                                    "chunk_text": r["content"],
                                    "similarity_score": r["similarity_score"],
                                    "document_title": r["title"],
                                    "document_url": r["url"],
                                    "chunk_meta": r.get("metadata", {}),
                                    "token_count": r.get("metadata", {}).get(
                                        "token_count", 0
                                    ),
                                    "subquery": subquery,
                                }
                                for r in search_results
                            ]

                            state["subquery_retrieval_results"][subquery] = (
                                subquery_results
                            )
                            all_results.extend(subquery_results)
                            total_confidence += retrieval_result["result"][
                                "confidence_score"
                            ]

                    # Calculate average confidence
                    avg_confidence = (
                        total_confidence / len(state["subqueries"])
                        if state["subqueries"]
                        else 0.0
                    )

                    # Deduplicate results
                    unique_results = {}
                    for result in all_results:
                        chunk_id = result["chunk_id"]
                        if (
                            chunk_id not in unique_results
                            or result["similarity_score"]
                            > unique_results[chunk_id]["similarity_score"]
                        ):
                            unique_results[chunk_id] = result

                    deduplicated_results = list(unique_results.values())

                    # Update aggregated results
                    state["retrieval_results"] = {
                        "results": deduplicated_results,
                        "confidence_score": avg_confidence,
                        "total_results": len(deduplicated_results),
                        "method": "post_scraping",
                        "subquery_count": len(state["subqueries"]),
                    }

                    state["confidence_score"] = avg_confidence

            # Add scraper message
            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Processed {len(state['subqueries'])} subqueries. "
                    f"Scraped {len(all_scraped_docs)} total documents. "
                    f"Updated confidence: {state['confidence_score']:.2f}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

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
        self.allowed_tools = ["comprehensive_analysis", "semantic_search"]

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Deep Analysis Agent."""
        try:
            logger.info("DeepAnalysisAgent processing analysis...")

            # Get aggregated retrieval results from all subqueries
            retrieval_results = state.get("retrieval_results", {}).get("results", [])
            subquery_retrieval_results = state.get("subquery_retrieval_results", {})

            if not retrieval_results:
                state["analysis_results"] = {"error": "No retrieval results to analyze"}
                return state

            # Prepare documents for comprehensive analysis, organized by subquery
            context_documents = []
            subquery_context_map = {}

            # Add results from each subquery with context
            for subquery, results in subquery_retrieval_results.items():
                subquery_docs = []
                for result in results:
                    doc = {
                        "chunk_id": result["chunk_id"],
                        "document_title": result["document_title"],
                        "content": result["chunk_text"],
                        "document_url": result["document_url"],
                        "similarity_score": result["similarity_score"],
                        "token_count": result.get("token_count", 0),
                        "subquery": subquery,  # Track which subquery this addresses
                    }
                    subquery_docs.append(doc)
                    context_documents.append(doc)

                subquery_context_map[subquery] = subquery_docs

            # Perform comprehensive analysis using GPT-4o with subquery context
            analysis_result = await mcp_registry.execute_tool(
                "comprehensive_analysis",
                self.name,
                {
                    "query": state["query"],
                    "context_documents": context_documents,
                    "analysis_type": "comprehensive_research",
                    "subqueries": state.get("subqueries", []),
                    "subquery_context_map": {
                        sq: [
                            {"content": doc["content"], "title": doc["document_title"]}
                            for doc in docs
                        ]
                        for sq, docs in subquery_context_map.items()
                    },
                },
                {"task_id": state["task_id"]},
            )

            if analysis_result["status"] == "ok":
                state["analysis_results"] = {
                    "comprehensive": analysis_result["result"],
                    "source_documents": len(context_documents),
                    "subquery_count": len(subquery_retrieval_results),
                    "analysis_confidence": 0.8,  # High confidence for GPT-4o analysis
                }
            else:
                state["analysis_results"] = {
                    "error": analysis_result.get("error", "Analysis failed"),
                    "source_documents": len(context_documents),
                    "subquery_count": len(subquery_retrieval_results),
                }

            # Add analysis message
            analysis_results = state.get("analysis_results", {})
            word_count = analysis_results.get("comprehensive", {}).get("word_count", 0)

            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Generated comprehensive analysis ({word_count} words) using GPT-4o. "
                    f"Analyzed {len(context_documents)} documents across {len(subquery_retrieval_results)} subqueries.",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            return state

        except Exception as e:
            logger.error(f"DeepAnalysisAgent failed: {e}")
            state["error"] = f"Analysis error: {str(e)}"
            return state

    def _extract_data_points(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract data points for trend analysis."""
        data_points = []

        for doc in documents:
            years = re.findall(r"\b(20\d{2})\b", doc["content"])
            numbers = re.findall(
                r"\b(\d+(?:\.\d+)?)\s*(?:billion|million|%)\b", doc["content"]
            )

            for year in years[:2]:
                for number in numbers[:2]:
                    try:
                        data_points.append(
                            {
                                "date": year,
                                "value": float(number),
                                "source": doc["title"],
                            }
                        )
                    except ValueError:
                        continue

        return data_points[:10]

    def _extract_events(
        self, documents: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Extract events for causal analysis."""
        events = []

        for doc in documents:
            event_keywords = ["impact", "effect", "result", "consequence"]

            for keyword in event_keywords:
                if keyword in doc["content"].lower():
                    events.append(
                        {
                            "description": f"Event related to {keyword} in {doc['title']}",
                            "source": doc["title"],
                            "relevance": doc["similarity"],
                        }
                    )

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
            "confidence_scorer",
        ]

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Fact-Checking Agent."""
        try:
            logger.info("FactCheckingAgent processing analysis results...")

            analysis_results = state.get("analysis_results", {})
            retrieval_results = state.get("retrieval_results", {}).get("results", [])

            if not analysis_results or not retrieval_results:
                state["fact_check_results"] = {
                    "error": "No analysis results to fact-check"
                }
                return state

            fact_check_results = {}

            # Check source credibility
            sources = []
            for result in retrieval_results:
                sources.append(
                    {
                        "id": result["chunk_id"],
                        "url": result["document_url"],
                        "title": result["document_title"],
                        "similarity_score": result["similarity_score"],
                    }
                )

            credibility_result = await mcp_registry.execute_tool(
                "source_credibility_checker",
                self.name,
                {"sources": sources, "query_context": state["query"]},
                {"task_id": state["task_id"]},
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
                        "sources": [
                            {"content": r["chunk_text"], **r} for r in retrieval_results
                        ],
                    },
                    {"task_id": state["task_id"]},
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
                    {"task_id": state["task_id"]},
                )

                if contradiction_result["status"] == "ok":
                    fact_check_results["contradictions"] = contradiction_result[
                        "result"
                    ]

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
                            "claims": claims,
                        }
                    ]
                },
                {"task_id": state["task_id"]},
            )

            if confidence_result["status"] == "ok":
                fact_check_results["confidence"] = confidence_result["result"]

                # Update overall confidence score
                confidence_scores = confidence_result["result"]["confidence_scores"]
                if confidence_scores:
                    state["confidence_score"] = confidence_scores[0]["confidence_score"]

            state["fact_check_results"] = fact_check_results

            # Add fact-check message
            contradictions = len(
                fact_check_results.get("contradictions", {}).get("contradictions", [])
            )

            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Fact-checking complete. Final confidence: {state['confidence_score']:.2f}. "
                    f"Found {contradictions} contradictions.",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

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
                claims.append(
                    {
                        "id": f"comparative_{i}",
                        "text": insight.get("description", ""),
                        "type": "comparative",
                    }
                )

        return claims[:5]

    def _extract_statements(
        self, retrieval_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract statements for contradiction detection."""
        statements = []

        for i, result in enumerate(retrieval_results[:3]):
            sentences = result["chunk_text"].split(". ")

            for j, sentence in enumerate(sentences[:2]):
                if len(sentence.strip()) > 20:
                    statements.append(
                        {
                            "id": f"stmt_{i}_{j}",
                            "text": sentence.strip(),
                            "source": result["document_title"],
                        }
                    )

        return statements


class OutputFormattingAgent:
    """Output Formatting Agent - creates final formatted output."""

    def __init__(self):
        self.name = "OutputFormattingAgent"
        self.allowed_tools = [
            "report_structuring",
            "citation_formatter",
            "executive_summary_generator",
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
                sources.append(
                    {
                        "id": result["chunk_id"],
                        "title": result["document_title"],
                        "url": result["document_url"],
                        "authors": [],
                        "publish_date": None,
                    }
                )

            citation_result = await mcp_registry.execute_tool(
                "citation_formatter",
                self.name,
                {"sources": sources},
                {"task_id": state["task_id"]},
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
                        "fact_check": fact_check_results,
                    }
                },
                {"task_id": state["task_id"]},
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
                retrieval_results,
            )

            # Create provenance records
            provenance = []
            for i, result in enumerate(retrieval_results):
                provenance.append(
                    {
                        "claim": f"Information from {result['document_title']}",
                        "chunk_id": result["chunk_id"],
                        "document_title": result["document_title"],
                        "document_url": result["document_url"],
                        "similarity_score": result["similarity_score"],
                        "citation_number": i + 1,
                    }
                )

            state["final_result"] = {
                "answer_markdown": answer_markdown,
                "citations": citations,
                "provenance": provenance,
                "confidence_score": state["confidence_score"],
                "processing_time": 0.0,
                "total_sources": len(retrieval_results),
                "search_method": state.get("retrieval_results", {}).get(
                    "method", "semantic"
                ),
            }

            # Add final message
            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Final report generated with {len(citations)} citations.",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

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
        retrieval_results: List[Dict[str, Any]],
    ) -> str:
        """Create final markdown response."""

        markdown_parts = []

        # Check if we have comprehensive analysis from DeepAnalysisAgent
        comprehensive_analysis = analysis_results.get("comprehensive", {}).get(
            "comprehensive_analysis", ""
        )

        if comprehensive_analysis:
            # Use the comprehensive analysis as the main content
            markdown_parts.append(comprehensive_analysis)
            markdown_parts.append("\n\n")
        else:
            # Fallback to basic structure if no comprehensive analysis
            # Title
            markdown_parts.append(f"# Research Analysis: {query}\n\n")

            # Executive Summary
            if executive_summary:
                markdown_parts.append("## Executive Summary\n\n")
                markdown_parts.append(executive_summary)
                markdown_parts.append("\n\n")

            # Key Findings
            markdown_parts.append("## Key Findings\n\n")

            # Add findings from different analysis types
            if "comparative" in analysis_results:
                comparative = analysis_results["comparative"]
                if "insights" in comparative:
                    for insight in comparative["insights"][:3]:
                        markdown_parts.append(
                            f"- {insight.get('description', 'N/A')} [1]\n"
                        )

            if "trend" in analysis_results:
                trend = analysis_results["trend"]
                if "trends" in trend:
                    for metric, trend_data in list(trend["trends"].items())[:2]:
                        direction = trend_data.get("trend_direction", "stable")
                        markdown_parts.append(
                            f"- {metric.title()} shows {direction} trend [2]\n"
                        )

            markdown_parts.append("\n")

            # Supporting Evidence
            markdown_parts.append("## Supporting Evidence\n\n")

            for i, result in enumerate(retrieval_results[:3], 1):
                title = result["document_title"]
                snippet = (
                    result["chunk_text"][:200] + "..."
                    if len(result["chunk_text"]) > 200
                    else result["chunk_text"]
                )
                markdown_parts.append(f"**Source {i}: {title}**\n\n")
                markdown_parts.append(f"{snippet} [{i}]\n\n")

        # Always add References section at the end
        if citations:
            markdown_parts.append("## References\n\n")
            for i, citation in enumerate(citations, 1):
                markdown_parts.append(f"[{i}] {citation}\n")

        return "".join(markdown_parts)
