"""
LangGraph multi-agent graph definitions and coordinator.
Implements explicit graph construction with agent nodes and tool bindings.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from app.mcp import mcp_registry
from app.workers.vector_store import retrieve_with_confidence
from app.config import config

logger = logging.getLogger(__name__)


# State definition for the graph
class ResearchState(TypedDict):
    """State passed between agents in the research workflow."""

    query: str
    task_id: str
    messages: Annotated[List[Dict[str, Any]], add_messages]
    retrieval_results: Optional[Dict[str, Any]]
    confidence_score: float
    subqueries: List[str]
    scraped_documents: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    fact_check_results: Dict[str, Any]
    final_result: Optional[Dict[str, Any]]
    agent_logs: List[Dict[str, Any]]
    processing_step: str
    error: Optional[str]


class ResearchOrchestrator:
    """
    Main orchestrator for the multi-agent research system.
    Builds and manages the LangGraph workflow.
    """

    def __init__(self):
        self.graph = None
        self.agents = {}
        self._initialize_agents()
        self._build_graph()

    def _initialize_agents(self):
        """Initialize all agent instances."""
        from app.agents import (
            ResearchCoordinatorAgent,
            WebScraperRetrievalAgent,
            DeepAnalysisAgent,
            FactCheckingAgent,
            OutputFormattingAgent,
        )

        self.agents = {
            "research_coordinator": ResearchCoordinatorAgent(),
            "web_scraper": WebScraperRetrievalAgent(),
            "deep_analysis": DeepAnalysisAgent(),
            "fact_checking": FactCheckingAgent(),
            "output_formatting": OutputFormattingAgent(),
        }

    def _build_graph(self):
        """Build the explicit LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(ResearchState)

        # Add agent nodes
        workflow.add_node("research_coordinator", self.agents["research_coordinator"])
        workflow.add_node("web_scraper", self.agents["web_scraper"])
        workflow.add_node("deep_analysis", self.agents["deep_analysis"])
        workflow.add_node("fact_checking", self.agents["fact_checking"])
        workflow.add_node("output_formatting", self.agents["output_formatting"])

        # Set entry point
        workflow.set_entry_point("research_coordinator")

        # Add conditional edges based on the specified structure
        workflow.add_conditional_edges(
            "research_coordinator",
            self._route_from_coordinator,
            {
                "low_retrieval_confidence": "web_scraper",
                "sufficient_retrieval_confidence": "deep_analysis",
            },
        )

        # Web scraper always goes to deep analysis
        workflow.add_edge("web_scraper", "deep_analysis")

        # Deep analysis always goes to fact checking
        workflow.add_edge("deep_analysis", "fact_checking")

        # Conditional routing from fact checking
        workflow.add_conditional_edges(
            "fact_checking",
            self._route_from_factcheck,
            {
                "low_fact_confidence": "research_coordinator",
                "verified_high_confidence": "output_formatting",
            },
        )

        # Output formatting ends the workflow
        workflow.add_edge("output_formatting", END)

        # Compile the graph
        self.graph = workflow.compile()

        logger.info("LangGraph workflow compiled successfully")

    def _route_from_coordinator(self, state: ResearchState) -> str:
        """Route from research coordinator based on retrieval confidence."""
        confidence = state.get("confidence_score", 0.0)

        if confidence >= config.CONFIDENCE_THRESHOLD:
            logger.info(f"High confidence ({confidence:.2f}) - proceeding to analysis")
            return "sufficient_retrieval_confidence"
        else:
            logger.info(f"Low confidence ({confidence:.2f}) - triggering web scraping")
            return "low_retrieval_confidence"

    def _route_from_factcheck(self, state: ResearchState) -> str:
        """Route from fact checking based on final confidence."""
        confidence = state.get("confidence_score", 0.0)

        if confidence >= 0.7:  # High threshold for final output
            logger.info(
                f"High fact-check confidence ({confidence:.2f}) - proceeding to output"
            )
            return "verified_high_confidence"
        else:
            logger.info(
                f"Low fact-check confidence ({confidence:.2f}) - returning to coordinator"
            )
            return "low_fact_confidence"

    async def process_query(self, query: str, task_id: str) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent workflow.

        Args:
            query: Research query to process
            task_id: Unique task identifier

        Returns:
            Final research result
        """
        try:
            # Initialize state
            initial_state = ResearchState(
                query=query,
                task_id=task_id,
                messages=[],
                retrieval_results=None,
                confidence_score=0.0,
                subqueries=[],
                scraped_documents=[],
                analysis_results={},
                fact_check_results={},
                final_result=None,
                agent_logs=[],
                processing_step="start",
                error=None,
            )

            logger.info(f"Starting research workflow for query: {query[:50]}...")

            # Execute the workflow
            final_state = await self.graph.ainvoke(initial_state)

            if final_state.get("error"):
                logger.error(f"Workflow failed: {final_state['error']}")
                return {
                    "success": False,
                    "error": final_state["error"],
                    "task_id": task_id,
                }

            # Extract final result
            final_result = final_state.get("final_result")

            if not final_result:
                logger.error("No final result generated")
                return {
                    "success": False,
                    "error": "No final result generated",
                    "task_id": task_id,
                }

            logger.info(f"Research workflow completed successfully for task {task_id}")

            return {
                "success": True,
                "result": final_result,
                "task_id": task_id,
                "agent_logs": final_state.get("agent_logs", []),
                "processing_steps": len(final_state.get("messages", [])),
            }

        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            return {
                "success": False,
                "error": f"Workflow error: {str(e)}",
                "task_id": task_id,
            }

    def get_graph_structure(self) -> Dict[str, Any]:
        """Get the graph structure for inspection."""
        return {
            "nodes": list(self.agents.keys()),
            "edges": [
                ["research_coordinator", "web_scraper", "low_retrieval_confidence"],
                [
                    "research_coordinator",
                    "deep_analysis",
                    "sufficient_retrieval_confidence",
                ],
                ["web_scraper", "deep_analysis", "always"],
                ["deep_analysis", "fact_checking", "always"],
                ["fact_checking", "research_coordinator", "low_fact_confidence"],
                ["fact_checking", "output_formatting", "verified_high_confidence"],
            ],
            "entry_point": "research_coordinator",
            "agent_tools": {
                agent_name: agent.allowed_tools
                for agent_name, agent in self.agents.items()
            },
        }

    def visualize_graph(self) -> str:
        """Generate a text visualization of the graph."""
        structure = self.get_graph_structure()

        viz_lines = ["Research Assistant Multi-Agent Graph:", ""]
        viz_lines.append("Nodes:")
        for node in structure["nodes"]:
            tools = ", ".join(structure["agent_tools"][node])
            viz_lines.append(f"  - {node}: [{tools}]")

        viz_lines.append("\nEdges:")
        for edge in structure["edges"]:
            source, target, condition = edge
            viz_lines.append(f"  - {source} → {target} ({condition})")

        return "\n".join(viz_lines)


# Global orchestrator instance
research_orchestrator: Optional[ResearchOrchestrator] = None


async def get_research_orchestrator() -> ResearchOrchestrator:
    """Get or create the global research orchestrator."""
    global research_orchestrator

    if research_orchestrator is None:
        research_orchestrator = ResearchOrchestrator()

    return research_orchestrator


async def process_research_query(query: str, task_id: str) -> Dict[str, Any]:
    """Process a research query using the global orchestrator."""
    orchestrator = await get_research_orchestrator()
    return await orchestrator.process_query(query, task_id)
