"""
Example demonstrating the intelligent URL generation feature.
This shows how the system now uses GPT-4 to generate relevant URLs for web scraping.
"""

import asyncio
import json
from app.mcp_tools import intelligent_url_generator_tool


async def demonstrate_intelligent_url_generation():
    """
    Demonstrate the intelligent URL generation with different types of research queries.
    """

    print("🧠 Intelligent URL Generation Demo")
    print("=" * 60)
    print("This demonstrates how the Deep Research Assistant now uses GPT-4")
    print("to intelligently generate relevant URLs for web scraping instead")
    print("of using hardcoded patterns.")
    print()

    # Example research scenarios
    test_scenarios = [
        {
            "name": "COVID-19 & Renewable Energy Research",
            "query": "How did COVID-19 impact renewable energy investment patterns in developing countries?",
            "subqueries": [
                "COVID-19 economic impacts on developing countries",
                "Renewable energy investment trends 2019-2023",
                "Regional regulatory frameworks for renewable energy",
            ],
        },
        {
            "name": "Climate Policy Analysis",
            "query": "What are the effectiveness of carbon pricing mechanisms in European countries?",
            "subqueries": [
                "Carbon tax implementation in Europe",
                "Emissions trading system effectiveness",
                "Comparative carbon pricing policies",
            ],
        },
        {
            "name": "Technology Innovation Research",
            "query": "How has artificial intelligence adoption affected productivity in manufacturing?",
            "subqueries": [
                "AI implementation in manufacturing",
                "Productivity metrics in automated factories",
                "Industry 4.0 case studies",
            ],
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        print(f"Main Query: {scenario['query']}")
        print(f"Subqueries: {len(scenario['subqueries'])}")
        for j, subquery in enumerate(scenario["subqueries"], 1):
            print(f"  {j}. {subquery}")
        print()

        # Test the intelligent URL generation
        test_input = {
            "query": scenario["query"],
            "subqueries": scenario["subqueries"],
            "max_urls": 6,
            "domains": [],
        }

        try:
            result = await intelligent_url_generator_tool(
                test_input, {"task_id": f"demo-{i}"}
            )

            if result["status"] == "ok":
                urls = result["result"]["generated_urls"]
                method = result["meta"].get("generation_method", "unknown")

                print(f"✅ Generated {len(urls)} URLs using {method}")
                print()

                print("🔗 Intelligent URL Suggestions:")
                for j, url_data in enumerate(urls, 1):
                    print(f"{j}. {url_data['url']}")
                    print(f"   📄 Type: {url_data['content_type']}")
                    print(f"   ⭐ Relevance: {url_data['relevance_score']:.2f}")
                    print(f"   💡 Why: {url_data['description'][:80]}...")
                    if url_data.get("expected_topics"):
                        print(
                            f"   🏷️  Topics: {', '.join(url_data['expected_topics'][:3])}"
                        )
                    print()

                # Show the difference from old hardcoded approach
                print("🔄 Old vs New Approach:")
                print(
                    "   ❌ Old: Hardcoded example.com URLs with simple keyword substitution"
                )
                print(
                    "   ✅ New: LLM-generated authoritative sources tailored to research context"
                )
                print()

            else:
                print(
                    f"❌ URL generation failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            print(f"❌ Demo failed for scenario {i}: {e}")

        print("=" * 60)
        print()

    print("🎯 Key Benefits of Intelligent URL Generation:")
    print("• Contextual relevance: URLs are tailored to the specific research question")
    print(
        "• Authoritative sources: Prioritizes academic, government, and reputable sources"
    )
    print(
        "• Content diversity: Suggests different types of sources (papers, reports, news)"
    )
    print(
        "• Adaptive: Considers both main query and subqueries for comprehensive coverage"
    )
    print(
        "• Fallback system: Gracefully handles LLM failures with pattern-based generation"
    )
    print()
    print("🔧 Integration: This runs automatically in the WebScraperRetrievalAgent")
    print("   when the system determines that web scraping is needed to improve")
    print("   confidence in the research results.")


if __name__ == "__main__":
    # Note: You'll need to set OPENAI_API_KEY environment variable
    print("Note: Set OPENAI_API_KEY environment variable to test with real LLM")
    print("Without it, the system will use the fallback pattern-based generation.")
    print()

    asyncio.run(demonstrate_intelligent_url_generation())
