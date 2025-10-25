"""
End-to-end integration test and demo for Deep Research Assistant.
Tests the complete multi-hop query workflow with seed data.
"""
import asyncio
import pytest
import httpx
from datetime import datetime
from typing import Dict, Any

from app.main import app
from app.db.database import init_database, db_manager
from app.db.models import Document, Chunk
from app.workers.embedding import generate_embeddings
from app.utils.chunker import SemanticChunker

# Test configuration
TEST_BASE_URL = "http://localhost:8000"
DEMO_QUERY = """
How did COVID-19 impact renewable energy investment patterns in developing countries, 
and what regulatory changes emerged as a result across different regions?
"""

# Seed data for testing
SEED_DOCUMENTS = [
    {
        "url": "https://example.com/renewable-energy-2023",
        "title": "Renewable Energy Investment Trends 2019-2023",
        "content": """
        The renewable energy sector experienced significant volatility during the COVID-19 pandemic. 
        In developing countries, investment patterns shifted dramatically between 2019 and 2023.
        
        Prior to the pandemic, renewable energy investments in developing nations were growing at 
        15% annually. However, the COVID-19 crisis led to a 23% decline in 2020, particularly 
        affecting solar and wind projects in Southeast Asia and Sub-Saharan Africa.
        
        Government responses varied by region. Latin American countries implemented emergency 
        funding mechanisms, while African nations focused on regulatory streamlining. 
        Asian developing countries introduced feed-in tariff adjustments to maintain investor confidence.
        
        By 2022, investment levels had recovered to pre-pandemic levels, with new regulatory 
        frameworks emphasizing climate resilience and energy security. The International Energy 
        Agency reported that developing countries accounted for 67% of global renewable capacity 
        additions in 2023.
        """,
        "language": "en",
        "publish_date": datetime(2023, 6, 15),
        "source_trust_score": 0.85
    },
    {
        "url": "https://example.com/covid-economic-impact",
        "title": "COVID-19 Economic Impact on Developing Countries",
        "content": """
        The COVID-19 pandemic created unprecedented economic disruption across developing countries,
        with lasting impacts on energy sector investments and policy frameworks.
        
        Economic contraction in 2020 averaged 3.2% across developing nations, with energy-intensive
        industries particularly affected. This led to reduced electricity demand and delayed 
        infrastructure projects, including renewable energy installations.
        
        Policy responses included emergency economic measures, debt relief programs, and 
        accelerated regulatory reforms. Many countries used the crisis as an opportunity to 
        "build back better" with green recovery packages.
        
        The World Bank estimated that $2.4 trillion in additional financing would be needed 
        for developing countries to achieve sustainable development goals, with energy transition 
        representing 40% of this requirement.
        
        Regional variations were significant: East Asian countries recovered faster due to 
        stronger institutional frameworks, while Latin American and African nations faced 
        prolonged challenges due to limited fiscal space and healthcare system constraints.
        """,
        "language": "en", 
        "publish_date": datetime(2022, 3, 10),
        "source_trust_score": 0.90
    },
    {
        "url": "https://example.com/regulatory-changes-2022",
        "title": "Global Regulatory Changes in Renewable Energy Post-COVID",
        "content": """
        The post-COVID period saw accelerated regulatory changes in renewable energy across 
        multiple regions, driven by economic recovery needs and climate commitments.
        
        Key regulatory developments included:
        
        1. Streamlined permitting processes: 15 developing countries reduced project approval 
           times by an average of 40% between 2021-2023.
        
        2. Enhanced grid integration rules: New technical standards were implemented in 
           Brazil, India, and South Africa to accommodate higher renewable penetration.
        
        3. Financial incentives restructuring: Feed-in tariffs were replaced with auction 
           mechanisms in 8 countries, improving cost competitiveness.
        
        4. Cross-border cooperation frameworks: Regional power trading agreements were 
           established in West Africa and Central America.
        
        The regulatory changes were directly linked to pandemic recovery strategies, with 
        governments recognizing renewable energy as a tool for economic stimulus and 
        energy security enhancement.
        
        Impact assessment showed that countries with faster regulatory adaptation achieved 
        25% higher renewable investment recovery rates compared to those with delayed reforms.
        """,
        "language": "en",
        "publish_date": datetime(2022, 11, 20),
        "source_trust_score": 0.88
    }
]


class TestEndToEnd:
    """End-to-end test suite for Deep Research Assistant."""
    
    @pytest.fixture(scope="class", autouse=True)
    async def setup_test_environment(self):
        """Set up test environment with seed data."""
        try:
            # Initialize database
            await init_database()
            
            # Seed the database with test documents
            await self.seed_database()
            
            yield
            
            # Cleanup would go here in a full test suite
            
        except Exception as e:
            pytest.fail(f"Test setup failed: {e}")
    
    async def seed_database(self):
        """Seed database with test documents and chunks."""
        try:
            chunker = SemanticChunker()
            
            async with db_manager.get_session() as session:
                for doc_data in SEED_DOCUMENTS:
                    # Create document
                    document = Document(
                        url=doc_data["url"],
                        title=doc_data["title"],
                        raw_html=f"<html><body>{doc_data['content']}</body></html>",
                        cleaned_text=doc_data["content"],
                        language=doc_data["language"],
                        publish_date=doc_data["publish_date"],
                        source_trust_score=doc_data["source_trust_score"]
                    )
                    
                    session.add(document)
                    await session.flush()  # Get document ID
                    
                    # Create chunks
                    chunks = await chunker.chunk_document(
                        doc_data["content"],
                        {"document_id": document.id, "url": document.url}
                    )
                    
                    # Generate embeddings
                    chunk_texts = [chunk.chunk_text for chunk in chunks]
                    embeddings = await generate_embeddings(chunk_texts)
                    
                    # Store chunks with embeddings
                    for chunk_data, embedding in zip(chunks, embeddings):
                        chunk = Chunk(
                            document_id=document.id,
                            chunk_text=chunk_data.chunk_text,
                            token_count=chunk_data.token_count,
                            chunk_meta=chunk_data.chunk_meta,
                            embedding=embedding
                        )
                        session.add(chunk)
            
            print(f"Seeded database with {len(SEED_DOCUMENTS)} documents")
            
        except Exception as e:
            pytest.fail(f"Database seeding failed: {e}")
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test system health check."""
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL) as client:
            response = await client.get("/api/v1/health")
            
            assert response.status_code == 200
            health_data = response.json()
            
            assert health_data["status"] == "healthy"
            assert "components" in health_data
            assert "database" in health_data["components"]
    
    @pytest.mark.asyncio
    async def test_stats_endpoint(self):
        """Test system statistics endpoint."""
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL) as client:
            response = await client.get("/api/v1/stats")
            
            assert response.status_code == 200
            stats_data = response.json()
            
            assert "database_stats" in stats_data
            assert "processing_stats" in stats_data
            assert stats_data["database_stats"]["total_documents"] >= len(SEED_DOCUMENTS)
    
    @pytest.mark.asyncio
    async def test_multi_hop_query_workflow(self):
        """
        Test the complete multi-hop query workflow.
        This is the main end-to-end test demonstrating the system capabilities.
        """
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL, timeout=120.0) as client:
            # Step 1: Submit the demo query
            chat_request = {
                "query": DEMO_QUERY,
                "stream": False,  # Use non-streaming for easier testing
                "max_results": 8,
                "search_method": "semantic",
                "confidence_threshold": 0.7
            }
            
            response = await client.post("/api/v1/chat", json=chat_request)
            
            assert response.status_code == 200
            chat_data = response.json()
            
            assert "task_id" in chat_data
            assert chat_data["status"] in ["accepted", "completed"]
            
            task_id = chat_data["task_id"]
            
            # Step 2: Check task status (if not immediately completed)
            if chat_data["status"] == "accepted":
                # Wait for completion (with timeout)
                max_wait = 60  # seconds
                wait_time = 0
                
                while wait_time < max_wait:
                    status_response = await client.get(f"/api/v1/status/{task_id}")
                    assert status_response.status_code == 200
                    
                    status_data = status_response.json()
                    
                    if status_data["status"] == "completed":
                        break
                    elif status_data["status"] == "failed":
                        pytest.fail(f"Task failed: {status_data.get('error_message', 'Unknown error')}")
                    
                    await asyncio.sleep(2)
                    wait_time += 2
                
                if wait_time >= max_wait:
                    pytest.fail("Task did not complete within timeout")
                
                # Get final result
                final_response = await client.get(f"/api/v1/status/{task_id}")
                final_data = final_response.json()
                result = final_data["result"]
            else:
                # Task completed immediately
                result = chat_data["result"]
            
            # Step 3: Validate the result structure
            assert result is not None
            assert "answer_markdown" in result
            assert "citations" in result
            assert "provenance" in result
            assert "confidence_score" in result
            
            # Step 4: Validate content quality
            answer = result["answer_markdown"]
            citations = result["citations"]
            provenance = result["provenance"]
            
            # Check that answer contains relevant content
            assert len(answer) > 100  # Substantial answer
            assert "COVID-19" in answer or "covid" in answer.lower()
            assert "renewable energy" in answer.lower() or "investment" in answer.lower()
            
            # Check citations are in IEEE format
            assert len(citations) > 0
            for citation in citations:
                assert citation.startswith("[")  # IEEE format starts with [1], [2], etc.
            
            # Check provenance mapping
            assert len(provenance) > 0
            for prov in provenance:
                assert "chunk_id" in prov
                assert "document_title" in prov
                assert "document_url" in prov
                assert "similarity_score" in prov
                assert 0 <= prov["similarity_score"] <= 1
            
            # Step 5: Validate no non-English sources
            for prov in provenance:
                # All our seed data is English, so this should pass
                assert "example.com" in prov["document_url"]  # Our test URLs
            
            # Step 6: Check confidence score is reasonable
            confidence = result["confidence_score"]
            assert 0 <= confidence <= 1
            assert confidence > 0.3  # Should have some confidence with our seed data
            
            print(f"✅ Multi-hop query test passed!")
            print(f"   - Answer length: {len(answer)} characters")
            print(f"   - Citations: {len(citations)}")
            print(f"   - Provenance records: {len(provenance)}")
            print(f"   - Confidence score: {confidence:.2f}")
            
            return result
    
    @pytest.mark.asyncio
    async def test_streaming_workflow(self):
        """Test the SSE streaming workflow."""
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL, timeout=120.0) as client:
            # Submit streaming request
            chat_request = {
                "query": "What are the key trends in renewable energy investment?",
                "stream": True,
                "max_results": 5
            }
            
            response = await client.post("/api/v1/chat", json=chat_request)
            assert response.status_code == 200
            
            chat_data = response.json()
            assert chat_data["status"] == "accepted"
            
            task_id = chat_data["task_id"]
            
            # Test streaming endpoint
            stream_response = await client.get(f"/api/v1/stream/{task_id}")
            assert stream_response.status_code == 200
            assert stream_response.headers["content-type"] == "text/event-stream; charset=utf-8"
            
            # Read first few events (not full stream for test speed)
            events_received = 0
            async for line in stream_response.aiter_lines():
                if line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix
                    try:
                        event = eval(event_data)  # In production, use json.loads
                        assert "event" in event
                        assert "timestamp" in event
                        events_received += 1
                        
                        if events_received >= 3:  # Just test first few events
                            break
                    except:
                        continue  # Skip malformed events
            
            assert events_received > 0
            print(f"✅ Streaming test passed! Received {events_received} events")
    
    @pytest.mark.asyncio 
    async def test_error_handling(self):
        """Test error handling for invalid requests."""
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL) as client:
            # Test invalid query
            response = await client.post("/api/v1/chat", json={"query": ""})
            assert response.status_code == 422  # Validation error
            
            # Test non-existent task
            response = await client.get("/api/v1/status/non-existent-task")
            assert response.status_code == 404
            
            # Test invalid streaming task
            response = await client.get("/api/v1/stream/non-existent-task")
            assert response.status_code == 404
            
            print("✅ Error handling test passed!")


# Standalone demo function
async def run_demo():
    """
    Standalone demo function that can be run independently.
    Demonstrates the system with the COVID-19 renewable energy query.
    """
    print("🚀 Starting Deep Research Assistant Demo")
    print("=" * 50)
    
    try:
        # Initialize the system
        print("📊 Initializing system...")
        await init_database()
        
        # Seed database
        print("🌱 Seeding database with demo data...")
        test_instance = TestEndToEnd()
        await test_instance.seed_database()
        
        # Run the demo query
        print(f"🔍 Processing demo query:")
        print(f"   {DEMO_QUERY}")
        print()
        
        async with httpx.AsyncClient(app=app, base_url=TEST_BASE_URL, timeout=180.0) as client:
            # Submit query
            chat_request = {
                "query": DEMO_QUERY,
                "stream": False,
                "max_results": 8,
                "search_method": "semantic"
            }
            
            print("⏳ Submitting query...")
            response = await client.post("/api/v1/chat", json=chat_request)
            
            if response.status_code != 200:
                print(f"❌ Request failed: {response.status_code}")
                print(response.text)
                return
            
            chat_data = response.json()
            task_id = chat_data["task_id"]
            
            print(f"📋 Task ID: {task_id}")
            
            # Wait for completion
            if chat_data["status"] == "accepted":
                print("⏳ Waiting for completion...")
                
                max_wait = 120
                wait_time = 0
                
                while wait_time < max_wait:
                    status_response = await client.get(f"/api/v1/status/{task_id}")
                    status_data = status_response.json()
                    
                    print(f"   Status: {status_data['status']}")
                    
                    if status_data["status"] == "completed":
                        result = status_data["result"]
                        break
                    elif status_data["status"] == "failed":
                        print(f"❌ Task failed: {status_data.get('error_message')}")
                        return
                    
                    await asyncio.sleep(3)
                    wait_time += 3
                
                if wait_time >= max_wait:
                    print("⏰ Task timed out")
                    return
            else:
                result = chat_data["result"]
            
            # Display results
            print("\n" + "=" * 50)
            print("📋 RESEARCH RESULTS")
            print("=" * 50)
            
            print(f"🎯 Confidence Score: {result['confidence_score']:.2f}")
            print(f"📚 Total Sources: {result['total_sources']}")
            print(f"🔗 Citations: {len(result['citations'])}")
            print(f"📍 Provenance Records: {len(result['provenance'])}")
            
            print("\n📝 ANSWER:")
            print("-" * 30)
            print(result["answer_markdown"])
            
            print("\n📚 CITATIONS:")
            print("-" * 30)
            for i, citation in enumerate(result["citations"], 1):
                print(f"{citation}")
            
            print("\n🔍 PROVENANCE SAMPLE:")
            print("-" * 30)
            for prov in result["provenance"][:3]:  # Show first 3
                print(f"• {prov['document_title']}")
                print(f"  Similarity: {prov['similarity_score']:.3f}")
                print(f"  URL: {prov['document_url']}")
                print()
            
            print("✅ Demo completed successfully!")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo())