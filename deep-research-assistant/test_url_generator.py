"""
Simple test script for the intelligent URL generator.
"""
import asyncio
import os
from app.mcp_tools import intelligent_url_generator_tool

async def test_url_generator():
    """Test the intelligent URL generator with a sample query."""
    
    # Set up test environment
    os.environ["OPENAI_API_KEY"] = "test-key"  # You'll need to set your actual key
    
    # Test input
    test_input = {
        "query": "How did COVID-19 impact renewable energy investment patterns in developing countries?",
        "subqueries": [
            "COVID-19 economic impacts on developing countries",
            "Renewable energy investment trends 2019-2023",
            "Regional regulatory frameworks for renewable energy"
        ],
        "max_urls": 5,
        "domains": []
    }
    
    test_context = {"task_id": "test-123"}
    
    print("🧪 Testing Intelligent URL Generator")
    print("=" * 50)
    print(f"Query: {test_input['query']}")
    print(f"Subqueries: {len(test_input['subqueries'])}")
    print()
    
    try:
        # Call the tool
        result = await intelligent_url_generator_tool(test_input, test_context)
        
        print("📊 Results:")
        print(f"Status: {result['status']}")
        
        if result["status"] == "ok":
            generated_urls = result["result"]["generated_urls"]
            generation_method = result["meta"].get("generation_method", "unknown")
            
            print(f"Generation Method: {generation_method}")
            print(f"Total URLs Generated: {len(generated_urls)}")
            print()
            
            print("🔗 Generated URLs:")
            for i, url_data in enumerate(generated_urls, 1):
                print(f"{i}. {url_data['url']}")
                print(f"   Type: {url_data['content_type']}")
                print(f"   Relevance: {url_data['relevance_score']:.2f}")
                print(f"   Description: {url_data['description'][:100]}...")
                print(f"   Method: {url_data['generation_method']}")
                print()
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_url_generator())