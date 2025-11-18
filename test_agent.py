"""
Simplified test version for MacBook M1
Tests core agent functionality without complex dependencies
"""

import asyncio
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key exists
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env file")
    print("Please add your API key to the .env file")
    exit(1)

print("✅ API Key loaded successfully")
print(f"   Key starts with: {api_key[:10]}...")
print()

try:
    import google.generativeai as genai
    print("✅ google-generativeai imported successfully")
except ImportError as e:
    print(f"❌ Error importing google-generativeai: {e}")
    print("Run: pip install google-generativeai")
    exit(1)

# Configure Gemini
genai.configure(api_key=api_key)
print("✅ Gemini configured successfully")
print()

# Test basic functionality
async def test_basic_agent():
    """Test basic agent functionality."""
    print("=" * 60)
    print("Testing Basic Agent Functionality")
    print("=" * 60)
    print()
    
    try:
        # Create a simple agent
        print("📝 Creating test agent...")
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        print("✅ Agent created successfully")
        print()
        
        # Test query
        print("🔍 Testing query: 'What is 2+2?'")
        start_time = time.time()
        
        response = model.generate_content("What is 2+2? Be brief.")
        
        elapsed = time.time() - start_time
        
        print(f"✅ Response received in {elapsed:.2f} seconds")
        print()
        print("Response:")
        print("-" * 60)
        print(response.text)
        print("-" * 60)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

async def test_multi_turn():
    """Test multi-turn conversation."""
    print("=" * 60)
    print("Testing Multi-Turn Conversation")
    print("=" * 60)
    print()
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Start chat
        chat = model.start_chat(history=[])
        
        print("💬 Query 1: 'Tell me about Python in one sentence'")
        response1 = chat.send_message("Tell me about Python in one sentence")
        print(f"   Response: {response1.text}")
        print()
        
        print("💬 Query 2: 'What did I just ask about?'")
        response2 = chat.send_message("What did I just ask about?")
        print(f"   Response: {response2.text}")
        print()
        
        print("✅ Multi-turn conversation works!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during multi-turn test: {e}")
        return False

async def test_parallel_execution():
    """Test parallel execution of multiple queries."""
    print("=" * 60)
    print("Testing Parallel Execution")
    print("=" * 60)
    print()
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        queries = [
            "What is machine learning?",
            "What is cloud computing?",
            "What is data science?"
        ]
        
        print(f"🚀 Executing {len(queries)} queries in parallel...")
        start_time = time.time()
        
        # Execute in parallel
        tasks = [
            model.generate_content_async(f"{q} Answer in one sentence.")
            for q in queries
        ]
        
        responses = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        print(f"✅ All queries completed in {elapsed:.2f} seconds")
        print()
        
        for i, (query, response) in enumerate(zip(queries, responses), 1):
            print(f"Query {i}: {query}")
            print(f"Answer: {response.text}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during parallel test: {e}")
        return False

async def main():
    """Run all tests."""
    print()
    print("🎯 Enterprise Agent - System Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Basic functionality
    result1 = await test_basic_agent()
    results.append(("Basic Agent", result1))
    
    # Test 2: Multi-turn conversation
    result2 = await test_multi_turn()
    results.append(("Multi-Turn", result2))
    
    # Test 3: Parallel execution
    result3 = await test_parallel_execution()
    results.append(("Parallel Execution", result3))
    
    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print()
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print()
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("🎉 All tests passed! Your system is ready.")
        print()
        print("Next steps:")
        print("1. Try running the full agent: python main.py")
        print("2. Check the interactive demo")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print()

if __name__ == "__main__":
    asyncio.run(main())
