"""
Simplified Enterprise Agent for MacBook M1
Core functionality without complex dependencies
"""

import asyncio
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=api_key)

@dataclass
class AgentResult:
    """Result from an agent."""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    confidence: float = 0.85

class SimpleAgent:
    """Base agent class."""
    
    def __init__(self, name: str, instruction: str):
        self.name = name
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            system_instruction=instruction
        )
    
    async def execute(self, query: str) -> AgentResult:
        """Execute agent task."""
        start_time = time.time()
        
        try:
            response = await self.model.generate_content_async(query)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={'response': response.text},
                execution_time=time.time() - start_time,
                confidence=0.90
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data={'error': str(e)},
                execution_time=time.time() - start_time,
                confidence=0.0
            )

class OrchestratorAgent:
    """Main orchestrator that coordinates agents."""
    
    def __init__(self):
        # Create specialized agents
        self.agents = {
            'research': SimpleAgent(
                'ResearchAgent',
                'You are a research specialist. Provide comprehensive, well-researched answers.'
            ),
            'analyst': SimpleAgent(
                'AnalystAgent',
                'You are a data analyst. Focus on insights, patterns, and analysis.'
            ),
            'summarizer': SimpleAgent(
                'SummarizerAgent',
                'You are a summarization expert. Provide concise, clear summaries.'
            )
        }
        
        self.query_count = 0
        
        print("✅ Orchestrator initialized with 3 agents")
        print(f"   Agents: {', '.join(self.agents.keys())}")
        print()
    
    async def process_query(self, query: str, use_parallel: bool = False) -> Dict:
        """Process a query using appropriate agents."""
        self.query_count += 1
        start_time = time.time()
        
        print(f"{'='*60}")
        print(f"Query #{self.query_count}: {query}")
        print(f"{'='*60}")
        print()
        
        # Determine which agents to use
        agents_to_use = self._route_query(query)
        
        print(f"🎯 Routing to agents: {', '.join(agents_to_use)}")
        print()
        
        # Execute agents
        if use_parallel and len(agents_to_use) > 1:
            print("⚡ Executing in PARALLEL...")
            results = await self._execute_parallel(query, agents_to_use)
        else:
            print("📝 Executing SEQUENTIALLY...")
            results = await self._execute_sequential(query, agents_to_use)
        
        # Synthesize results
        synthesis = await self._synthesize(query, results)
        
        total_time = time.time() - start_time
        
        print()
        print(f"✅ Completed in {total_time:.2f}s")
        print()
        print("📊 Results:")
        print("-" * 60)
        
        for result in results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.agent_name}: {result.execution_time:.2f}s")
        
        print()
        print("🎯 Synthesis:")
        print("-" * 60)
        print(synthesis)
        print("-" * 60)
        print()
        
        return {
            'query': query,
            'results': results,
            'synthesis': synthesis,
            'total_time': total_time
        }
    
    def _route_query(self, query: str) -> List[str]:
        """Determine which agents to use."""
        query_lower = query.lower()
        
        if 'analyze' in query_lower or 'data' in query_lower:
            return ['research', 'analyst']
        elif 'summary' in query_lower or 'summarize' in query_lower:
            return ['research', 'summarizer']
        else:
            return ['research']
    
    async def _execute_parallel(self, query: str, agent_names: List[str]) -> List[AgentResult]:
        """Execute agents in parallel."""
        tasks = [
            self.agents[name].execute(query)
            for name in agent_names
        ]
        return await asyncio.gather(*tasks)
    
    async def _execute_sequential(self, query: str, agent_names: List[str]) -> List[AgentResult]:
        """Execute agents sequentially."""
        results = []
        for name in agent_names:
            result = await self.agents[name].execute(query)
            results.append(result)
        return results
    
    async def _synthesize(self, query: str, results: List[AgentResult]) -> str:
        """Synthesize results from multiple agents."""
        if len(results) == 1:
            return results[0].data.get('response', 'No response')
        
        # Combine results
        combined = "\n\n".join([
            f"[{r.agent_name}]: {r.data.get('response', 'Error')}"
            for r in results if r.success
        ])
        
        # Use LLM to synthesize
        synthesizer = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        synthesis_prompt = f"""
        Synthesize these agent responses into a cohesive answer:
        
        Original Query: {query}
        
        Agent Responses:
        {combined}
        
        Provide a clear, comprehensive synthesis.
        """
        
        response = await synthesizer.generate_content_async(synthesis_prompt)
        return response.text

async def demo():
    """Run demonstration queries."""
    print()
    print("🎯 Enterprise Content Intelligence Agent")
    print("=" * 60)
    print()
    
    orchestrator = OrchestratorAgent()
    
    # Example queries
    queries = [
        {
            'query': 'What are the key trends in artificial intelligence for 2024?',
            'parallel': False
        },
        {
            'query': 'Analyze the impact of cloud computing on modern businesses',
            'parallel': True
        },
        {
            'query': 'Summarize the main benefits of using multi-agent systems',
            'parallel': True
        }
    ]
    
    for example in queries:
        await orchestrator.process_query(
            example['query'],
            use_parallel=example['parallel']
        )
        await asyncio.sleep(1)  # Brief pause between queries
    
    print()
    print("=" * 60)
    print("✅ Demo Complete!")
    print(f"   Total queries processed: {orchestrator.query_count}")
    print("=" * 60)
    print()

if __name__ == "__main__":
    asyncio.run(demo())
