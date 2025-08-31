"""
Multi-Agent System using LangGraph
This example demonstrates a research and analysis workflow with multiple specialized agents.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from datetime import datetime
import operator
from dataclasses import dataclass

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # You can replace with any LLM
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

# For this example, we'll use a mock LLM for demonstration
class MockLLM:
    """Mock LLM for demonstration purposes"""
    def __init__(self, name: str):
        self.name = name
    
    async def ainvoke(self, messages):
        # Simulate different agent responses based on the last message
        if isinstance(messages, list) and messages:
            content = messages[-1].content.lower()
            
            if "research" in content:
                return AIMessage(content=f"[{self.name}] Research findings: Based on the query, I found several key points about the topic. The main insights include market trends, technological developments, and future predictions.")
            elif "analyze" in content:
                return AIMessage(content=f"[{self.name}] Analysis complete: The data shows positive trends with 15% growth, key challenges in scalability, and opportunities in emerging markets.")
            elif "write" in content or "report" in content:
                return AIMessage(content=f"[{self.name}] Report generated: Executive Summary: Our analysis reveals significant opportunities with manageable risks. Recommendations include strategic partnerships and technology investments.")
            elif "coordinate" in content:
                return AIMessage(content=f"[{self.name}] Coordination: Task assigned to appropriate specialist agents. Monitoring progress and ensuring quality standards.")
            else:
                return AIMessage(content=f"[{self.name}] Task processed successfully with relevant insights and recommendations.")
        
        return AIMessage(content=f"[{self.name}] Ready to assist with the requested task.")

# Define the shared state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_task: str
    research_results: Dict[str, Any]
    analysis_results: Dict[str, Any]
    report_draft: str
    agent_responses: Dict[str, str]
    task_history: List[Dict[str, Any]]
    next_agent: Optional[str]

# Tools for agents to use
@tool
def web_search_tool(query: str) -> str:
    """Simulate web search functionality"""
    mock_results = [
        f"Result 1: {query} shows promising developments in Q4 2024",
        f"Result 2: Industry experts predict 20% growth in {query} sector",
        f"Result 3: Recent studies indicate {query} adoption increasing globally"
    ]
    return json.dumps({"query": query, "results": mock_results, "timestamp": datetime.now().isoformat()})

@tool
def data_analysis_tool(data: str) -> str:
    """Simulate data analysis functionality"""
    return json.dumps({
        "analysis": f"Statistical analysis of {data}",
        "key_metrics": {"growth_rate": 15.5, "confidence": 0.85, "trend": "positive"},
        "insights": ["Strong upward trend", "Seasonal variations observed", "Market expansion potential"],
        "timestamp": datetime.now().isoformat()
    })

@tool
def document_generator_tool(content: str, doc_type: str) -> str:
    """Generate documents based on content and type"""
    return json.dumps({
        "document_type": doc_type,
        "content_length": len(content),
        "sections": ["Executive Summary", "Key Findings", "Recommendations", "Conclusion"],
        "generated_at": datetime.now().isoformat(),
        "status": "completed"
    })

# Create tool executor
tools = [web_search_tool, data_analysis_tool, document_generator_tool]
tool_executor = ToolExecutor(tools)

class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "coordinator": CoordinatorAgent(),
            "researcher": ResearchAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "reviewer": ReviewerAgent()
        }
        
        # Create the state graph
        self.workflow = StateGraph(AgentState)
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Setup the workflow graph with nodes and edges"""
        
        # Add nodes for each agent
        self.workflow.add_node("coordinator", self._coordinator_node)
        self.workflow.add_node("researcher", self._researcher_node)
        self.workflow.add_node("analyst", self._analyst_node)
        self.workflow.add_node("writer", self._writer_node)
        self.workflow.add_node("reviewer", self._reviewer_node)
        
        # Set entry point
        self.workflow.set_entry_point("coordinator")
        
        # Add conditional edges based on coordinator decisions
        self.workflow.add_conditional_edges(
            "coordinator",
            self._route_next_agent,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "reviewer": "reviewer",
                "end": END
            }
        )
        
        # Add edges from each agent back to coordinator
        for agent in ["researcher", "analyst", "writer", "reviewer"]:
            self.workflow.add_edge(agent, "coordinator")
        
        # Compile the workflow
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    async def _coordinator_node(self, state: AgentState) -> AgentState:
        """Coordinator agent node"""
        result = await self.agents["coordinator"].process(state)
        return result
    
    async def _researcher_node(self, state: AgentState) -> AgentState:
        """Research agent node"""
        result = await self.agents["researcher"].process(state)
        return result
    
    async def _analyst_node(self, state: AgentState) -> AgentState:
        """Analysis agent node"""
        result = await self.agents["analyst"].process(state)
        return result
    
    async def _writer_node(self, state: AgentState) -> AgentState:
        """Writer agent node"""
        result = await self.agents["writer"].process(state)
        return result
    
    async def _reviewer_node(self, state: AgentState) -> AgentState:
        """Reviewer agent node"""
        result = await self.agents["reviewer"].process(state)
        return result
    
    def _route_next_agent(self, state: AgentState) -> str:
        """Determine the next agent based on current state"""
        if state.get("next_agent"):
            next_agent = state["next_agent"]
            state["next_agent"] = None  # Reset for next iteration
            return next_agent
        return "end"

# Base Agent Class
class BaseAgent:
    def __init__(self, name: str, role: str, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.llm = MockLLM(name)
        self.tools = tools
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state"""
        # Add processing logic here
        current_messages = state.get("messages", [])
        
        # Create a prompt based on the agent's role
        prompt = self._create_prompt(state)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Update state with agent's response
        state["messages"].append(response)
        state["agent_responses"][self.name] = response.content
        
        # Add task to history
        task_entry = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "task": state.get("current_task", ""),
            "response": response.content
        }
        state["task_history"].append(task_entry)
        
        return await self._specialized_processing(state)
    
    def _create_prompt(self, state: AgentState) -> str:
        """Create a prompt based on the agent's role and current state"""
        base_prompt = f"You are a {self.role} agent. Your capabilities include: {', '.join(self.capabilities)}."
        current_task = state.get("current_task", "No specific task")
        return f"{base_prompt}\n\nCurrent task: {current_task}\n\nPlease provide your specialized input."
    
    async def _specialized_processing(self, state: AgentState) -> AgentState:
        """Override this method in specialized agents"""
        return state

# Specialized Agent Implementations
class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Coordinator",
            role="Project Coordinator",
            capabilities=["task_planning", "workflow_management", "quality_control"]
        )
        self.workflow_steps = ["research", "analysis", "writing", "review"]
        self.current_step = 0
    
    async def _specialized_processing(self, state: AgentState) -> AgentState:
        """Coordinate the workflow and decide next agent"""
        current_task = state.get("current_task", "")
        
        # Determine next agent based on workflow progress
        if "research" in current_task.lower() and not state.get("research_results"):
            state["next_agent"] = "researcher"
        elif "analysis" in current_task.lower() and not state.get("analysis_results"):
            state["next_agent"] = "analyst"
        elif "report" in current_task.lower() or "write" in current_task.lower():
            if state.get("research_results") and state.get("analysis_results"):
                state["next_agent"] = "writer"
        elif "review" in current_task.lower() and state.get("report_draft"):
            state["next_agent"] = "reviewer"
        else:
            # Check workflow completion
            if (state.get("research_results") and 
                state.get("analysis_results") and 
                state.get("report_draft")):
                state["next_agent"] = "reviewer"
            elif state.get("research_results") and state.get("analysis_results"):
                state["next_agent"] = "writer"
            elif state.get("research_results"):
                state["next_agent"] = "analyst"
            else:
                state["next_agent"] = "researcher"
        
        return state

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Researcher",
            role="Research Specialist",
            capabilities=["web_search", "data_gathering", "source_validation"]
        )
    
    async def _specialized_processing(self, state: AgentState) -> AgentState:
        """Perform research tasks"""
        current_task = state.get("current_task", "")
        
        # Simulate research using tools
        search_results = web_search_tool.invoke(current_task)
        
        research_data = {
            "topic": current_task,
            "search_results": json.loads(search_results),
            "key_findings": [
                "Market growth trends identified",
                "Competitive landscape mapped",
                "Technology adoption patterns analyzed"
            ],
            "sources": ["Industry Report 2024", "Market Research Database", "Expert Interviews"],
            "confidence_level": 0.85,
            "completed_at": datetime.now().isoformat()
        }
        
        state["research_results"] = research_data
        return state

class AnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Analyst",
            role="Data Analyst",
            capabilities=["statistical_analysis", "trend_identification", "forecasting"]
        )
    
    async def _specialized_processing(self, state: AgentState) -> AgentState:
        """Perform analysis tasks"""
        research_data = state.get("research_results", {})
        
        if research_data:
            # Simulate analysis using tools
            analysis_results = data_analysis_tool.invoke(str(research_data))
            
            analysis_data = {
                "analysis_type": "comprehensive_market_analysis",
                "raw_data": research_data,
                "statistical_results": json.loads(analysis_results),
                "trends": [
                    {"trend": "upward", "strength": 0.8, "period": "12_months"},
                    {"trend": "seasonal_variation", "strength": 0.6, "period": "quarterly"}
                ],
                "predictions": {
                    "short_term": "15% growth expected in next quarter",
                    "long_term": "Sustained growth over 24 months"
                },
                "risk_assessment": {"level": "moderate", "factors": ["market_volatility", "regulatory_changes"]},
                "completed_at": datetime.now().isoformat()
            }
            
            state["analysis_results"] = analysis_data
        
        return state

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Writer",
            role="Technical Writer",
            capabilities=["report_writing", "documentation", "content_creation"]
        )
    
    async def _specialized_processing(self, state: AgentState) -> AgentState:
        """Generate reports and documents"""
        research_data = state.get("research_results", {})
        analysis_data = state.get("analysis_results", {})
        
        if research_data and analysis_data:
            # Generate document using tool
            content_summary = f"Research: {len(research_data.get('key_findings', []))} findings, Analysis: {analysis_data.get('analysis_type', 'standard')}"
            doc_result = document_generator_tool.invoke(content_summary, "comprehensive_report")
            
            report_content = f"""
# Comprehensive Analysis Report

## Executive Summary
Based on our research and analysis, we have identified significant opportunities in the target market with manageable risk factors.

## Research Findings
- {len(research_data.get('key_findings', []))} key findings identified
- High confidence level: {research_data.get('confidence_level', 0)}
- Multiple validated sources consulted

## Analysis Results
- Statistical confidence: {analysis_data.get('statistical_results', {}).get('key_metrics', {}).get('confidence', 'N/A')}
- Growth rate: {analysis_data.get('statistical_results', {}).get('key_metrics', {}).get('growth_rate', 'N/A')}%
- Trend: {analysis_data.get('statistical_results', {}).get('key_metrics', {}).get('trend', 'stable')}

## Recommendations
1. Proceed with strategic implementation
2. Monitor key performance indicators
3. Prepare for scaling opportunities

## Risk Mitigation
- Regular monitoring protocols
- Contingency planning
- Stakeholder communication

Generated: {datetime.now().isoformat()}
            """
            
            state["report_draft"] = report_content
        
        return state

class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Reviewer",
            role="Quality Reviewer",
            capabilities=["quality_assurance", "fact_checking", "final_approval"]
        )
    
    async def _specialized_processing(self, state: AgentState) -> AgentState:
        """Review and approve final outputs"""
        report = state.get("report_draft", "")
        
        if report:
            review_results = {
                "quality_score": 0.92,
                "completeness": "high",
                "accuracy": "verified",
                "recommendations": [
                    "Report meets quality standards",
                    "All sections properly documented",
                    "Approved for final delivery"
                ],
                "approved": True,
                "reviewed_at": datetime.now().isoformat()
            }
            
            state["review_results"] = review_results
            # Mark workflow as complete
            state["workflow_complete"] = True
        
        return state

async def run_multi_agent_workflow():
    """Run a complete multi-agent workflow demonstration"""
    print("=" * 70)
    print("LANGGRAPH MULTI-AGENT SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Initialize the orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Define the initial state
    initial_state = {
        "messages": [HumanMessage(content="I need a comprehensive market analysis report for AI automation tools")],
        "current_task": "Create a comprehensive market analysis report for AI automation tools including research, analysis, and final report",
        "research_results": {},
        "analysis_results": {},
        "report_draft": "",
        "agent_responses": {},
        "task_history": [],
        "next_agent": None
    }
    
    print(f"\n📋 Initial Task: {initial_state['current_task']}")
    print("\n🚀 Starting workflow execution...\n")
    
    # Run the workflow
    config = {"configurable": {"thread_id": "demo_workflow"}}
    
    try:
        final_state = None
        step_count = 0
        max_steps = 10  # Prevent infinite loops
        
        async for state in orchestrator.app.astream(initial_state, config=config):
            step_count += 1
            if step_count > max_steps:
                print("⚠️  Maximum steps reached, stopping execution")
                break
            
            final_state = state
            
            # Print step information
            for node_name, node_state in state.items():
                if node_name in orchestrator.agents:
                    print(f"📌 Step {step_count}: {node_name.upper()} Agent")
                    
                    if node_state.get("agent_responses", {}).get(orchestrator.agents[node_name].name):
                        response = node_state["agent_responses"][orchestrator.agents[node_name].name]
                        print(f"   Response: {response[:100]}...")
                    
                    # Show progress indicators
                    if node_state.get("research_results"):
                        print("   ✅ Research completed")
                    if node_state.get("analysis_results"):
                        print("   ✅ Analysis completed")
                    if node_state.get("report_draft"):
                        print("   ✅ Report draft created")
                    if node_state.get("review_results"):
                        print("   ✅ Review completed")
                    
                    print()
            
            # Check if workflow is complete
            if any(state_dict.get("workflow_complete", False) for state_dict in state.values()):
                print("🎉 Workflow completed successfully!")
                break
        
        # Display final results
        if final_state:
            print("\n" + "=" * 50)
            print("FINAL RESULTS")
            print("=" * 50)
            
            for node_name, node_state in final_state.items():
                if node_name in orchestrator.agents:
                    task_count = len([t for t in node_state.get("task_history", []) if t.get("agent") == orchestrator.agents[node_name].name])
                    print(f"🤖 {orchestrator.agents[node_name].name}: {task_count} tasks completed")
            
            # Show report summary if available
            report = None
            for node_state in final_state.values():
                if node_state.get("report_draft"):
                    report = node_state["report_draft"]
                    break
            
            if report:
                print(f"\n📄 Report Generated: {len(report)} characters")
                print("   Executive Summary section included ✅")
                print("   Research Findings section included ✅")
                print("   Analysis Results section included ✅")
                print("   Recommendations section included ✅")
            
            # Show review results if available
            review = None
            for node_state in final_state.values():
                if node_state.get("review_results"):
                    review = node_state["review_results"]
                    break
            
            if review:
                print(f"\n✅ Quality Review: {review['quality_score']} score")
                print(f"   Status: {'APPROVED' if review['approved'] else 'NEEDS REVISION'}")
    
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

# Additional utility functions
async def demonstrate_agent_capabilities():
    """Demonstrate individual agent capabilities"""
    print("\n🔧 AGENT CAPABILITIES DEMONSTRATION")
    print("-" * 50)
    
    agents = [
        CoordinatorAgent(),
        ResearchAgent(),
        AnalystAgent(),
        WriterAgent(),
        ReviewerAgent()
    ]
    
    for agent in agents:
        print(f"\n🤖 {agent.name} ({agent.role})")
        print(f"   Capabilities: {', '.join(agent.capabilities)}")
        
        # Create a test state
        test_state = {
            "messages": [],
            "current_task": f"Test task for {agent.name}",
            "research_results": {"sample": "data"} if agent.name != "Researcher" else {},
            "analysis_results": {"sample": "analysis"} if agent.name not in ["Researcher", "Analyst"] else {},
            "report_draft": "",
            "agent_responses": {},
            "task_history": [],
            "next_agent": None
        }
        
        try:
            result_state = await agent.process(test_state)
            print(f"   ✅ Processing successful")
            
            if agent.name == "Researcher" and result_state.get("research_results"):
                print(f"   📊 Research data generated with {len(result_state['research_results'].get('key_findings', []))} findings")
            elif agent.name == "Analyst" and result_state.get("analysis_results"):
                print(f"   📈 Analysis completed with confidence: {result_state['analysis_results'].get('statistical_results', {}).get('key_metrics', {}).get('confidence', 'N/A')}")
            elif agent.name == "Writer" and result_state.get("report_draft"):
                print(f"   📝 Report draft created ({len(result_state['report_draft'])} characters)")
            elif agent.name == "Reviewer" and result_state.get("review_results"):
                print(f"   ✅ Review completed - Approved: {result_state['review_results'].get('approved', False)}")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")

# Main execution
async def main():
    """Main execution function"""
    print("🌟 Welcome to the LangGraph Multi-Agent System Demo!")
    print("This demonstration showcases a collaborative workflow with specialized agents.")
    
    # Run individual agent demonstrations
    await demonstrate_agent_capabilities()
    
    # Run the complete workflow
    await run_multi_agent_workflow()
    
    print("\n🎯 Key Features Demonstrated:")
    print("   • State-based workflow management")
    print("   • Specialized agent roles and capabilities")
    print("   • Inter-agent communication and coordination")
    print("   • Dynamic task routing and decision making")
    print("   • Tool integration and usage")
    print("   • Quality assurance and review processes")
    print("   • Persistent state management with checkpoints")

if __name__ == "__main__":
    # Note: You'll need to install the required packages:
    # pip install langgraph langchain-openai langchain-anthropic langchain-core
    print("📦 Required packages: langgraph, langchain-openai, langchain-anthropic, langchain-core")
    print("💡 Install with: pip install langgraph langchain-openai langchain-anthropic langchain-core")
    print()
    
    # Run the demonstration
    asyncio.run(main())
