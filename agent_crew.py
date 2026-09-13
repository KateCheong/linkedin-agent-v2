# agent_crew.py
# 3 workers: Researcher + Writer + Editor
# Supervisor uses Command API 
# Folder: lindin_agent_v2

import os
from typing import Literal
from typing_extensions import TypedDict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
Path("outputs").mkdir(exist_ok=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#==========================================================================
# TOOLS - each worker gets its own set
#==========================================================================

@tool
def search_web(query: str) -> str:
    """Search internet for current information. Use for news, trends, stats."""
    from tavily import TavilyClient
    try:
        t = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = t.search(query=query, max_results=3)
        output = [
            f"TITLE: {r['title']}\nINFO: {r['content'][:350]}"
            for r in results.get("results",[])
        ]
        return "\n----\n".join(output) if output else "No results."
    except Exception as e:
        return f"Search failed: {e}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search personal FAISS knowledge base for past posts and finance knowledge."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        if not (Path("faiss_index")/"index.faiss").exists():
            return "Knowledge base not found. Use search web."
        store = FAISS.load_local(
            "faiss_index",
            OpenAIEmbeddings(model="text-embedding-3-small"),
            allow_dangerous_deserialization=True
        )
        results = store.similarity_search(query, k=3)
        return "\n\n".join([r.page_content[:350]] for r in results) if results else "No docs found."
    except Exception as e:
        return f"KB search failed: {e}."

@tool
def save_post(title: str, content: str) -> str:
    """Save the final LinkedIn post to outputs/ folder."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)[:40]
    path = f"outputs/crew_{safe}_{ts}.txt"
    Path(path).write_text(content, encoding="utf-8")
    return f"Past saved: {path}"

#==========================================================================
# SUPERVSOR ROUTING SCHEMA
#==========================================================================

class Router(TypedDict):
    """Which worker agent should act next?"""
    next: Literal["researcher", "writer", "editor", "FINISH"]

#==========================================================================
# SUPERVSOR NODE - routes, never work
#==========================================================================

def supervisior_node(state: MessagesState) -> Command[Literal[
    "researcher", "writer", "editor", "__end__"
]]:
    system = """You are a supervisor for a LinkedIn content team.
    Workers available: researcher, writer, editor.
    Standard workflow:
        1. researcher - gather facts and current information.
        2. writer - draft the LinkedIn post using research.
        3. editor - refine tone, check length (150-300 words), add hastags, save.
        4. FINISH - when editor has saved the final post.
        
    Look at conversation history to determine what has been done.
    Route to FINISH only when editor has completed and saved the post."""

    messages = [{"role": "system", "content": system}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_agent = response["next"]

    print(f"    [SUPERVISOR] ➡ routing to: {next_agent}")

    if next_agent == "FINISH":
        return Command(goto=END)
    return Command(goto= next_agent)

#==========================================================================
# WORKER NODEs - use creat_react_agent for built-in tool loop
#==========================================================================

# create_agent gives each worker its own ReAct loop
research_agent = create_agent(
    llm,
    tools=[search_web, search_knowledge_base],
    system_prompt = """You are a research specialist for LinkedIn content.
    Your job: gather current facts, statistics, and relevant information.
    Search both the web and personal knowledge bsae.
    Return a structured research summary with key facts ans sources."""
)

writer_agent = create_agent(
    llm,
    tools=[],
    system_prompt = """You are a LinkedIn post writer for an Operations PM in financial services.
    Your audence: finance and operations professionals (non-technical).
    Tone: professional, humble, reflective.
    Given the research provided, write a LinkedIn post:
    - Hook: myth-busting or reflective opening line
    - Body: 3 concrete insights from research
    - Human angle: always show the human role n AI
    - Close: one question for finance/ops community
    - Length: 150 - 300 words
    - Tags: #BuildnPublic #AIJourney + topic tags
    Write the full post only - no preamble."""
)

editor_agent = create_agent(
    llm,
    tools=[save_post],
    system_prompt="""You are a LinkedIn content editor.
    Review the draft post for:
    1. Length: must be 150 - 300 words (count carefully)
    2. Tone: professional but human, not robotic.
    3. Hook: first line must grab attention immediately
    4. Hastags: #BuildInPuclic and #AIJourney must be present
    4. Human angle: must mention the human's role in AI
    
    Make refinements, then use save_post to save the final version.
    Title format: use the main topic as the file slug"""
)

def researcher_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("     [RESEARCHER] Gathering information...")
    result = research_agent.invoke(state)
    last_msg = result["messages"][-1]
    return Command(
        update={"messages": [HumanMessage(
            content=last_msg.content, name="researcher"
        )]},
        goto="supervisor"
    )

def writer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("     [WRITER] Drafting LinkedIn post...")
    result = editor_agent.invoke(state)
    last_msg = result["messages"][-1]
    return Command(
        update={"messages": [HumanMessage(
            content=last_msg.content, name="editor"
        )]},
        goto="supervisor"
    )

def editor_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    print("     [EDITOR] Refining and saving post...")
    result = editor_agent.invoke(state)
    last_msg = result["messages"][-1]
    return Command(
        update={"messages": [HumanMessage(
            content=last_msg.content, name="editor"
        )]},
        goto="supervisor"
    )

#==========================================================================
# BUILD THE GRAPH - only ONE explicit edge needed
#==========================================================================
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisior_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)
builder.add_edge(START, "supervisor")

app = builder.compile(checkpointer=MemorySaver())

#==========================================================================
# RUN
#==========================================================================

def run_crew(topic: str, thread_id: str = "crew_001") -> str:
    print(f"\n {'='*55}")
    print(f"    CREW STARTING: {topic[:50]}")
    print(f"{'='*55}\n")

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(
            content=f"Create a LinkedIn post about: {topic}"
        )]
    }

    result = app.invoke(initial_state, config=config)
    final = result["messages"][-1].content
    print(f"\n [CREW DONE] {final[:200]}...")
    return final

if __name__ == "__main__":
    run_crew(
        "Testing, how to stay healthy"
    )