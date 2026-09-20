#agent_crew_mem0.py
# Multi-agent crew with MEM0 episodic memory
# Pattern: Retrieve-Reason-Store 
# Folder: linkedin_agent_v2

import os
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from mem0 import MemoryClient

load_dotenv()
Path("outputs").mkdir(exist_ok=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
mem0 = MemoryClient()
USER_ID = "linkedin_user"

# ===============================================================================
# TOOLS 
# ===============================================================================

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
def save_post(title: str, content: str) -> str:
    """Save the final LinkedIn post to outputs/ folder."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)[:40]
    path = f"outputs/crew_{safe}_{ts}.txt"
    Path(path).write_text(content, encoding="utf-8")
    return f"Past saved: {path}"

# ===============================================================================
# MEM0 NODES - wrap around the supervisor
# ===============================================================================

def retrieve_memory_node(state: MessagesState) -> dict:
    """Retrieve relevant past memories and nject into context."""
    last_msg = state["messages"][-1].content

    # Search Memo for relevant memories
    results = mem0.search(
        query=last_msg,
        filters = {'user_id': USER_ID},
        limit = 5
    )
    memories = results.get("results", [])

    if memories:
        mem_text = "\n".join([f"- {m['memory']}" for m in memories])
        context = f""" MEMORY CONTECT (from past sessions):
        {mem_text}
use this to :
- Avoid repeating topics already covered
- Match the user's established writing tone
- Serve ther finance/ ops audience correctly"""
        print(f"    [MEM0] Retreved {len(memories)} memories")
        return {"messages": [SystemMessage(content=context)]}

    print(" [MEM0] No memories yet - first session")
    return{"messages": []}

# def store_memory_node(state: MessagesState) ->dict:
#     """Store this session's interaction back to Mem0."""
#     messages = state['messages']

#     # Build conversation log from named worker messages
#     interaction = []
#     for msg in messages:
#         if isinstance(msg, (HumanMessage, AIMessage)):
#             role = "user" if isinstance(msg, HumanMessage) else "assistant"
#             if hasattr(msg, "name") and msg.name:
#                 role = msg.name     # researcher/ writer/ editor
#             if msg.content and len(msg.content) > 20:
#                 interaction.append({"role": role, "content": msg.content[:500]})

#     if interaction:
#         # Mem0 auto extract facts and logs the episode
#         mem0.add(
#             interaction,
#             user_id = USER_ID,
#             metadata = {
#                 "session_date": datetime.now().isoformat(),
#                 "source": "linkedin_crew"
#             }
#         )
#         print(f"    [MEM0] Stored {len(interaction)} interaction turns")

#     return {} # no state update needed

def store_memory_node(state: MessagesState) ->dict:
    """Store this session's interaction back to Mem0."""
    print(" [MEM0] store_memory_node triggered")
    messages = state['messages']

    #Build clean interacyion list
    interaction = []
    for msg in messages:
        #SKip SystemMessages - only sotre Huamn and AI messages
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            role = getattr(msg, "name", None) or "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue

        #Only store messages with actual content
        content = msg.content if isinstance(msg.content, str) else ""
        if content and len(content.strip()) > 20:
            interaction.append({
                "role": role,
                "content": content[:500]
            })
        if interaction: 
            try:
                mem0.add(
                    interaction,
                    user_id= USER_ID,
                    metadate = {
                        "session_date": datetime.now().isoformat(),
                        "sources": "linkedin_crew"
                    }
                )
                print(f"    [MEM0] Stored {len(interaction)} turns")

            except Exception as e:
                print(f"    [MEM0] Store failed (non-fatal): {e}")

        else:
            print(" [MEM0] Nothing to store")
        return{}

# ===============================================================================
# SUPERVISOR + WORKER NODES (same COmmand pattern)
# ===============================================================================

class Router(TypedDict):
    """Which worker agent should act next?"""
    next: Literal["researcher", "writer", "editor", "FINISH"]

#==========================================================================
# SUPERVSOR NODE - routes, never work
#==========================================================================

def supervisior_node(state: MessagesState) -> Command:
    system = """You are a supervisor for a LinkedIn content team.
    Workflow: researcher ➡ writer ➡ editor ➡ FINISH.
    Check history to avoid repeating covered topics.
    Use any memory context provided at the top."""

    messages = [{"role": "system", "content": system}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_agent = response["next"]

    print(f"    [SUPERVISOR] ➡ {next_agent}")

    if next_agent == "FINISH":
        return Command(goto="store_memory")
    return Command(goto= next_agent)

#==========================================================================
# WORKER NODEs - use creat_react_agent for built-in tool loop
#==========================================================================

# create_agent gives each worker its own ReAct loop
researcher_agent = create_agent(
    llm,
    tools=[search_web],
    system_prompt = """You are a research specialist for LinkedIn content.
    Your job: gather current facts, statistics, and relevant information.
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

def make_worker(agent, name):
    """Factory - creates a worker node that returns to supervisor"""
    def worker_node(state: MessagesState) -> Command:
        print(f"    [{name.upper()}] Working...")
        result = agent.invoke(state)
        last = result["messages"][-1]
        return Command(
            update={"messages": [HumanMessage(content=last.content, name= name)]},
            goto= "supervisor"
        )
    worker_node.__name__ = name
    return worker_node

#==========================================================================
# BUILD THE GRAPH - retrieve ➡ supervisor ➡ workers ➡ sotre ➡ END
#==========================================================================
builder = StateGraph(MessagesState)

# Memory nodes
builder.add_node("retrieve_memory", retrieve_memory_node)
builder.add_node("store_memory", store_memory_node)

# supervisor + worker nodes
builder.add_node("supervisor", supervisior_node)
builder.add_node("researcher", make_worker(researcher_agent, "researcher"))
builder.add_node("writer", make_worker(writer_agent, "writer"))
builder.add_node("editor", make_worker(editor_agent, "editor"))

# Flow: START ➡ retrieve ➡ supervisor ➡ workers ➡ sotre ➡ END
builder.add_edge(START, "retrieve_memory")
builder.add_edge("retrieve_memory", "supervisor")
builder.add_edge("store_memory", END)

app = builder.compile(checkpointer=MemorySaver())

#==========================================================================
# RUN
#==========================================================================

def run_crew(topic: str, thread_id: str = "default") -> str:
    print(f"\n {'='*55}")
    print(f"    CREW + MEM0: {topic[:50]}")
    print(f"{'='*55}\n")
    
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
         {"messages": [HumanMessage(content=f"Create a LinkedIn post about: {topic}")]},
          config = config
    )
       
    
    final = result["messages"][-1].content
    print(f"\n [DONE] {final[:200]}...")
    return final

if __name__ == "__main__":
    # Run 1 - no memory yet
    run_crew("AI agents with memories applcation in finance industry")

    # Run 2 - with run 1 contect
    run_crew("Human role in finance industry - why people still matter?")