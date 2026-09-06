# agent_langgraph,py
# LinedIn agent rebult as LangGraph

import os
from typing import Annotated, TypedDict
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3





load_dotenv()
Path("outputs").mkdir(exist_ok=True)

SYSTEM= """You are a LinkedIn content strategist.

Role: Operations PM in finance services -> AL catalyst
Audience: Fiance and operations professionals (non-technical)
Tone: Professional but human. humble and reflecive
Steps: Search knowledge base -> search web -> calcualte if needed -> write post
Post:150 - 300 words
Tag always: #BuildInPublic #AIJourney
"""


#====================================================================
# STATE - The typed dict that flows through every node
#====================================================================
class AgentState(TypedDict):
    """State that presists across all graph nodes."""
    messages: Annotated[list[BaseMessage], operator.add]
    human_approved: bool # track approve status
    draft_post: str

#====================================================================
# FAISS - load existing index
#====================================================================

def _load_faiss():
    """Load your existing FAISS index from faiss_index/ folder."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        index_path = "faiss_index"
        if not (Path(index_path)/ "index.faiss").exists():
            print("⚠️ faiss_index/ not found - run python3 build_faiss.py first")
            return None
        store = FAISS.load_local(
            folder_path = index_path,
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small"),
            allow_dangerous_deserialization = True
        )
        print("✅ FAISS KB loaded")
        return store
    except Exception as e:
        print(f"⚠️ FAISS:{e}")
        return None
    
# Load at startup - only once
faiss_store = _load_faiss()


#====================================================================
# Tools - same as before
#====================================================================

#-------TOOL 1: WEB SEARCH ------------------------
@tool
def search_web(query: str) -> str:
    """Search the internet for current, real-time information.
    Use for: news, trends, stats, live documentation.
    DO NOT use for math. Returns top search results."""
    from tavily import TavilyClient
    t = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    try:
        results = t.search(query=query, max_results=3)
        output = [
            f"TITLE: {r['title']}\n URL: {r['url']}\n INFO: {r['content'][:400]}"
            for r in results.get("results", [])
        ]
        return "\n---\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Search failed: {e}. Try different keywords."
        

#--------TOOL 2: CALCULATOR----------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a Python math expression.
    Use for ALL arithmetic. Not for search.
    GOOD: '(150/7)*100'. BAD: 'what is 150 divided by 7.'
    Returns: numeric result."""
    try:
        result = eval(expression, {"__builtins__":{}})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculateion error: {e}. Check expression."

#--------TOOL 3: LINKEDIN POST WRITER---------------------
@tool
def write_linkedin_post(title: str, content: str) -> str:
    """Write and save a LinkedIn post as a .txt file.
    Use ONLY after completing research. Always research first.
    Returns: path to saved file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/{title}_{ts}.txt"
    with open(path, "w") as f:
        f.write(content)
    return f"Post saved: {path}"

#------TOOL 4: FAISS from own document--------------------



@tool
def search_knowledge_base(query: str) -> str:
    """Search personal knowledge base - LinkedIn posts and finance/ops notes.
    Use for: past post topucs, your writing style, finance domain knowlwdge.
    Do NOT use for current news or live events - use search_web for that.
    Returns: relevant text chunks from your stored documents."""
    if faiss_store is None:
        return(
            "Knowledge base not available."
            "Use search_web instead."
        )
    try:
        results = faiss_store.similarity_search(query, k=3)
        if not results:
            return "No relevant documents found. Try web_search."
        chunks = [
            f"[Doc {i+1} - {r.metadata.get('category', 'kb')}]:"
            f"{r.page_content[:400]}"
            for i, r in enumerate(results)
        ]
        return "\n\n".join(chunks)
    except Exception as e:
        return f"Knowledge base research failed: {e}. Use web_search."

# ----------Tool list ------------------------------------
# register all tools - just a list, no REGISTRY dict needed
TOOLS = [search_knowledge_base, search_web, calculator, write_linkedin_post]



#====================================================================
# LLM + TOOL NODE
#====================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)

#====================================================================
# GAPRH NODES
#====================================================================

def agent_node(state: AgentState) -> dict:
    """LLM decide next action"""
    messages = state['messages']
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM)] + messages
    response = llm_with_tools.invoke(messages)
    return{"messages": [response]}

def humman_review_node(state: AgentState) -> dict:
    """Human-in-the-loop gate using new interrupt() API
    
    IMPORTANT RULES:
    - interrupt() can be called anywhere in a node
    - DO NOT wrap in try/ except - GraphInterrupt must propagate
    - Node re-executes from start on resume - keep side effects afer interrupt()
    - Command (resume=value) makes value the return of interrupt"""

    #Ectract the last writtern post form messages for review
    draft = "No Draft found yet"
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if len(msg.content) > 100: #likely a post not tool call
                draft = msg.content[:600]
                break   

    # interrupt() PAUSE here - send payload to the human
    # DO NOT put try/ except in this call
    # the return value = wahtever Command(resume=Value) passes back
    human_decision = interrupt({
        "message": "🔎 Human Review Required",
        "draft_preview": draft,
        "option": ["approve", "reject"],
        "instruction": "Type 'approve' to save the post, 'reject; to discard."
    })

    # --- Run after Command(resume=human_decision) called ----
    approved = str(human_decision).strip().lower() == "approve"
    print(f"\n [HUMAN DECISION] {'✅Approved' if approved else '❎Rejected'}")
    return{
        "human_approve": approved,
        "draft_post": draft
    }
#====================================================================
# CONDITONAL EDGE
#====================================================================

def should_continue(state: AgentState) -> str:
    """Route: tools -> back to agent, or no tools -> human review."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "human_review"

def after_review(state: AgentState) -> str:
    """Route after human review: aprroved -> end, rejected -> back to agent."""
    if state.get("human_approved"):
        return "end"
    return "agent"

#====================================================================
# BUILD THE GRAPH
#====================================================================

graph = StateGraph(AgentState)

#add nodes
graph.add_node("agent",     agent_node)
graph.add_node("tools",     tool_node)
graph.add_node("human_review", humman_review_node)

#Entry POnt
graph.set_entry_point("agent")

# Agent -> tools or human_review
graph.add_conditional_edges(
    "agent", should_continue,
    {"tools": "tools", "human_review": "human_review"}
)

# Tools -> back to agent (ReAct)
graph.add_edge("tools", "agent")

#After human review -> end or back to agent
graph.add_conditional_edges(
    "human_review", after_review,
    {"end": END, "agent": "agent"}
)

#====================================================================
# COMPILE WITH SQLTE CHECKPOINTER
# 
#====================================================================

conn = sqlite3.connect("langgraph_state_db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app= graph.compile(
    checkpointer=checkpointer
)

#====================================================================
# RUN WITH HUMAN-IN-THE-LOOP
# 
#====================================================================

def run_with_human_review(topic: str, thread_id: str = "default"):
    """Run agent until t writes a post, then pause for human revie,
    After human approves, saves and finishes. if rejected, loops back."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=f"Write a LinkedIn Post about: {topic}.")],
        "human_approved": False,
        "draft_post": ""
    }

    print(f"\n{"="*55}")
    print(f" LANGGRAPH AGENT: {topic[:50]}")
    print(f"\n{"="*55}")

    # -----Phase 1: Run Agent until interrupt() hit --------
    print("Phase 1: researching and writing...")
    result = app.invoke(initial_state, config=config)

    # Extract & display the intrrupt payload
    if hasattr(result, "__interrupt__") or "__interrupt__" in str(type(result)):
        interrupt_payload = result
    else:
        interrupt_payload = result

    print("\n"+"-"*55)
    print("⏸️ HUMAN REVIEW REQUIRED")
    print("\n"+"-"*55)
    print("\n Draft post preview:")

    for msg in reversed(result.get("messages",[])):
        if hasattr(msg, "content") and len(msg.content)> 100:
            print(f"\n {msg.content[:500]}...")
            break

      # -----Phase 2: Get Human decision --------
    print(f"\n{"="*55}")
    decision = input("Your decison (approve/ reject): ").strip().lower()

    #-------Phase 3: Resume with Command(resume=decision)------
    print(f"\n Phase 3: Resuming with decision: {decision}")
    final = app.invoke(
        Command(resume=decision),
        config=config
    )

    print("\n ✅ Agent complete.")
    return final
#====================================================================
# ENTRY POINT
#====================================================================

if __name__ == "__main__":
    run_with_human_review(
        topic="how to lead in AI era",
        thread_id="trial001"
    )
