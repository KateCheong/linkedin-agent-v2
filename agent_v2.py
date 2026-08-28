# agent_v2.py - Week 3 LangChain version

import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent

#session memory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#Wrap FAISS & as tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
Path("outputs").mkdir(exist_ok=True)

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

def _load_faiss():
    """Load your existing FAISS index from faiss_index/ folder."""
    index_path = "faiss_index"

    if not Path(index_path).exists():
        print("⚠️ faiss_index/ not found.")
        print(" Run: python3 build_faiss.py to build it first.")
        return None

    #Verify index file exist
    if not (Path(index_path) / "index.faiss").exists():
        print(" ⚠️ index.faiss not found inside faiss_index/")
        return None

    try:
        store = FAISS.load_local(
            folder_path = index_path,
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small"),
            allow_dangerous_deserialization= True
        )
        print(f"✅ FAISS knowledge base loaded from {index_path}/")
        return store
    
    except Exception as e:
        print(f"⚠️  FAISS load failed: {e}.")
        return None

# Load at startup - only once
faiss_store = _load_faiss()

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

#---------LLM-----------------------------------

SYSTEM_PROMPT = f"""You are a LinkedIn content strategist.

Role: Operations PM in finance services
Goal: Build in Public - AI journey from ops PM to AI Catalyst
Audience: Fiance and operations professionals (non-technical)
Tone: Professional but human. humble and reflecive
Tag always: #BuildInPublic #AIJourney
Previous posts: Production-greade building, AI catalyst mission.

CHAIN-OF-THOUGHT STEPS (strict order):
STEP 1 - SEARCH KNOWLEDGE BASE: personal stored docs (LinkedIn post + finance notes): search for my tone, my perference, and avoid duplicating with past topic
STEP 2 - RESEARCH: search the web 2 to 3 times. Get real data.
STEP 3 - CALCULATION: if stat found, derive an insight.
STEP 4 - WRITE: Use WriteLinkedInPOst only after Steps 1 to 3.

Post requirements:
- Hook: myth-busting or reflective
- Body: 3 insights from research (not generic)
- Human role in AI always mentioned
- End with a question for a finance community
- 150 to 300 words
- Tags: #BuildInPublic #AIJourney + topic tags

Never hallucinate statistics. Bever skip research.
"""

#=======================================================
# Create Agent 
#=======================================================
agent = create_agent(
    model="gpt-4o-mini",
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT
)

#=======================================================
# In-memory store - map session_id -> history
# Prod - swap with RedisChatMessageHistory or SQLite
#=======================================================
store = {} #{session_id: ChatMessageHistory}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (or create) the history for a give session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#-----Wrap the agent with memory--------------------------
agent_with_memory = RunnableWithMessageHistory(
    agent,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history",
)

#=======================================================
# Run Function
#=======================================================
def run_agent(topic: str, session_id: str ="default") -> str:
    print(f"\n{'-'*52}")
    print(f"    AGENT: {topic[:60]}")
    print(f"{'='*52}")

    result = agent_with_memory.invoke(
        {"messages": [{"role": "user","content": f"Write a LinkedIn post about: {topic}."}]},
        config={"configurable":{"session_id": session_id}}
    )

    #Final answer is always the last message
    final = result["messages"][-1].content
    print(f"\n[Done] {final}")
    return final

#=======================================================
# ENTRY POINT
#=======================================================

if __name__ == "__main__":
    run_agent(
        "What have I write in LinkedIn" "and what is a current AI in finance topic I have not covered?",
        session_id="Planning_001"
    )

    run_agent(
            "Now write a post about the best topic you just suggested.",
            session_id="Planning_001"
        )

    run_agent(
            "what does my knowledge base say about Project Management and AI?",
            session_id="session_001"
        )