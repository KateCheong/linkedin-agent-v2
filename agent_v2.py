# agent_v2.py - Week 3 LangChain version

import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent



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

# register all tools - just a list, no REGISTRY dict needed
TOOLS = [search_web, calculator, write_linkedin_post]

#---------LLM-----------------------------------

SYSTEM_PROMPT = f"""You are a LinkedIn content strategist.

Role: Operations PM in finance services
Goal: Build in Public - AI journey from ops PM to AI Catalyst
Audience: Fiance and operations professionals (non-technical)
Tone: Professional but human. humble and reflecive
Tag always: #BuildInPublic #AIJourney
Previous posts: Production-greade building, AI catalyst mission.

CHAIN-OF-THOUGHT STEPS (strict order):
STEP 1 - REAEARCH: search the web 2 to 3 times. Get real data.
STEP 2 - CALCULATION: if stat found, derive an insight.
STEP 3 - WRITE: Use WriteLinkedInPOst only after Steps 1 to 2.

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
# Run Function
#=======================================================
def run_agent(topic: str) -> str:
    print(f"\n{'-'*52}")
    print(f"    AGENT: {topic[:60]}")
    print(f"{'='*52}")

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Write a LinkedIn post about: {topic}."
        }]
    })

    #Final answer is always the last message
    final = result["messages"][-1].content
    print(f"\n[Done] {final}")
    return final

#=======================================================
# ENTRY POINT
#=======================================================

if __name__ == "__main__":
    run_agent(
        "how does AI changed from 2016 to 2026"
        "for operations professionals in financial services"
    )