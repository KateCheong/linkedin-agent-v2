#agent_response.py 
# Responses API + Conversation API + Vector Stores
# The Assistant API is doen

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI()
Path("outputs").mkdir(exist_ok=True)

SYSTEM= """You are a LinkedIn content strategist.

Role: Operations PM in finance services -> AL catalyst
Audience: Fiance and operations professionals (non-technical)
Tone: Professional but human. humble and reflecive
Steps: Search knowledge base -> search web -> calcualte if needed -> write post
Post:150 - 300 words
Tag always: #BuildInPublic #AIJourney
"""

#=================================================
# Step 1 : CREATE VECTOR STORE (once saved the ID)
# Replace FAISS and Assistant object
#=================================================

def create_vector_store() -> str:
    """Upload Knowledge_vase/ files to a vector store.
    Run once - then save and reuse VECTOR_STORE_ID."""
    kb_dir = Path("knowledge_base")
    if not kb_dir.exists():
        print("No Knowleged_base/ folder found. Skipping.")
        return None

    # Upload files
    file_ids =[]
    for txt_file in kb_dir.rglob("*.txt"):
        print(f" Uploading: {txt_file.name}")
        with open(txt_file, "rb") as f:
            resp = client.files.create(file=f, purpose="assistants")
            file_ids.append(resp.id)

    if not file_ids:
        print("No .txt files found in knowldge_base/")
        return None

    #create vector store and wait for indexing
    vs = client.vector_stores.create(name="LinkedIn Knowledge Base")
    batch = client.vector_stores.file_batches.create_and_poll(
        vector_store_id = vs.id,
        file_ids=file_ids
    )
    if batch.status != "completed":
        print(f"Warning: Vector store status = {batch.status}.")

    print(f"✅ Vector store ready : {vs.id}")
    print(f" Save this: VECTOR_STORE_ID: '{vs.id}'")
    return vs.id


#=================================================
# Step 2 : CREATE CONVERSATION (once per session)
# Replace thread - store history server-side
#=================================================

def create_conversation() ->str:
    """Create a new COnversation for a session.
    Returns conversation_od - reuse for multi-turn."""
    conv = client.conversations.create()
    print(f"✅ Coversation created: {conv.id}")
    return conv.id

#=================================================
# Step 3 : RUN - Responses API (no polling)
# Replace Run - Polling loop
#=================================================

def run_agent(
        topic: str,
        conversation_id: str,
        vector_store_id: str = None
) -> str:
    """Run the agent usng the Response API.
    Synchronous - no polling loop needed.
    Pass same conversation_id for multi-turn memory."""

    print(f"\n {'='*55}")
    print(f"    TOPIC: {topic[:55]}")
    print(f"{'='*55}")

    # Build Tools list
    tools = []
    if vector_store_id:
        tools.append({
            "type": "file_search",
            "vector_store_ids":[vector_store_id]
        })

    # Create response - SYNCHRONOUS, no while loop
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM,
        conversation=conversation_id,
        input=f"Write a LinkedIn post about: {topic}.",
        tools=tools
    )

    # Get the answer directly - no pooling
    final = response.output_text
    print(f"\n [DONE] {final[:150]}...")

    # Save to outputs/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "_-" else "_"
                   for c in topic)[:40]
    path = f"outputs/{safe}_{ts}.txt"
    Path(path).write_text(final, encoding="utf-8")
    print(f"    Saved: {path}")
    return final
#=================================================
# MAIN
#=================================================

if __name__ == "__main__":
    # --- SETUP: Run these ONCE then save IDs ------------
    # ----STEP 1: Create vector store (save you knowledge base)
    #VENCTOR_STORE_ID = create_vector_store()
    # After 1st run: 
    VENCTOR_STORE_ID = "vs_6aa66deb829c81919d7d87a3fbec4803"

    # ---STEP 2: Create conversation (once per session)
    CONVERSATION_ID = create_conversation()
    # Reuse this ID for multi-turn in the same session

    # --TEST 1: Research and write a post ------------------
    run_agent(
        topic="keeping up in AI era",
        conversation_id = CONVERSATION_ID,
        vector_store_id=VENCTOR_STORE_ID
    )

    # --TEST 2: Research and write a post ------------------
    run_agent(
        topic="Now make it shorter - under 150 words",
        conversation_id = CONVERSATION_ID,
        vector_store_id=VENCTOR_STORE_ID
    )