#build_faiss.py
# Run once: python3 build_faiss.py
#Re-run when add content to knowledge_base/

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#-----config------------------------------------
KNOWLEDGE_DIR = Path("knowledge_base")
INDEX_PATH = "faiss_index"
CHINK_SIZE = 400
CHUNK_OVERLAP = 50

#---------Step 1: LOAD .txt files ---------------
def load_text_files() -> list[Document]:
    """Load all .txt files, split by --- separator, tag with metadata."""
    docs = []
    for txt_file in KNOWLEDGE_DIR.rglob("*.txt"):
        print(f"    Loading: {txt_file}")
        content = txt_file.read_text(encoding='utf-8')
        sections = [s.strip() for s in content.split("---") if s.strip()]

        for section in sections:
            #Skip comment-only sections
            if section.startswith("#") and len(section) <100:
                continue
            docs.append(Document(
                page_content=section,
                metadata={
                    "source": str(txt_file),
                    "category": txt_file.parent.name,
                    "file": txt_file.name
                }
            ))
    print(f"    Loaded {len(docs)} sections total.")
    return docs


#---------Step 2: CHUNK LONG SECTIONS ---------------
def chunk_documents(docs: list) -> list:
    """Split sections into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHINK_SIZE,
        chunk_overlap= CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"    Split into {len(chunks)} chunks")
    return chunks

#---------Step 3: Build and Save FAISS ---------------
def build_faiss():
    print("\n" + "="*48)
    print(" Building FAISS Knowledge Base")
    print("="*48 + "\n")

    # Check content exists
    if not KNOWLEDGE_DIR.exists():
        print("ERROR: knowledge_base/ folder not found.")
        print("Run Step 2 first: mkdir -p knowledge_base/my_posts knowledge/finance_ops")
        return

    #Load
    print("Loading content files...")
    docs = load_text_files()
    if not docs:
        print("No content found. add .txt files to knowledge_base/ first.")
        return

    # Chunk
    print("\nChunking content...")
    chunks = chunk_documents(docs)

    #Embed + build FAISS
    print("\nEmbedding with OpenAI text-embedding-3-small ...")
    print("(This coest ~$0.001 - runs for about 10 secounds)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_documents(chunks, embeddings)

    # Save to disk
    store.save_local(INDEX_PATH)
    print(f"\n ✅ FAISS index saved to: {INDEX_PATH}/")
    print(f"    Sections: {len(docs)}")
    print(f"    Chunks: {len(chunks)}")

    # Quick test
    print("\n --- Quick Search Test ---")
    for query in ["AI agents in finance", "my LinkedIn posts about building AI"]:
        results = store.similarity_search(query, k=1)
        if results:
            r = results[0]
            print(f"\n Query: '{query}")
            print(f"Category: [{r.metadata['category']}]")
            print(f"Result: {r.page_content[:150]}...")
    print("\n ✅ Build complete! Run agent_v2.py to use your knowledge base.")


if __name__ == "__main__":
    build_faiss()

