import os
import uuid
from html import escape
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# ==============================
# ENVIRONMENT SETUP
# ==============================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "pcsk_5DrW9k_TUGaPA9NdjkseTgWpTzhkyeHeAfyubh8nG3pR8QJsWYEZFS47LgGwk7WxWV3Som")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "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")
NVIDIA_API_KEY1 = os.getenv("NVIDIA_API_KEY1", "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")
INDEX_NAME = os.getenv("PINECONE_INDEX", "sample")
EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.1-70b-instruct")
DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1024"))

# ==============================
# INITIALIZE PINECONE
# ==============================

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# ==============================
# FASTAPI APP
# ==============================

app = FastAPI(title="DocuMind Enterprise RAG")


def render_page(title, eyebrow, heading, description, body_html, side_html=""):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{escape(title)}</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

            :root {{
                --paper: #f5efe4;
                --ink: #102417;
                --muted: #4a5d4f;
                --sea: #0f766e;
                --moss: #8ea86d;
                --sun: #f08a24;
                --card: rgba(255, 252, 247, 0.82);
                --line: rgba(16, 36, 23, 0.12);
                --shadow: 0 24px 60px rgba(30, 50, 38, 0.14);
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                min-height: 100vh;
                font-family: "Space Grotesk", "Segoe UI", sans-serif;
                color: var(--ink);
                background:
                    radial-gradient(circle at top left, rgba(240, 138, 36, 0.26), transparent 24%),
                    radial-gradient(circle at top right, rgba(15, 118, 110, 0.20), transparent 30%),
                    linear-gradient(180deg, #fbf7ef 0%, var(--paper) 52%, #efe7d8 100%);
            }}

            .shell {{
                width: min(1120px, calc(100% - 32px));
                margin: 0 auto;
                padding: 28px 0 56px;
            }}

            .topbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 16px;
                margin-bottom: 28px;
                padding: 16px 20px;
                border: 1px solid var(--line);
                border-radius: 999px;
                background: rgba(255, 252, 247, 0.74);
                backdrop-filter: blur(12px);
                box-shadow: var(--shadow);
            }}

            .brand {{
                color: var(--ink);
                text-decoration: none;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                font-size: 0.92rem;
            }}

            nav {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}

            nav a,
            .button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                padding: 11px 16px;
                border-radius: 999px;
                border: 1px solid rgba(16, 36, 23, 0.14);
                background: rgba(255, 255, 255, 0.55);
                color: var(--ink);
                text-decoration: none;
                font-weight: 500;
                transition: transform 150ms ease, background 150ms ease, border-color 150ms ease;
            }}

            nav a:hover,
            .button:hover {{
                transform: translateY(-1px);
                background: rgba(255, 255, 255, 0.92);
                border-color: rgba(16, 36, 23, 0.26);
            }}

            .hero {{
                display: grid;
                grid-template-columns: minmax(0, 1.5fr) minmax(280px, 0.9fr);
                gap: 24px;
                align-items: stretch;
                margin-bottom: 24px;
            }}

            .panel,
            .hero-card,
            .mini-card {{
                border: 1px solid var(--line);
                border-radius: 28px;
                background: var(--card);
                box-shadow: var(--shadow);
                backdrop-filter: blur(12px);
            }}

            .hero-copy,
            .hero-card,
            .panel {{
                padding: 28px;
            }}

            .eyebrow {{
                margin: 0 0 12px;
                color: var(--sea);
                text-transform: uppercase;
                letter-spacing: 0.16em;
                font-size: 0.8rem;
                font-weight: 700;
            }}

            h1,
            h2,
            h3 {{
                margin: 0;
                line-height: 0.96;
            }}

            h1 {{
                font-size: clamp(2.5rem, 5vw, 5.6rem);
                max-width: 10ch;
            }}

            .lede {{
                max-width: 56ch;
                font-size: 1.02rem;
                line-height: 1.7;
                color: var(--muted);
                margin: 18px 0 0;
            }}

            .stack {{
                display: grid;
                gap: 16px;
            }}

            .mini-card {{
                padding: 20px;
            }}

            .mini-card p,
            .panel p,
            li {{
                color: var(--muted);
                line-height: 1.7;
            }}

            .metric {{
                display: block;
                margin-top: 8px;
                font-size: 2rem;
                font-weight: 700;
                color: var(--ink);
            }}

            .content-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px;
            }}

            .action-card {{
                padding: 22px;
                border-radius: 24px;
                border: 1px solid var(--line);
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.8), rgba(255, 249, 241, 0.84));
            }}

            .action-card h3 {{
                font-size: 1.4rem;
                margin-bottom: 10px;
            }}

            form {{
                display: grid;
                gap: 14px;
                margin-top: 8px;
            }}

            .field-label {{
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: var(--muted);
            }}

            input[type="text"],
            input[type="file"] {{
                width: 100%;
                padding: 14px 16px;
                border-radius: 18px;
                border: 1px solid rgba(16, 36, 23, 0.15);
                background: rgba(255, 255, 255, 0.74);
                color: var(--ink);
                font: inherit;
            }}

            input[type="file"] {{
                padding: 16px;
            }}

            button {{
                appearance: none;
                border: 0;
                border-radius: 18px;
                padding: 14px 18px;
                font: inherit;
                font-weight: 700;
                color: white;
                background: linear-gradient(135deg, var(--sea), #169c92);
                box-shadow: 0 14px 26px rgba(15, 118, 110, 0.24);
                cursor: pointer;
                transition: transform 150ms ease, box-shadow 150ms ease;
            }}

            button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 18px 30px rgba(15, 118, 110, 0.28);
            }}

            .note,
            .mono {{
                font-family: "IBM Plex Mono", Consolas, monospace;
                font-size: 0.92rem;
            }}

            .note {{
                color: var(--muted);
            }}

            .answer {{
                margin: 0;
                padding: 18px;
                border-radius: 22px;
                background: #fffdf8;
                border: 1px solid rgba(16, 36, 23, 0.1);
                white-space: pre-wrap;
                line-height: 1.8;
                font-family: "IBM Plex Mono", Consolas, monospace;
                font-size: 0.96rem;
            }}

            .question-chip {{
                display: inline-block;
                margin-bottom: 14px;
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(240, 138, 36, 0.14);
                color: #7a4700;
                font-size: 0.92rem;
            }}

            .footer-row {{
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 18px;
            }}

            @media (max-width: 840px) {{
                .hero,
                .content-grid {{
                    grid-template-columns: 1fr;
                }}

                .topbar {{
                    border-radius: 28px;
                    align-items: flex-start;
                    flex-direction: column;
                }}

                .hero-copy,
                .hero-card,
                .panel {{
                    padding: 22px;
                }}
            }}
        </style>
    </head>
    <body>
        <main class="shell">
            <header class="topbar">
                <a class="brand" href="/">DocuMind Enterprise</a>
                <nav>
                    <a href="/">Overview</a>
                    <a href="/upload">Upload PDF</a>
                    <a href="/chat">Ask Question</a>
                </nav>
            </header>

            <section class="hero">
                <div class="hero-copy panel">
                    <p class="eyebrow">{escape(eyebrow)}</p>
                    <h1>{escape(heading)}</h1>
                    <p class="lede">{escape(description)}</p>
                </div>
                <aside class="hero-card">
                    <div class="stack">
                        {side_html or '''
                        <div class="mini-card">
                            <div class="eyebrow">Pipeline</div>
                            <strong>PDF -> chunks -> embeddings -> Pinecone</strong>
                            <p>Upload documents and query only against indexed context.</p>
                        </div>
                        <div class="mini-card">
                            <div class="eyebrow">Models</div>
                            <span class="metric">RAG</span>
                            <p>Embeddings are sent to NVIDIA and stored in Pinecone for retrieval.</p>
                        </div>
                        '''}
                    </div>
                </aside>
            </section>

            <section class="panel">
                {body_html}
            </section>
        </main>
    </body>
    </html>
    """


def wants_html(request: Request):
    return "text/html" in request.headers.get("accept", "").lower()

# ==============================
# EMBEDDING FUNCTION
# ==============================

def get_embedding(text, input_type):

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBED_MODEL,
            "input": [text],
            "input_type": input_type,
            "modality": "text",
            "dimensions": DIMENSION,
        },
        timeout=60,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"NVIDIA embeddings API error: {response.text}") from exc

    payload = response.json()
    data = payload.get("data")
    if not data:
        raise RuntimeError(f"Unexpected embeddings response: {payload}")

    return data[0]["embedding"]

# ==============================
# LLM RESPONSE FUNCTION
# ==============================

SYSTEM_PROMPT = """
You are DocuMind Enterprise.

Rules:
- Answer ONLY using the provided context.
- Provide source citation if possible.
- If answer not found say:
"This information is not available in the provided documents."
"""

def ask_llm(context, question):

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY1}",
            "Content-Type": "application/json"
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:{question}"
                }
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# ==============================
# WEEK 1 - DOCUMENT INGESTION
# ==============================

def ingest_pdf(path):

    elements = partition_pdf(path)

    texts = [e.text for e in elements if e.text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.create_documents(texts)

    vectors = []

    for chunk in chunks:

        embedding = get_embedding(chunk.page_content, input_type="passage")

        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": path
            }
        })

    index.upsert(vectors=vectors)

    return "Document Ingested Successfully"

# ==============================
# WEEK 2 - RETRIEVAL ENGINE
# ==============================

def retrieve_context(query):

    query_embedding = get_embedding(query, input_type="query")

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    context = ""

    for match in results["matches"]:
        context += match["metadata"]["text"] + "\n"

    return context

# ==============================
# API ENDPOINTS
# ==============================

@app.get("/", response_class=HTMLResponse)
def home():
    body_html = """
    <div class="content-grid">
        <article class="action-card">
            <p class="eyebrow">Step 1</p>
            <h3>Ingest a policy PDF</h3>
            <p>Send a file into the pipeline, split it into chunks, embed the text, and index it for retrieval.</p>
            <a class="button" href="/upload">Open Upload</a>
        </article>
        <article class="action-card">
            <p class="eyebrow">Step 2</p>
            <h3>Interrogate the corpus</h3>
            <p>Ask a question and force the model to answer only from retrieved enterprise context.</p>
            <a class="button" href="/chat">Open Chat</a>
        </article>
    </div>
    """
    side_html = """
    <div class="mini-card">
        <div class="eyebrow">Index</div>
        <strong class="mono">sample</strong>
        <p>Dense retrieval with cosine similarity and a 1024-dimensional vector shape.</p>
    </div>
    <div class="mini-card">
        <div class="eyebrow">Workflow</div>
        <span class="metric">2-step</span>
        <p>Upload once, then ask focused questions from the browser or the API.</p>
    </div>
    """
    return render_page(
        title="DocuMind Enterprise",
        eyebrow="Enterprise Search",
        heading="Ask documents like they are a live system.",
        description="A compact RAG console for policy PDFs with a browser layer on top of the FastAPI endpoints.",
        body_html=body_html,
        side_html=side_html,
    )

# Browser-friendly upload form
@app.get("/upload", response_class=HTMLResponse)
def upload_form():
    body_html = """
    <div class="content-grid">
        <article class="action-card">
            <p class="eyebrow">Upload</p>
            <h3>Send a PDF to the index</h3>
            <p>Use the browser form for manual ingestion. The API route remains available for terminal clients.</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <label class="field-label" for="file">PDF document</label>
                <input id="file" type="file" name="file" accept=".pdf" required />
                <button type="submit">Ingest Document</button>
            </form>
        </article>
        <article class="action-card">
            <p class="eyebrow">What happens next</p>
            <h3>Indexing pipeline</h3>
            <p>DocuMind extracts text from the PDF, chunks it, creates NVIDIA embeddings, then upserts the vectors into Pinecone.</p>
            <p class="note">Accepted flow: PDF only. Large files can take time because ingestion is synchronous in the current app.</p>
        </article>
    </div>
    """
    side_html = """
    <div class="mini-card">
        <div class="eyebrow">Chunking</div>
        <span class="metric">800 / 150</span>
        <p>Current split settings use 800-character chunks with 150-character overlap.</p>
    </div>
    <div class="mini-card">
        <div class="eyebrow">Target</div>
        <strong class="mono">/upload</strong>
        <p>Browser requests render HTML. API clients still receive structured JSON.</p>
    </div>
    """
    return render_page(
        title="Upload PDF",
        eyebrow="Document Intake",
        heading="Feed the index with source material.",
        description="Use this panel to ingest a PDF and prepare it for retrieval-driven answers.",
        body_html=body_html,
        side_html=side_html,
    )

# Upload PDF and ingest
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):

    file_location = f"temp_{file.filename}"

    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        result = ingest_pdf(file_location)
    except Exception as exc:
        if wants_html(request):
            error_html = """
            <div class="action-card">
                <p class="eyebrow">Upload Failed</p>
                <h3>The document was not ingested.</h3>
                <p class="note">Check the API keys, embedding model, and Pinecone settings, then try again.</p>
                <pre class="answer">{detail}</pre>
                <div class="footer-row">
                    <a class="button" href="/upload">Try Again</a>
                    <a class="button" href="/">Back Home</a>
                </div>
            </div>
            """.replace("{detail}", escape(str(exc)))
            return HTMLResponse(
                content=render_page(
                    title="Upload Failed",
                    eyebrow="Ingestion Error",
                    heading="The pipeline stopped before indexing.",
                    description="The browser view is showing the exact server-side error returned by the ingestion path.",
                    body_html=error_html,
                ),
                status_code=500,
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    if wants_html(request):
        success_html = f"""
        <div class="content-grid">
            <article class="action-card">
                <p class="eyebrow">Complete</p>
                <h3>Document indexed successfully.</h3>
                <p>Your PDF <span class="mono">{escape(file.filename)}</span> has been processed and pushed into Pinecone.</p>
                <div class="footer-row">
                    <a class="button" href="/chat">Ask a Question</a>
                    <a class="button" href="/upload">Ingest Another</a>
                </div>
            </article>
            <article class="action-card">
                <p class="eyebrow">Status</p>
                <h3>{escape(result)}</h3>
                <p class="note">This browser response is for the human workflow. API clients still receive JSON from the same endpoint.</p>
            </article>
        </div>
        """
        return HTMLResponse(
            content=render_page(
                title="Upload Complete",
                eyebrow="Ingestion Complete",
                heading="The document is ready for retrieval.",
                description="The index now contains fresh chunks from the uploaded PDF.",
                body_html=success_html,
            )
        )

    return JSONResponse(content={"status": result})

# Ask question
@app.get("/chat", response_class=HTMLResponse)
def chat_form(question: str | None = None):
    if not question:
        body_html = """
        <div class="content-grid">
            <article class="action-card">
                <p class="eyebrow">Query</p>
                <h3>Ask against indexed context</h3>
                <p>This form uses the retrieval pipeline first, then sends the retrieved context to the chat model.</p>
                <form action="/chat" method="get">
                    <label class="field-label" for="question">Question</label>
                    <input id="question" type="text" name="question" placeholder="What does the policy say about remote work?" required />
                    <button type="submit">Run Retrieval</button>
                </form>
            </article>
            <article class="action-card">
                <p class="eyebrow">Guardrail</p>
                <h3>Context only</h3>
                <p>The system prompt tells the model to answer only from retrieved passages and say when the answer is not present.</p>
            </article>
        </div>
        """
        side_html = """
        <div class="mini-card">
            <div class="eyebrow">Retrieval</div>
            <span class="metric">Top 5</span>
            <p>The current query flow retrieves five nearest matches from Pinecone before generation.</p>
        </div>
        <div class="mini-card">
            <div class="eyebrow">Target</div>
            <strong class="mono">/chat</strong>
            <p>Use the browser for manual questioning or POST /chat for programmatic access.</p>
        </div>
        """
        return render_page(
            title="Ask Question",
            eyebrow="Question Console",
            heading="Interrogate the indexed corpus.",
            description="Ask a direct question and inspect the model output inside the browser UI.",
            body_html=body_html,
            side_html=side_html,
        )

    try:
        context = retrieve_context(question)
        answer = ask_llm(context, question)
    except Exception as exc:
        error_html = """
        <div class="action-card">
            <p class="eyebrow">Query Failed</p>
            <h3>The answer could not be generated.</h3>
            <pre class="answer">{detail}</pre>
            <div class="footer-row">
                <a class="button" href="/chat">Try Another Question</a>
                <a class="button" href="/">Back Home</a>
            </div>
        </div>
        """.replace("{detail}", escape(str(exc)))
        return HTMLResponse(
            content=render_page(
                title="Chat Failed",
                eyebrow="Retrieval Error",
                heading="The question flow stopped before completion.",
                description="The UI is surfacing the backend error directly so you can debug the pipeline.",
                body_html=error_html,
            ),
            status_code=500,
        )

    answer_html = f"""
    <div class="action-card">
        <p class="eyebrow">Answer</p>
        <h3>Retrieved response</h3>
        <div class="question-chip">{escape(question)}</div>
        <pre class="answer">{escape(answer)}</pre>
        <div class="footer-row">
            <a class="button" href="/chat">Ask Another Question</a>
            <a class="button" href="/upload">Upload More Files</a>
        </div>
    </div>
    """
    side_html = f"""
    <div class="mini-card">
        <div class="eyebrow">Question</div>
        <p>{escape(question)}</p>
    </div>
    <div class="mini-card">
        <div class="eyebrow">Context Length</div>
        <span class="metric">{len(context.split())}</span>
        <p>Approximate word count in the concatenated retrieved context.</p>
    </div>
    """
    return render_page(
        title="Answer",
        eyebrow="Question Resolved",
        heading="The model answered from indexed context.",
        description="This view renders the retrieved answer directly from the browser workflow.",
        body_html=answer_html,
        side_html=side_html,
    )

@app.post("/chat")

def chat(question: str):

    try:
        context = retrieve_context(question)
        answer = ask_llm(context, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "question": question,
        "answer": answer
    }
