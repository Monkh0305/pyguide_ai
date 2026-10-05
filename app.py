from pathlib import Path
import streamlit as st


st.set_page_config(page_title="PyGuide AI", page_icon="🐍", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-3.5-flash"

TOP_K = 4
SIMILARITY_THRESHOLD = 0.32

# -----------------------------
# API KEY
# -----------------------------
def get_gemini_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None

# -----------------------------
# DOMAIN CHECK
# -----------------------------
def is_python_question(question: str) -> bool:
    q = question.lower().strip()
    python_keywords = [
        "python", "pip", "venv", "virtualenv", "list", "tuple", "dict",
        "dictionary", "set", "function", "def ", "class", "object", "oop",
        "if ", "elif", "else", "for ", "while", "loop", "range",
        "import", "module", "package", "exception", "try", "except",
        "file", "open(", "read(", "write(", "lambda", "comprehension",
        "string", "int", "float", "bool", "print(", "input(",
        "append", "extend", "numpy", "pandas", "streamlit", "flask",
        "django", "fastapi", "async", "await", "decorator", "generator"
    ]
    thai_keywords = [
        "ไพทอน", "ตัวแปร", "ลูป", "ฟังก์ชัน", "คลาส", "ลิสต์", "ทูเพิล",
        "ดิกชันนารี", "เซต", "เงื่อนไข", "โมดูล", "แพ็กเกจ", "ไฟล์",
        "ข้อผิดพลาด", "เอ็กเซปชัน", "เขียนโค้ด", "โค้ด python"
    ]
    return any(k in q for k in python_keywords + thai_keywords)

# -----------------------------
# LOCAL DATA
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_rag():
    from rag_store import load_local
    return load_local(Path(__file__).parent / 'data' / 'rag.npz', EMBED_MODEL)

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(question, embed_model, index, all_chunks):
    import numpy as np

    qv = embed_model.encode([question], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")

    scores, indices = index.search(qv, TOP_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(all_chunks[idx])
        item["score"] = float(score)
        results.append(item)

    return results

# -----------------------------
# LLM ANSWER: LOCAL CONTEXT
# -----------------------------
def answer_from_data(question, docs, client):
    from google.genai import types

    context = "\n\n".join(
        [
            f"[SOURCE {i}]\nTitle: {d['source']}\nURL: {d['url']}\nContent:\n{d['text']}"
            for i, d in enumerate(docs, start=1)
        ]
    )

    system = """
You are PyGuide AI, a chatbot for Python programming only.

Rules:
- Answer using ONLY the supplied local dataset context.
- Do not invent facts outside the context.
- Answer in the same language as the user.
- Explain clearly and briefly.
- Include a small code example when useful.
- End with a Sources section using only URLs from the context.
"""

    user = f"""
LOCAL DATASET CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned no text. Please try rephrasing your question.")
    return response.text

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.stApp { background: #f4f7f7; }
[data-testid="stMainBlockContainer"] { max-width: 1000px; padding-top: 1.5rem; padding-bottom: 6rem; }
[data-testid="stSidebar"] { background: #112c37; color: #edf7f5; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #edf7f5; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li::marker { color: #edf7f5; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a { color: #94e1c0; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li { line-height: 1.8; }
[data-testid="stSidebar"] button[kind="headerNoPadding"] { color: #edf7f5; }
[data-testid="stSidebar"] .stButton button { background: #21434c; color: #edf7f5; border-color: #365b63; }
.guide-hero { background: #112c37; border-radius: 24px; padding: 22px 28px;
    color: #f3faf7; margin: 0 0 16px; position: relative; overflow: hidden; }
.guide-hero .eyebrow { color: #94e1c0; font-size: 12px; letter-spacing: 3px; font-weight: 700; }
.guide-hero h1 { color: #f3faf7; font-size: clamp(24px, 3vw, 32px); line-height: 1.4; margin: 8px 0; }
.guide-hero p { color: #c0d6d5; max-width: 650px; line-height: 1.8; margin: 0; }
.guide-hero .hero-code { display: inline-block; color: #94e1c0; font-family: monospace;
    padding: 8px 14px; border: 1px solid #365b63; border-radius: 10px; margin-top: 22px; }
[data-testid="stChatMessage"] { background: white; border: 1px solid #e1e9e8; border-radius: 18px; margin-bottom: 14px; }
.stButton button { border-radius: 12px; min-height: 48px; }
.stButton button p { font-size: 15px; }
[data-testid="stChatMessage"] p { line-height: 1.8; }
.guide-compact { padding: 14px 22px; }
.guide-compact h1 { font-size: 24px; margin: 0; }
[data-testid="stChatInput"] { border-radius: 16px; }
@media (max-width: 640px) { .guide-hero { padding: 26px 22px; } }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ PyGuide")
    st.caption("พื้นที่เรียนรู้ Python ของคุณ")
    st.divider()
    if st.button("＋ เริ่มบทสนทนาใหม่", use_container_width=True):
        st.session_state.pop("messages", None)
    st.markdown("### วิธีใช้งาน")
    st.markdown("1. เลือกคำถามตัวอย่าง หรือพิมพ์ด้านล่าง\n2. กดส่งเพื่อรับคำอธิบาย\n3. เปิดแหล่งอ้างอิงใต้คำตอบเพื่ออ่านต่อ")
    st.caption("ส่งโค้ดที่ติดปัญหามาพร้อมคำถามได้")
    st.divider()
    st.caption("ตอบจากบทเรียน Python ของ W3Schools ที่ดาวน์โหลดไว้")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.messages:
    st.markdown('<div class="guide-hero guide-compact"><h1>🐍 PyGuide AI</h1><p>ถามเรื่อง Python ต่อได้ในช่องด้านล่าง</p></div>', unsafe_allow_html=True)
else:
    st.markdown("""
<div class="guide-hero">
  <div class="eyebrow">PYGUIDE AI</div>
  <h1>เรียนรู้ Python ผ่านคำถามของคุณ</h1>
  <p>เลือกคำถามด้านล่าง หรือพิมพ์คำถามและโค้ดที่อยากให้ช่วยอธิบาย</p>
</div>
""", unsafe_allow_html=True)

api_key = get_gemini_api_key()
if not api_key:
    st.error("ตั้งค่า GEMINI_API_KEY ใน Secrets ของ Streamlit Community Cloud (รันในเครื่องใช้ .streamlit/secrets.toml) เพื่อเริ่มใช้งาน")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

suggested_question = None
starter_questions = [
    "ตัวแปรใน Python สร้างและใช้งานอย่างไร?",
    "วันนี้อากาศเป็นอย่างไร?",
    "list กับ tuple ใน Python ต่างกันอย่างไร?",
    "เย็นนี้กินอะไรดี?",
    "จัดการข้อผิดพลาดด้วย try และ except อย่างไร?",
    "ช่วยแนะนำสถานที่ท่องเที่ยวหน่อย",
]


def show_starter_questions():
    selected = None
    columns = st.columns(2)
    for side, column in enumerate(columns):
        with column:
            st.markdown("##### 🐍 คำถาม Python" if side == 0 else "##### 🧪 คำถามทดสอบนอกเรื่อง")
            st.caption("กดเพื่อรับคำอธิบาย" if side == 0 else "กดเพื่อทดสอบการปฏิเสธคำถาม")
            for i in range(side, len(starter_questions), 2):
                prompt = starter_questions[i]
                if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                    selected = prompt
    return selected


if not st.session_state.messages:
    st.markdown("#### เริ่มต้นด้วยคำถามไหนดี?")
    suggested_question = show_starter_questions()
    st.caption("หรือพิมพ์คำถามและวางโค้ดในช่องด้านล่างได้เลย")
else:
    with st.expander("เลือกคำถามแนะนำ"):
        suggested_question = show_starter_questions()


def show_sources(sources):
    if sources:
        with st.expander("อ่านเอกสารประกอบคำตอบ"):
            seen = set()
            for item in sources:
                if item["url"] not in seen:
                    seen.add(item["url"])
                    st.markdown(f"- [{item['source']}]({item['url']})")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧭" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

question = st.chat_input("ถามเรื่อง Python หรือวางโค้ดที่อยากให้ช่วยดู…") or suggested_question
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🧭"):
        sources = []
        if not is_python_question(question):
            answer = "ไม่เข้าใจคำถาม กรุณาถามเกี่ยวกับการเขียนโปรแกรม Python เช่น การใช้ลูป ฟังก์ชัน หรือการแก้ข้อผิดพลาดในโค้ด"
        else:
            try:
                with st.spinner("กำลังค้นหา…"):
                    embed_model, index, all_chunks, loaded_docs, load_errors = build_rag()
                    results = retrieve(question, embed_model, index, all_chunks)
                if load_errors:
                    st.info("เอกสารบางหน้ายังโหลดไม่ได้ คำตอบนี้ใช้หน้าที่โหลดสำเร็จ")
                from google import genai
                with genai.Client(api_key=api_key, vertexai=False) as client:
                    with st.spinner("กำลังเรียบเรียงคำตอบ…"):
                        if results and results[0]["score"] >= SIMILARITY_THRESHOLD:
                            answer = answer_from_data(question, results, client)
                            sources = results
                        else:
                            answer = "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในชุดข้อมูลที่ดาวน์โหลดไว้ กรุณาปรับคำถามหรือเพิ่มข้อมูล"
            except Exception:
                st.error("ยังตอบไม่ได้ในตอนนี้ กรุณาตรวจสอบไฟล์ data/rag.npz โมเดลในเครื่อง การเชื่อมต่อ API key และโควตา แล้วลองอีกครั้ง")
                st.stop()
        st.markdown(answer)
        show_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
