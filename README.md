# PyGuide AI

Python Programming Chatbot with Local-data RAG

## ตัวอย่าง Prompt ที่ส่งให้ AI

System prompt ใน `app.py`:

```text
You are PyGuide AI, a chatbot for Python programming only.
Rules:
- Answer using ONLY the supplied local dataset context.
- Do not invent facts outside the context.
- Answer in the same language as the user.
- Explain clearly and briefly.
- Include a small code example when useful.
- End with a Sources section using only URLs from the context.
```

User prompt ประกอบด้วย `LOCAL DATASET CONTEXT` ซึ่งมีชื่อบทเรียน URL และเนื้อหาที่ค้นพบ ตามด้วย `QUESTION` ของผู้ใช้ เช่น “list กับ tuple ใน Python ต่างกันอย่างไร”

## คำถามทดสอบ

`test_questions.csv` มี 12 ข้อพร้อมคำตอบที่คาดหวัง (`expected_answer`) โดย 10 ข้อมีคำตอบในเอกสาร และ 2 ข้อเกี่ยวกับ Qiskit ไม่มีคำตอบในชุดข้อมูล (`answer_in_documents=false`) ระบบควรแจ้งว่าข้อมูลไม่เพียงพอ คำตอบใน CSV เป็นเกณฑ์ตรวจ ไม่ใช่ผลการทดสอบที่บันทึกจากแอป

## แนวคิด
ระบบนี้เป็น Chatbot ที่ตอบเฉพาะหัวข้อ Python Programming โดยมีลำดับดังนี้

1. ตรวจว่าคำถามเกี่ยวกับ Python
2. อ่านข้อมูลที่ดาวน์โหลดไว้จาก `data/rag.npz`
3. ค้น chunks ที่เกี่ยวข้องด้วย embeddings และ FAISS
4. ให้ Gemini ตอบจากข้อมูลที่ค้นพบ พร้อม URL ของแหล่งข้อมูลเดิม
5. ถ้าข้อมูลไม่เพียงพอ แจ้งผู้ใช้ให้ปรับคำถามหรือเพิ่มข้อมูล

## วิธีติดตั้ง

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

ติดตั้ง package:

```bash
pip install -r requirements.txt
```

## จุดใส่ Google AI Studio API Key

สร้างไฟล์:

```text
.streamlit/secrets.toml
```

แล้วใส่:

```toml
GEMINI_API_KEY = "YOUR_GOOGLE_AI_STUDIO_API_KEY"
```

ดูตัวอย่างได้จากไฟล์:

```text
.streamlit/secrets.toml.example
```

ห้ามอัปโหลด API Key จริงขึ้น GitHub

## รันโปรแกรม

```bash
streamlit run app.py
```

## ตัวอย่างคำถาม

- for loop ใน Python ใช้ยังไง
- list กับ tuple ต่างกันยังไง
- Python exception handling ทำอย่างไร
- class ใน Python คืออะไร
- append() ใช้ยังไง

ถ้าถามเรื่องอื่น เช่น Java หรืออากาศ ระบบจะปฏิเสธ เพราะ Chatbot ถูกจำกัดไว้เฉพาะ Python

## Google AI Studio

สร้าง API key ได้ที่ https://aistudio.google.com/apikey แล้วคัดลอก .streamlit/secrets.toml.example เป็น .streamlit/secrets.toml และใส่ key ของคุณ

ใช้ Google Gen AI SDK (google-genai) โดยอ่าน GEMINI_API_KEY จาก Streamlit Secrets ผ่าน `st.secrets["GEMINI_API_KEY"]`

## API Key บน Streamlit Community Cloud

ตั้งค่าใน Secrets ของแอปบน Streamlit Community Cloud ด้วยรูปแบบ TOML:

```toml
GEMINI_API_KEY = "YOUR_GOOGLE_AI_STUDIO_API_KEY"
```

**ห้ามอัปโหลด API Key จริงขึ้น GitHub** สำหรับรันในเครื่องให้ใช้ `.streamlit/secrets.toml` ซึ่ง Git ignore แล้ว ไฟล์ `.streamlit/secrets.toml.example` ต้องมีเฉพาะค่าตัวอย่างเท่านั้น

## ข้อมูลในเครื่อง

แอปอ่านบทเรียน Python จาก W3Schools และ embeddings จาก `data/rag.npz` โดยไม่ดึงหน้าเว็บขณะตอบคำถามและไม่มีวันหมดอายุ URL ใช้ระบุแหล่งที่มาของบทเรียน เก็บไฟล์นี้ไว้เมื่อย้ายหรือ deploy โปรเจกต์ จำนวนเอกสารและ chunks อยู่ใน `data/dataset_info.json`

แอปดาวน์โหลดโมเดล embedding จาก Hugging Face เมื่อยังไม่มีใน cache เพื่อรองรับ Streamlit Community Cloud การโหลดครั้งแรกอาจใช้เวลา ส่วนข้อมูลบทเรียนยังอ่านจาก `data/rag.npz` และ Gemini ใช้อินเทอร์เน็ตเพื่อเรียบเรียงคำตอบจากข้อมูลที่ค้นพบ หากข้อมูลไม่เพียงพอจะแจ้งผู้ใช้โดยไม่ตอบจากความรู้ทั่วไปของ AI

ดาวน์โหลดหรืออัปเดตบทเรียนจาก https://www.w3schools.com/python/ ด้วยคำสั่ง:

```powershell
.\.venv\Scripts\python.exe download_data.py
```

สคริปต์สร้าง `documents.json/csv`, `chunks.json/csv`, `dataset_info.json` และ `rag.npz` ใน `data` เมื่อดาวน์โหลดและสร้าง embeddings สำเร็จครบทุกหน้าแล้ว ระบบใช้ `rag.npz` นี้โดยตรง หลังอัปเดตให้รีสตาร์ต Streamlit เพื่อโหลดชุดข้อมูลใหม่
