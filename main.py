import streamlit as st
from io import BytesIO
import os
import tempfile
import zipfile
import textwrap
import streamlit as st
from datetime import datetime
import json
import os
import pandas as pd
import subprocess
from uuid import uuid4
import openai
import re
import json
from datetime import date

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key

# ---- Import the flow editor ----
from streamlit_flow import (
    StreamlitFlowNode,
    StreamlitFlowEdge,
    StreamlitFlowState,
    streamlit_flow,
)
st.set_page_config(page_title="CodeGenius", layout="wide")

# Directories
FRONTEND_DIR = "generated/frontend"
DATA_FILE = os.path.join(FRONTEND_DIR, "inventory_data.json")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Accent Color
ACCENT = "#00F5D4"

# Sidebar
st.sidebar.title("CodeGenius")

# Header function
def header(title):
    st.markdown(f"## {title}")

# UI Component Library
ui_library = {
    "login": ["Text Input: Username", "Text Input: Password", "Login Button"],
    "dashboard": ["Metric Cards", "Bar Chart", "Pie Chart"],
    "inventory": ["Item Table", "Search Box", "Add Item Form"],
    "form": ["Input Field", "Dropdown", "Submit Button"],
    "contact": ["Email Input", "Checkbox", "Date Picker", "Submit Button"],
    "profile": ["File Uploader", "Toggle Switch"],
    "report": ["Line Chart", "Table View"]
}

# DB Templates
db_templates = {
    "users": ["id", "username", "email", "password"],
    "inventory": ["id", "item_name", "quantity", "price"],
    "orders": ["order_id", "user_id", "item_id", "date"],
    "contacts": ["id", "email", "message", "date"]
}


MODULE = st.sidebar.radio("Choose Module", 
                          [
                                "Compose  (Spec → Code)",
                                "Weave    (Repo Integration)",
                                "Genesis AI  (App Generator)",
                                "AI Genie (RAG Builder)",
                                # "Evolve   (Continuous Refactor)",
                                "Skill-Gap Radar",
                                # "🧪 Synthetic Data Forge",
                                "Cost-Insights Assist",
                                # "🛡️ Compliance Packs"
                            ])


def detect_skill_gap(employee, task):
    prompt = (
        f"Employee: {employee}\n"
        f"Recent Task: {task}\n"
        "As a tech lead, analyze the above and return JSON with keys: "
        "'employee', 'recent_task', 'suggested_learning' (max 20 words), 'link' (1 high-quality resource). "
        "Learning should target the most relevant missing skill for the task."
    )
    client = openai.OpenAI(api_key=openai.api_key) 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert tech mentor and instructional designer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        temperature=0.2,
    )
    import json
    # Extract the first JSON object in the response
    import re
    match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
    return json.loads(match.group(0)) if match else {}

def generate_rag_project_code(nodes, edges, creds):
    # 1. Build a project description for LLM
    client = openai.OpenAI(api_key=openai.api_key) 
    prompt = (
        f"You are a senior ML/backend architect. The user is building a RAG (retrieval augmented generation) system."
        f"\nNodes: {nodes}\nEdges: {edges}\nCredentials/Options: {creds}\n"
        f"Generate a Python project with:\n"
        f"- FastAPI main.py covering all APIs implied by nodes/edges (health, upload, chat, vector index/query, etc)\n"
        f"- Modular code (split db.py, vector.py, storage.py as needed)\n"
        f"- Requirements.txt with needed libs\n"
        f"- README.md with setup/run steps\n"
        f"Each file should be a code block: '### filename.ext', then ```language ... ```."
        f"Use the creds/options inline (e.g., OpenAI key, S3 bucket), but never print secrets in code."
        f"Follow best practices, minimal but functional code, ready to run.\n"
        f"Respond with only code, do not add explanations."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You write production-grade backend code. Follow instructions strictly."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    return response.choices[0].message.content


def parse_project_files(llm_output):
    files = {}
    file_sections = re.findall(r"### ([\w\.\-/]+)\n```([\w]*)\n(.*?)```", llm_output, re.DOTALL)
    for fname, lang, content in file_sections:
        files[fname.strip()] = content.strip()
    return files

from io import BytesIO
import zipfile

def offer_zip_download(files, project_name):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, content in files.items():
            z.writestr(fname, content)
    buf.seek(0)
    st.download_button(
        "⬇️ Download Project",
        data=buf,
        file_name=f"{project_name}.zip",
        mime="application/zip"
    )

def generate_code_with_gpt4(feature_spec, stack):
    client = openai.OpenAI(api_key=openai.api_key) 
    prompt = f"""You are a world-class full-stack architect. 
Given this feature: "{feature_spec}", 
and target stack: {stack}, generate:
1. UI component code (JSX or TSX)
2. API endpoint (Python/JS/Java, as per stack)
3. DB schema & migration (SQL)
Follow best practices: accessibility, tests, security (OWASP).
Respond ONLY with code, in markdown, sections titled 'UI Component', 'API Endpoint', 'Database Schema & Migration'."""
    response = client.chat.completions.create(
        model="gpt-4o", # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are an expert software engineer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1400,
        temperature=0.1,
    )
    return response.choices[0].message.content

def parse_sections(text):
    # Simple parser to split markdown sections
    import re
    ui = re.search(r"UI Component\s*```[\w]*\n(.*?)```", text, re.DOTALL)
    api = re.search(r"API Endpoint\s*```[\w]*\n(.*?)```", text, re.DOTALL)
    db = re.search(r"Database Schema & Migration\s*```[\w]*\n(.*?)```", text, re.DOTALL)
    return (ui.group(1) if ui else ""), (api.group(1) if api else ""), (db.group(1) if db else "")


def extract_code_from_zip(zip_path):
    code_text = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_dir = tempfile.mkdtemp()
        zip_ref.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java', '.go', '.rb', '.php', '.cs', '.cpp', '.c', '.html', '.json', '.yaml', '.yml')):
                    try:
                        with open(os.path.join(root, file), 'r', encoding="utf-8", errors="ignore") as f:
                            code_text.append(f"\n# File: {os.path.join(root, file)}\n{f.read()}")
                    except Exception as e:
                        continue
    return "\n".join(code_text)

def analyze_code_with_gpt(code_string, sonar_summary=None):
    prompt = (
        "You are a senior software architect. Analyze the following codebase for:\n"
        "- Duplicate form flows, repeated schemas, reusable component opportunities\n"
        "- Violations of modern best practices\n"
        "- SonarQube or static analysis suggestions (if present)\n"
        "- For each, provide:\n"
        "    1. A summary of the issue\n"
        "    2. Suggested refactoring (show new reusable modules/components)\n"
        "    3. Ready-to-run bash/git commands for automation (if possible)\n"
        "    4. Ready-to-use PR description\n"
        "Codebase:\n"
        f"{code_string[:8000]}" # Limiting to 8K chars for context length
        + (f"\n\nSonarQube Analysis:\n{sonar_summary}" if sonar_summary else "")
    )
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a world-class code review and refactoring expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1800,
        temperature=0.1,
    )
    return response.choices[0].message.content

def gpt_generate_components(prompt):
    # system_prompt = (
    #     "You are a senior product engineer. "
    #     "Given the user’s app idea, output 3 sections, strictly markdown codeblocks:\n"
    #     "1. UI BLOCKS (as Python Streamlit pseudo-code components list)\n"
    #     "2. DATABASE SCHEMA (as JSON: table->fields)\n"
    #     "3. API BLUEPRINT (as JSON: route, method, request, response)\n"
    #     "Use best practices. Each section must start with ### and proper code fences."
    # )
    system_prompt = (
        "You are a senior product engineer. "
        "Given the user's app idea, generate ONLY self-contained Streamlit UI code in a single Python code block. "
        "Use placeholder data for DB/API if needed. Do not include explanations. "
        "The code should define a function `def app():` that renders the UI."
    )
    client = openai.OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1800,
        temperature=0.2,
    )
    code_match = re.search(r'```python\n(.*?)```', completion.choices[0].message.content, re.DOTALL)
    return code_match.group(1) if code_match else ""

def parse_gpt_sections(gpt_text):
    ui_match = re.search(r'### UI BLOCKS\s+```(?:python)?\s*(.*?)```', gpt_text, re.DOTALL)
    db_match = re.search(r'### DATABASE SCHEMA\s+```(?:json)?\s*(.*?)```', gpt_text, re.DOTALL)
    api_match = re.search(r'### API BLUEPRINT\s+```(?:json)?\s*(.*?)```', gpt_text, re.DOTALL)
    ui_blocks = ui_match.group(1).strip() if ui_match else ""
    db_schema = json.loads(db_match.group(1)) if db_match else {}
    api_blueprint = json.loads(api_match.group(1)) if api_match else {}
    return ui_blocks, db_schema, api_blueprint


# ------------------------------------------------------------
# AI GENIE – Drag-and-Drop RAG Builder
# ------------------------------------------------------------
from uuid import uuid4
def extract_and_summarize(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        # Simple scan: collect first N lines of key files for AI context
        code_samples = []
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(('.tf', '.yaml', '.yml', '.py', '.sql', '.sh', '.js', '.json')):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = ''.join([next(f) for _ in range(15)])  # first 15 lines
                        code_samples.append(f"### {file}\n{content}\n")
                    except Exception as e:
                        pass
        context = "\n".join(code_samples[:10])  # limit for prompt size
        return context


if MODULE.startswith("Cost"):
    st.header("Cost-Insights Assist")
    st.markdown("Upload your code/config (.zip) for AI-powered cost optimization suggestions.")

    uploaded_zip = st.file_uploader("Upload code/config zip:", type=["zip"])


    if uploaded_zip and st.button("Analyze & Suggest"):
        st.info("Extracting and analyzing uploaded files...")
        context = extract_and_summarize(uploaded_zip)
        if not context:
            st.warning("Could not extract usable code/config for analysis.")
        else:
            prompt = f"""You are a cloud cost optimization AI. Today is {date.today()}.Here is a sample of the user's codebase and configuration files:\n\n{context}\n\n
                        Analyze the infrastructure, data, and code patterns, and generate a JSON list of the top 3 most actionable cost-saving suggestions.
                        Each suggestion must include: date (YYYY-MM-DD), action, estimated_savings (USD/year), and a brief rationale.
                        Output only valid JSON (array of objects), nothing else."""
            with st.spinner("AI is analyzing your project for savings..."):
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert in cloud cost optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.2,
                )
                try:
                    insights = response.choices[0].message.content
                    st.success("Recent findings:")
                    st.markdown(insights)

                except Exception as e:
                    st.error(f"Error parsing AI response: {e}")
                    st.error("Error parsing AI response. Try uploading a more standard codebase.")

if MODULE.startswith("Skill"):
    # ---- Streamlit UI ----
    st.header("🧭 Skill-Gap Radar")
    st.markdown("AI agents detect dev skill-gaps and deliver tailored learning nudges.")

    employee = st.text_input("Employee Name", value="Asha")
    recent_task = st.text_input("Recent Task", value="Terraform IAM refactor")

    if st.button("Detect Skill Gap") and employee and recent_task:
        with st.spinner("Analyzing…"):
            result = detect_skill_gap(employee, recent_task)
        st.subheader("📈 Detection Result")
        st.json(result)
        if st.button("Send Learning Nudge"):
            # (Integrate with Slack/Email API in production)
            st.success(f"Learning suggestion sent to {employee}'s Slack/Email!")

if MODULE.startswith("AI"):

    # ---- Node Library ----
    NODE_TYPES = {
        "Teams": "source",
        "OpenAI": "model",
        "Claude": "model",
        "Cloud‑S3": "storage",
        "Azure‑Blob": "storage",
        "IndexDB": "vector",
        "Chatbot": "sink",
    }

    # ---- Always initialize session state for flow ----
    if "flow_state" not in st.session_state:
        st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])

    st.sidebar.subheader("Node Library")
    for label in NODE_TYPES.keys():
        if st.sidebar.button(f"Add {label}"):
            new_node = StreamlitFlowNode(
                id=f"node_{uuid4().hex[:8]}",
                pos=(20, 20),
                data={"content": label, "type": NODE_TYPES[label]},
                node_type="default",
                source_position="right",
                target_position="left",
            )
            st.session_state.flow_state.nodes.append(new_node)
            st.rerun()

    st.title("🧠 Drag-and-Drop Flow Editor Example")

    st.subheader("Flow Editor (drag nodes; connect via UI below)")

    # ---- Show the visual flow editor ----
    st.session_state.flow_state = streamlit_flow("rag_flow", st.session_state.flow_state)

    nodes = st.session_state.flow_state.nodes
    edges = st.session_state.flow_state.edges

    # ---- Edge Creation UI ----
    st.subheader("Connect Nodes (Create Edge)")
    if len(nodes) >= 2:
        options = [f"{n.data['content']} ({n.id})" for n in nodes]
        source_idx = st.selectbox("Source node", range(len(nodes)), format_func=lambda i: options[i], key="src_edge")
        target_idx = st.selectbox("Target node", range(len(nodes)), format_func=lambda i: options[i], key="tgt_edge")
        if st.button("Connect nodes"):
            source_id = nodes[source_idx].id
            target_id = nodes[target_idx].id
            if source_id == target_id:
                st.warning("Cannot connect a node to itself.")
            elif any(e.source == source_id and e.target == target_id for e in edges):
                st.info("Edge already exists.")
            else:
                st.session_state.flow_state.edges.append(StreamlitFlowEdge(None, source_id, target_id))
                st.success(f"Edge added: {source_id} → {target_id}")
                st.rerun()
    else:
        st.info("Add at least two nodes to create an edge.")

    # ---- Show nodes and edges for debug/inspection ----
    with st.expander("🔍 Inspect Flow State"):
        st.write("Nodes:")
        for n in nodes:
            st.json(n.data)
        st.write("Edges:")
        for e in edges:
            st.write(f"{e.source} → {e.target}")

    # ---- Credentials/options ----
    with st.expander("🔑 Credentials / Options"):
        project_name = st.text_input("Project name", "genesis_ai_genie")
        openai_key = st.text_input("OpenAI API Key", type="password")
        # add more as needed...

    # ---- Generate Project button: only when valid flow exists ----
    if st.button("⚡ Generate Project"):
        if not nodes or not edges:
            st.warning("⚠️ Please add nodes and connect them before generating.")
        else:
            st.success("Project ready to generate!")
            st.code(f"Nodes:\n{[n.data for n in nodes]}\n\nEdges:\n{[{'source': e.source, 'target': e.target} for e in edges]}", language="python")
            prompt = (
                f"You are a backend architect. "
                f"Given these nodes and edges, generate production-ready Python FastAPI code for a RAG app:\n\n"
                f"Nodes: {[n.data for n in nodes]}\n"
                f"Edges: {[{'source': e.source, 'target': e.target} for e in edges]}\n"
                f"Project name: {project_name}\n"
                f"(If Teams is connected to OpenAI, generate a Teams bot endpoint that calls OpenAI API, etc.)"
            )
            st.markdown("#### Example LLM Prompt")
            st.code(prompt, language="markdown")

    st.info(
        "Drag nodes onto the canvas (via sidebar), arrange them, and connect via the edge UI below. "
        "Inspect your flow, then generate your project when ready."
    )




# ---------- Compose ----------
elif MODULE.startswith("Compose"):
    # --- Streamlit UI ---
    st.header("Describe a feature — get a full stack")
    spec = st.text_area("Feature Specification", height=180, placeholder="e.g. Create employee onboarding form with AD sync & security audit …")
    stack = st.selectbox("Target Stack", ["React + FastAPI + PostgreSQL", "Next.js + NestJS + MySQL", "Vue + SpringBoot + Oracle"])

    if st.button("Generate") and spec:
        with st.spinner("Composing best-practice implementation …"):
            gpt_output = generate_code_with_gpt4(spec, stack)
            ui_code, api_code, db_code = parse_sections(gpt_output)
        st.success("Done! Review below →")
        st.subheader("UI Component")
        st.code(ui_code, language="javascript")
        st.subheader("API Endpoint")
        st.code(api_code, language="python")  # Or auto-switch language based on stack
        st.subheader("Database Schema & Migration")
        st.code(db_code, language="sql")
        st.caption("All artifacts include tests, accessibility checks, and OWASP guidelines 🎯")

# ---------- Weave ----------
elif MODULE.startswith("Weave"):
    header("Bring CodeGenius into an existing repo")
    repo_file = st.file_uploader("Upload a .zip of your repo or select a path (stub)")
    if st.button("Analyze for duplication") and repo_file:
        with st.spinner("Scanning and analyzing code with AI…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
                temp_zip.write(repo_file.read())
                temp_zip_path = temp_zip.name
            
            codebase_str = extract_code_from_zip(temp_zip_path)

            # (Optional) Integrate SonarQube CLI here, run static analysis, and get text summary
            sonar_summary = None  # Placeholder for future integration

            analysis_result = analyze_code_with_gpt(codebase_str, sonar_summary)

            st.info("AI code review and refactor suggestions:")
            st.markdown(analysis_result)
            st.success("PR-ready refactoring plan generated! 🚀")

# ---------- Evolve ----------
elif MODULE.startswith("Evolve"):
    header("Continuous Refactor & Runtime Guardrails")
    st.write("Recent suggestions:")
    st.json({
        "2025-05-29 09:14": "Extract common error-handler middleware across 5 services (120 LOC)",
        "2025-05-29 16:02": "Adjust DB index on employees.start_date (↑ query perf 45%)",
        "2025-05-30 08:55": "Add missing ARIA labels to 3 MODULEs for WCAG compliance"
    })
    if st.button("Apply all & open PR"):
        st.success("PR #128 created. CI pipeline green ✅")

# ---------- Genesis AI ----------
elif MODULE.startswith("Genesis"):
    st.title("🧠 Genesis.AI – App Generator (Smart Inventory)")
    user_prompt = st.text_input("🔍 Describe your app:", placeholder="e.g. Build an inventory dashboard with login")

    if user_prompt:
        with st.spinner("Generating components using AI …"):
            code_str = gpt_generate_components(user_prompt)
            # gpt_output = gpt_generate_components(user_prompt)
            # ui_blocks, db_schema, api_blueprint = parse_gpt_sections(gpt_output)


        st.subheader("🔧 Generated UI Code")

        st.subheader("👀 Preview")
        try:
            # Local scope for exec safety
            local_vars = {}
            exec_globals = {"st": st, "pd": pd, "json": json, "os": os, "datetime": datetime,"date": date, "BytesIO": BytesIO}
            exec(code_str, exec_globals, local_vars)
            if 'app' in local_vars:
                local_vars['app']()   # call the generated app()
            else:
                st.error("No 'app()' function found in generated code.")
        except Exception as e:
            st.error(f"Error in preview: {e}")



# Footer
st.sidebar.caption(f"Prototype v0.1 — generated {datetime.utcnow().strftime('%Y-%m-%d')} 🛠️")