# ================== STEP 1: LOAD MODULES ==================

import os
import re
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_pdf_viewer import pdf_viewer
from tavily import TavilyClient

# Try importing create_agent from langchain or langchain.agents
try:
    from langchain.agents import create_agent
except ImportError:
    try:
        from langchain.agents import initialize_agent
        create_agent = None
    except ImportError:
        create_agent = None


# ================== STEP 2: PAGE CONFIG & SESSION STATE ==================

st.set_page_config(
    page_title="AI Resume & Skill Gap Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "redrawn_html" not in st.session_state:
    st.session_state.redrawn_html = None
if "job_code" not in st.session_state:
    st.session_state.job_code = None
if "pdf_file_path" not in st.session_state:
    st.session_state.pdf_file_path = None


# ================== STEP 3: CUSTOM STYLING (GLASSMORPHISM DARK THEME) ==================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
    :root {
        --bg-dark:       #0A0F0D;
        --panel-bg:      #121A15;
        --panel-card:    #18231C;
        --panel-hover:   #1F2E24;
        --border-color:  #293B2F;
        --border-glow:   #34D399;
        --primary-amber: #F59E0B;
        --primary-emerald: #10B981;
        --accent-cyan:   #06B6D4;
        --accent-rose:   #F43F5E;
        --text-main:     #ECFDF5;
        --text-muted:    #9CA3AF;
        --text-sub:      #6EE7B7;
    }

    /* Base Styling */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--panel-bg);
        border-right: 1px solid var(--border-color);
    }
    
    .sidebar-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary-amber);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px dashed var(--border-color);
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-main) !important;
        font-weight: 700;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #34D399 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 0.95rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 1.5rem;
    }

    /* Custom Glass Card */
    .glass-card {
        background-color: var(--panel-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease-in-out;
    }
    
    .glass-card:hover {
        border-color: rgba(52, 211, 153, 0.4);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Pill Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-match {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-missing {
        background-color: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    .badge-add {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-remove {
        background-color: rgba(156, 163, 175, 0.15);
        color: #D1D5DB;
        border: 1px solid rgba(156, 163, 175, 0.3);
    }

    .badge-keyword {
        background-color: rgba(6, 182, 212, 0.15);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-emerald), #059669);
        color: #FFFFFF !important;
        border: none;
        border-radius: 6px;
        padding: 0.6em 1.4em;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease-in-out;
        width: 100%;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #34D399, var(--primary-emerald));
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        transform: translateY(-1px);
    }

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div,
    textarea {
        background-color: var(--panel-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif;
    }
    
    div[data-testid="stTextInput"] input:focus,
    textarea:focus {
        border-color: var(--primary-emerald) !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.8rem 1.2rem;
        border-radius: 6px 6px 0 0;
    }

    button[aria-selected="true"] {
        color: var(--primary-emerald) !important;
        border-bottom: 2px solid var(--primary-emerald) !important;
        background-color: rgba(16, 185, 129, 0.08) !important;
    }

    /* Progress bar */
    div[data-testid="stProgress"] div[role="progressbar"] {
        background-color: var(--panel-card) !important;
        border-radius: 10px;
        height: 14px;
    }
    div[data-testid="stProgress"] div[role="progressbar"] > div {
        background: linear-gradient(90deg, #F59E0B, #10B981, #34D399);
        border-radius: 10px;
    }

    /* Stat Box */
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .metric-high { color: #34D399; }
    .metric-med { color: #FBBF24; }
    .metric-low { color: #FB7185; }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown('<div class="main-title">⚡ AI Resume & Skill Gap Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Bridge your qualifications to target career opportunities with AI precision</div>', unsafe_allow_html=True)


# ================== STEP 4: SIDEBAR CONFIGURATION ==================

with st.sidebar:
    st.markdown('<div class="sidebar-header">🛠️ Configuration & Credentials</div>', unsafe_allow_html=True)
    
    GOOGLE_API_KEY = st.text_input(
        "Google Gemini API Key",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        type="password",
        help="Get your key from Google AI Studio (gemini-3.5-flash)"
    )
    
    TAVILY_API_KEY = st.text_input(
        "Tavily Search API Key (Optional)",
        value=os.environ.get("TAVILY_API_KEY", ""),
        type="password",
        help="Required for live web job search functionality"
    )

    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        st.success("✓ Gemini API Key Loaded", icon="✅")
    else:
        st.info("💡 Provide Google API Key to enable AI analysis", icon="🔑")

    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    
    st.divider()
    
    st.markdown('<div class="sidebar-header">⚙️ Model Settings</div>', unsafe_allow_html=True)
    
    selected_model_name = st.selectbox(
        "Gemini Model",
        options=["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    
    temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)


# ================== STEP 5: HELPERS & AGENT FUNCTIONS ==================

def get_llm(model_name=None, temp=temperature):
    if not GOOGLE_API_KEY:
        return None
    if model_name is None:
        model_name = selected_model_name
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        google_api_key=GOOGLE_API_KEY
    )

def search_latest_news_jobs(query):
    """Fetch latest job postings using Tavily Search API"""
    if not TAVILY_API_KEY:
        return "Tavily API Key not configured."
    client = TavilyClient(api_key=TAVILY_API_KEY)
    return client.search(query)

def get_agent_instance():
    llm = get_llm()
    if not llm:
        return None
    if create_agent:
        try:
            return create_agent(model=llm, tools=[search_latest_news_jobs])
        except Exception:
            return None
    return None

def save_uploaded_file(file):
    save_dir = "pdf_files"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

@st.cache_data(show_spinner=False)
def load_resume_text(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

def extract_agent_text(response):
    if isinstance(response, str):
        return response
    if isinstance(response, dict) and 'messages' in response:
        content = response['messages'][-1].content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get('type') == 'text':
                    return block['text']
    return str(response)

def generate_optimized_resume(llm, query):
    prompt = ChatPromptTemplate.from_template(
        """You are a professional resume writer and front-end developer.

Using the candidate's resume details and skill-gap analysis below, generate a complete,
single-file HTML resume page.

Requirements:
- Output ONLY raw HTML (starting with <!DOCTYPE html> or <html>). No markdown, no code fences, no commentary.
- Include inline <style> CSS for a modern, executive, ATS-friendly design.
- Use clean typography, dark/light contrast cards, clear section headers, and bullet points.
- Use only real experience, skills, and education found in the details below - never invent job titles, employers, dates, or accomplishments.
- No placeholder text like "Lorem ipsum" or "[Your Name]" - use candidate's actual details.

CANDIDATE DETAILS AND SKILL GAP ANALYSIS:
{query}
"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})

def fetch_job_recommendations(llm, profile, location):
    if TAVILY_API_KEY:
        try:
            query = f"Current job openings for {profile} in {location} hiring now"
            search_res = search_latest_news_jobs(query)
            prompt = ChatPromptTemplate.from_template(
                """You are a job search assistant. Based on these search results:
{search_res}

Format up to 8 real job listings for target role: {profile} in location: {location}.

Output ONLY raw HTML with clean CSS cards (no markdown, no commentary). Each card should contain:
- Job Title
- Company Name & Location
- Key Snippet / Salary if available
- An "Apply / View Details" link using actual URLs found in the search results.

If limited details are found, display clean formatted cards with available information.
"""
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"search_res": str(search_res), "profile": profile, "location": location})
        except Exception as e:
            st.warning(f"Live search notice: {e}")
    
    # Fallback prompt without web tool if key is absent or search fails
    prompt = ChatPromptTemplate.from_template(
        """Create a styled HTML card list of key industry roles, skill requirements, and search parameters for:
Role: {profile}
Location: {location}

Output ONLY clean raw HTML (no markdown) formatted as modern cards with estimated salary ranges and job search recommendations.
"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"profile": profile, "location": location})

def build_chain(llm):
    prompt = ChatPromptTemplate.from_template(
        """You are an expert career coach and technical recruiter.

Compare the RESUME below against the TARGET JOB below and produce a skill gap analysis.

RESUME:
{resume_text}

TARGET JOB:
{target_job}

Respond ONLY with valid JSON (no markdown fences, no extra text) in exactly this shape:

{{
  "match_score": <integer 0-100>,
  "matching_skills": ["skill1", "skill2", ...],
  "missing_skills": ["skill1", "skill2", ...],
  "skills_to_add": ["specific skill or experience to add, with a short reason", ...],
  "content_to_remove": ["specific bullet/section/keyword to remove or de-emphasize, with a short reason", ...],
  "keyword_suggestions": ["ATS keyword1", "ATS keyword2", ...],
  "summary": "2-3 sentence overall verdict and top priority action"
}}
"""
    )
    return prompt | llm | StrOutputParser()

def parse_json_response(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)

def create_markdown_report(result, target_job):
    score = result.get("match_score", 0)
    report = f"""# 📊 Resume Skill Gap Analysis Report

**Target Role:** {target_job}  
**Match Score:** {score}/100  

---

## 📌 Executive Summary
{result.get("summary", "N/A")}

---

## ✅ Matching Skills (On the Path)
"""
    for skill in result.get("matching_skills", []):
        report += f"- {skill}\n"
    
    report += "\n## ⚠️ Missing Skills (Skill Gaps)\n"
    for skill in result.get("missing_skills", []):
        report += f"- {skill}\n"
        
    report += "\n## 🚀 Recommended Additions & Actions\n"
    for item in result.get("skills_to_add", []):
        report += f"- {item}\n"

    report += "\n## ✂️ Content to De-emphasize or Remove\n"
    for item in result.get("content_to_remove", []):
        report += f"- {item}\n"

    report += "\n## 🏷️ Essential ATS Keywords\n"
    report += ", ".join(result.get("keyword_suggestions", []))
    
    return report


# ================== STEP 6: MAIN INPUT FORM ==================

with st.container():
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📄 1. Candidate Resume")
        uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
        if uploaded_file is not None:
            binary_data = uploaded_file.getvalue()
            with st.expander("🔍 Interactive PDF Previewer", expanded=True):
                pdf_viewer(input=binary_data, width=450, height=480)

    with col2:
        st.subheader("🎯 2. Target Job Role & Criteria")
        
        job_category = st.selectbox(
            'Target Role Category',
            ('Software Engineer / Developer', 'Data Analyst / Data Scientist', 'AI / ML Engineer', 'DevOps / Cloud Engineer', 'Product / Project Manager', 'Custom Role')
        )
        
        if job_category == 'Custom Role':
            target_job = st.text_area('Specify Custom Target Job Description or Title', height=100)
        else:
            target_job = st.text_input('Target Job Title', value=job_category)
            
        location_input = st.text_input("Preferred Location / Market", value="Remote / Global")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 Analyze Gap & Generate Resume Insights", type="primary")


# ================== STEP 7: RUN ANALYSIS LOGIC ==================

if analyze_clicked:
    if not GOOGLE_API_KEY:
        st.error("⚠️ Please enter your Google Gemini API key in the sidebar configuration.")
    elif not uploaded_file:
        st.warning("⚠️ Please upload a PDF resume file to proceed.")
    elif not target_job.strip():
        st.warning("⚠️ Please specify a target job title or role description.")
    else:
        llm = get_llm()
        if not llm:
            st.error("Failed to initialize Google Gemini LLM. Check your API Key.")
        else:
            with st.spinner("⏳ Processing resume PDF and extracting content..."):
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.pdf_file_path = file_path
                st.session_state.resume_text = load_resume_text(file_path)

            with st.spinner("🧠 Performing AI Skill Gap Analysis against Target Role..."):
                try:
                    chain = build_chain(llm)
                    raw_result = chain.invoke({
                        "resume_text": st.session_state.resume_text,
                        "target_job": target_job
                    })
                    st.session_state.analysis_result = parse_json_response(raw_result)
                except Exception as e:
                    st.error(f"Error during AI analysis: {e}")
                    st.session_state.analysis_result = None

            if st.session_state.analysis_result:
                with st.spinner("✍️ Crafting ATS-Optimized HTML Resume..."):
                    try:
                        query_data = json.dumps(st.session_state.analysis_result) + "\n\n" + st.session_state.resume_text
                        st.session_state.redrawn_html = generate_optimized_resume(llm, query_data)
                    except Exception as e:
                        st.warning(f"Could not generate HTML resume: {e}")

                with st.spinner("💼 Fetching Market Opportunities..."):
                    try:
                        st.session_state.job_code = fetch_job_recommendations(llm, target_job, location_input)
                    except Exception as e:
                        st.warning(f"Could not fetch job recommendations: {e}")

                st.toast("Analysis complete! Check out the results below.", icon="🎉")


# ================== STEP 8: RENDER RESULTS & TAB WORKSPACE ==================

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    score = result.get("match_score", 0)
    
    st.divider()

    # Create Tabs Workspace
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Skill Gap & Insights",
        "📝 AI Redrawn Resume",
        "💼 Job Listings & Opportunities",
        "📄 Resume Diagnostics & Data"
    ])

    # ---------------- TAB 1: INSIGHTS ----------------
    with tab1:
        col_score, col_summary = st.columns([1, 2], gap="large")
        
        with col_score:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Match Score")
            
            score_class = "metric-high" if score >= 75 else ("metric-med" if score >= 50 else "metric-low")
            
            st.markdown(f'<div class="metric-value {score_class}">{score}<span style="font-size:1.5rem">%</span></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(min(max(score, 0), 100) / 100)
            
            report_md = create_markdown_report(result, target_job)
            st.download_button(
                label="📥 Export Report (.md)",
                data=report_md,
                file_name="resume_skill_gap_analysis.md",
                mime="text/markdown",
                help="Download full analysis report in Markdown format"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_summary:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📌 Executive Summary")
            st.write(result.get("summary", "No summary provided."))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### ✅ Matching Qualifications")
            matching = result.get("matching_skills", [])
            if matching:
                pills_html = "".join([f'<span class="badge-pill badge-match">✓ {item}</span>' for item in matching])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.write("No direct matches identified.")
            
            st.markdown("---")
            st.markdown("### ⚡ Recommended Skills to Add")
            skills_add = result.get("skills_to_add", [])
            for item in skills_add:
                st.markdown(f"- {item}")
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Skill Gaps (Missing)")
            missing = result.get("missing_skills", [])
            if missing:
                pills_html = "".join([f'<span class="badge-pill badge-missing">✗ {item}</span>' for item in missing])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.write("No major missing skills noted.")
                
            st.markdown("---")
            st.markdown("### ✂️ Recommended Content to De-emphasize")
            remove_list = result.get("content_to_remove", [])
            for item in remove_list:
                st.markdown(f"- {item}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Critical ATS Keywords")
        keywords = result.get("keyword_suggestions", [])
        if keywords:
            kw_html = "".join([f'<span class="badge-pill badge-keyword">{kw}</span>' for kw in keywords])
            st.markdown(kw_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # ---------------- TAB 2: AI RESUME GENERATOR ----------------
    with tab2:
        st.markdown("### 📝 AI-Generated Tailored Resume")
        st.caption("Custom formatted single-file HTML resume optimized for your target position.")
        
        if st.session_state.redrawn_html:
            btn_col1, btn_col2 = st.columns([1, 3])
            with btn_col1:
                st.download_button(
                    label="📥 Download HTML Resume",
                    data=st.session_state.redrawn_html,
                    file_name="optimized_resume.html",
                    mime="text/html",
                    key="dl_html"
                )
            
            with st.expander("🔍 Toggle Raw HTML Code View"):
                st.code(st.session_state.redrawn_html, language="html")
                
            st.markdown("---")
            st.components.v1.html(st.session_state.redrawn_html, height=700, scrolling=True)
        else:
            st.info("HTML Resume generation in progress or not available.")


    # ---------------- TAB 3: JOB LISTINGS ----------------
    with tab3:
        st.markdown(f"### 💼 Recommended Roles & Market Opportunities")
        st.caption(f"Curated listings for **{target_job}** ({location_input})")
        
        if st.session_state.job_code:
            st.components.v1.html(st.session_state.job_code, height=600, scrolling=True)
        else:
            st.info("Job recommendations not generated.")


    # ---------------- TAB 4: DIAGNOSTICS & RAW DATA ----------------
    with tab4:
        st.markdown("### 📄 Extracted Resume Content & Metrics")
        
        if st.session_state.resume_text:
            text = st.session_state.resume_text
            words = len(text.split())
            chars = len(text)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Word Count", words)
            m2.metric("Character Count", chars)
            m3.metric("Estimated Reading Time", f"{max(1, words // 200)} min")
            
            st.divider()
            st.markdown("#### Raw Extracted PDF Text")
            st.text_area("Extracted Text", value=text, height=350, disabled=True)
