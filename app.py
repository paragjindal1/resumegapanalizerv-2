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
from langchain.agents import create_agent
from tavily import TavilyClient


# ================== STEP 2: PAGE CONFIG ==================

st.set_page_config(
    page_title="Resume Skill Gap Analyzer",
    layout="wide"
)

st.sidebar.title("SET API CONFIG")
st.title("Resume Skill Gap Analyzer 🎯")
st.caption("Upload your resume, enter your target job, and get an AI-powered gap analysis.")


# ================== STEP 3: API KEY ==================

GOOGLE_API_KEY = st.sidebar.text_input(
    "GOOGLE_API_KEY",
    type="password"
)

TAVILY_API_KEY = st.sidebar.text_input(
    "TAVILY_API_KEY",
    type="password"
)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if GOOGLE_API_KEY:
    st.sidebar.success("Api key LOADED")
else:
    st.sidebar.info("Give Api Key")


# ================== STEP 4: INPUTS ==================

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
    # Read the binary stream from the uploader
        binary_data = uploaded_file.getvalue()
        pdf_viewer(input=binary_data, width=400 , height=500)


with col2:
    target_job = st.selectbox('Targeted jobs',('webdev','Software Enginner','data Analises','other'))
    if target_job == 'other':
        target_job = st.text_area('enter other jobs')



analyze_clicked = st.button("Analyze Gap", type="primary")


# ================== STEP 5: HELPERS ==================

model = ""

if GOOGLE_API_KEY :
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3
    )

def search_latest_news_jobs(query):
  """this function helps to fetch latest
  news or jibs related article using tavily"""

  client = TavilyClient(
      api_key = TAVILY_API_KEY
  )
  response = client.search(query)

  return response

agent = create_agent(
    model = model,
    tools = [search_latest_news_jobs]
)

def save_uploaded_file(file):
    save_dir = "pdf_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def extract_agent_text(response):
    """Safely pulls the final text block out of an agent response,
    since the last content block isn't always plain text (e.g. tool calls)."""
    content = response['messages'][-1].content
    if isinstance(content, str):
        return content
    for block in reversed(content):
        if isinstance(block, dict) and block.get('type') == 'text':
            return block['text']
    raise ValueError("No text content found in agent response")

def main_agent(agent,query):
    """this is the main agent or main agent orchestrate sub agents"""

    prompt = f"""You are a professional resume writer and front-end developer.

Using the candidate's resume details and skill-gap analysis below, generate a complete,
single-file HTML resume page.

Requirements:
- Output ONLY raw HTML (starting with <!DOCTYPE html> or <html>). No markdown, no code fences, no commentary.
- Include inline <style> CSS for a modern, professional, ATS-friendly design.
- Use only real experience, skills, and education found in the details below - never invent
  job titles, employers, dates, or accomplishments that weren't provided.
- No placeholder text like "Lorem ipsum" or "[Your Name]" - use the candidate's actual details.

CANDIDATE DETAILS AND SKILL GAP ANALYSIS:
{query}
"""

    response = agent.invoke({'messages':[{'role':'user','content':prompt}]})

    code = extract_agent_text(response)

    return code

def get_jobs(agent,Location = "Noida,Delhi",Profile = "DATA ANALYSIS,AI ENGINEER"):
  prompt = f"""You are a job search assistant with access to a web search tool.

Search for current, real job listings matching:
- Profile: {Profile}
- Location: {Location}

Use the search tool to find actual postings. Do not invent job listings, company names,
salaries, or links under any circumstances - if you're not certain a detail came from a
search result, leave it out rather than guessing.

Output ONLY raw HTML (no markdown, no commentary) with up to 10 job cards. Each card must include:
- Job title
- Company name
- Location
- Salary (only if explicitly stated in a source; omit the field otherwise)
- An "Apply" link using the EXACT URL returned by the search tool

Style as clean, professional cards with inline CSS (no external stylesheets).
If fewer than 10 real listings are found, show only what you found - never pad the list.
"""

  response = agent.invoke({'messages':[{'role':'user','content':prompt}]})

  code = extract_agent_text(response)

  return code


@st.cache_data(show_spinner=False)
def load_resume_text(file_path):
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)


def build_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert career coach and technical recruiter.

Compare the RESUME below against the TARGET JOB below and produce a skill
gap analysis.

RESUME:
{resume_text}

TARGET JOB:
{target_job}

Respond ONLY with valid JSON (no markdown fences, no extra text) in exactly
this shape:

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


# ================== STEP 6: RUN ANALYSIS ==================

if analyze_clicked:
    if not GOOGLE_API_KEY:
        st.warning("Please enter your Google API key in the sidebar.")
    elif not uploaded_file:
        st.warning("Please upload a resume PDF.")
    elif not target_job.strip():
        st.warning("Please enter a target job title or description.")
    else:
        with st.spinner("Reading resume..."):
            file_path = save_uploaded_file(uploaded_file)
            resume_text = load_resume_text(file_path)

        with st.spinner("Analyzing skill gap..."):
            chain = build_chain()
            try:
                raw_result = chain.invoke({
                    "resume_text": resume_text,
                    "target_job": target_job
                })
                result = parse_json_response(raw_result)
            except Exception as e:
                st.error(f"Something went wrong while analyzing: {e}")
                result = None

        if result:
            st.divider()

            score = result.get("match_score", 0)
            st.subheader("Match Score")
            st.progress(min(max(score, 0), 100) / 100)
            st.write(f"**{score}/100**")

            st.subheader("Summary")
            st.write(result.get("summary", ""))

            st.divider()

            colA, colB = st.columns(2)

            with colA:
                st.subheader("✅ Matching Skills")
                for item in result.get("matching_skills", []):
                    st.write(f"- {item}")

                st.subheader("➕ What to Add")
                for item in result.get("skills_to_add", []):
                    st.write(f"- {item}")

            with colB:
                st.subheader("❌ Missing Skills")
                for item in result.get("missing_skills", []):
                    st.write(f"- {item}")

                st.subheader("➖ What to Remove / De-emphasize")
                for item in result.get("content_to_remove", []):
                    st.write(f"- {item}")

            st.divider()
            st.subheader("🔑 ATS Keyword Suggestions")
            st.write(", ".join(result.get("keyword_suggestions", [])))

            with st.expander("View extracted resume text"):
                st.text(resume_text)

            with st.spinner("Agent Running"):
                code = main_agent(agent,json.dumps(result)+resume_text)
                st.html(code , width="stretch" ,
                unsafe_allow_javascript=True)
                st.divider()  # to give horizontal div
                job_code = get_jobs(agent,"india",target_job)
                st.html(job_code , width="stretch" ,
                unsafe_allow_javascript=True)
