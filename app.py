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

def main_agent(agent,query):
    """this is the main agent or main agent orchestrate sub agents"""
    # Giving promt to create detailed prompt for code generation

    prompt = """You are Ai assistent and
    below given is a prompt , your
    rask is to give detailed prompt for this.
    you are a proffessional Resume Generator where user
    will give their personal info ,
    you have to create detail Resume for student or professional one,
    it must be with dynamic ui and ux and, with advance CSS profestional Desiging make sure
    to give output in html format only
    no markdown allowed"""

    response = agent.invoke({'messages':[{'role':'user','content':prompt}]})

    detail_prompt = response['messages'][-1].content[-1]['text']

    with open('prompt.txt','w') as f:
      f.write(detail_prompt)

    user_details = query

    final_prompt = prompt + detail_prompt + json.dumps(user_details)

    response = agent.invoke({'messages':[{'role':'user','content':final_prompt}]})

    code = response['messages'][-1].content[-1]['text']

    return code

def get_jobs(agent,Location = "Noida,Delhi",Profile = "DATA ANALYSIS,AI ENGINEER"):
  prompt = f"""Based on user given Job profile,
  fetch latest jobs or jobs apply article using naukri , linkindin,indeed, or all popular job apply platforms , show Results with JOB PROFILE NAME,
  LACATION,SALARY,COMAPNY NAME, SHOW jobs related to given {Location} and {Profile}, Out put must be in
  Professinal HTML , naukri theme cards with dynamic DEsign,
  show atleast top 10-20 results with direct apply link"""

  response = agent.invoke({'messages':[{'role':'user','content':prompt}]})

  code = response['messages'][-1].content[-1]['text']

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
                code = main_agent(agent,result)
                st.html(code , width="stretch" ,
                unsafe_allow_javascript=True)
                st.divider()  # to give horizontal div
                job_code = get_jobs(agent,"india",target_job)
                st.html(job_code , width="stretch" ,
                unsafe_allow_javascript=True)



