# Handle SQLite for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import streamlit as st
import os
import re
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Set API keys
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY', '')

#--------------------------------#
#         CrewAI Agents          #
#--------------------------------#

def create_lead_qualification_crew(model_name="gpt-4o"):
    """Create CrewAI agents and crew for lead qualification"""
    
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    # Agent 1: Email Parser - Extracts information from emails
    email_parser = Agent(
        role='Email Information Extractor',
        goal='Extract all relevant contact and company information from email content',
        backstory='You are an expert at parsing emails and extracting structured data. '
                  'You can identify names, companies, job titles, and intent from email text.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Agent 2: Company Researcher - Researches company details
    company_researcher = Agent(
        role='Company Research Specialist',
        goal='Research and gather detailed information about companies including industry, size, and location',
        backstory='You are a business intelligence expert who can infer company details from domains '
                  'and email information. You understand business classifications and market segments.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Agent 3: Lead Scorer - Scores the lead based on criteria
    lead_scorer = Agent(
        role='Lead Qualification Specialist',
        goal='Score leads based on email quality, company fit, role seniority, and message intent',
        backstory='You are an experienced sales qualification expert who can assess lead quality. '
                  'You understand buyer personas, decision-making hierarchies, and sales readiness signals.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Agent 4: Recommendation Generator - Provides next steps
    recommendation_agent = Agent(
        role='Sales Strategy Advisor',
        goal='Provide actionable recommendations for engaging with leads based on their qualification score',
        backstory='You are a senior sales strategist who advises on lead engagement tactics. '
                  'You know when to prioritize, nurture, or disqualify leads.',
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    return {
        'email_parser': email_parser,
        'company_researcher': company_researcher,
        'lead_scorer': lead_scorer,
        'recommendation_agent': recommendation_agent,
        'llm': llm
    }


def run_email_lead_qualification(agents, sender_email, email_subject, email_content, target_config):
    """Run lead qualification workflow for email input"""
    
    # Task 1: Parse Email
    parse_task = Task(
        description=f"""
        Extract the following information from this email:
        - Sender Email: {sender_email}
        - Subject: {email_subject}
        - Content: {email_content}
        
        Extract and return:
        1. Sender's full name (if mentioned)
        2. Company name (if mentioned or inferred from email domain)
        3. Job title/designation (if mentioned)
        4. Email domain
        5. Main intent/purpose of the email
        
        Return in structured format.
        """,
        agent=agents['email_parser'],
        expected_output='Structured JSON with sender_name, company_name, designation, domain, and intent'
    )
    
    # Task 2: Research Company
    research_task = Task(
        description=f"""
        Based on the parsed email information and domain, research and infer:
        1. Company industry (classify into: Technology, Healthcare, Finance, Manufacturing, Retail, Education, Consulting, Real Estate, or Other)
        2. Estimated company size (Startup 1-50, SMB 51-500, Enterprise 500+)
        3. Geographic location (infer from domain or context)
        
        Use the email domain and any context clues from the email content.
        If domain is generic (gmail, yahoo, etc.), note it as "Personal Email - No Company Data"
        
        Return structured company information.
        """,
        agent=agents['company_researcher'],
        expected_output='JSON with industry, company_size, location, and domain_type',
        context=[parse_task]
    )
    
    # Task 3: Score the Lead
    score_task = Task(
        description=f"""
        Score this lead out of 100 points based on:
        
        Target Criteria:
        - Target Industries: {', '.join(target_config['industries'])}
        - Target Company Sizes: {', '.join(target_config['company_sizes'])}
        - Target Regions: {', '.join(target_config['regions'])}
        
        Scoring Rubric (100 points total):
        
        1. Email Domain Score (20 points):
           - Business email domain (not gmail/yahoo/hotmail): 20 points
           - Generic email but company mentioned: 10 points
           - Generic email only: 0 points
        
        2. Company Fit Score (40 points):
           - Industry matches target: 20 points
           - Company size matches target: 10 points
           - Location matches target region: 10 points
        
        3. Contact Role Score (20 points):
           - C-level, VP, Director (decision maker): 20 points
           - Manager, Lead, Specialist (influencer): 10 points
           - No clear role or junior role: 0 points
        
        4. Message Intent Score (20 points):
           - Specific interest/request with clear need: 20 points
           - General inquiry about services: 10 points
           - Vague or spam-like message: 0 points
        
        Return:
        - Total score (0-100)
        - Breakdown for each category with justification
        - Qualification status (Qualified 80+, Needs Review 50-79, Unqualified <50)
        """,
        agent=agents['lead_scorer'],
        expected_output='JSON with total_score, score_breakdown, and qualification_status',
        context=[parse_task, research_task]
    )
    
    # Task 4: Generate Recommendations
    recommendation_task = Task(
        description=f"""
        Based on the lead score and analysis, provide:
        1. Clear recommendation on next steps (Forward to Sales, Manual Review, or Disqualify)
        2. Specific reasons for the recommendation
        3. Suggested talking points or concerns to address
        4. Priority level (High, Medium, Low)
        
        Be specific and actionable.
        """,
        agent=agents['recommendation_agent'],
        expected_output='Detailed recommendations with next steps and priority',
        context=[parse_task, research_task, score_task]
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[
            agents['email_parser'],
            agents['company_researcher'],
            agents['lead_scorer'],
            agents['recommendation_agent']
        ],
        tasks=[parse_task, research_task, score_task, recommendation_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result


def run_form_lead_qualification(agents, name, company, designation, email, query, target_config):
    """Run lead qualification workflow for form input"""
    
    # Task 1: Structure Form Data
    structure_task = Task(
        description=f"""
        Structure the following form submission data:
        - Name: {name}
        - Company: {company}
        - Designation: {designation or 'Not provided'}
        - Email: {email}
        - Query: {query}
        
        Extract:
        1. Email domain
        2. Assess if email is business or personal
        3. Classify the intent/purpose from the query
        
        Return structured format.
        """,
        agent=agents['email_parser'],
        expected_output='Structured JSON with all form data and domain analysis'
    )
    
    # Task 2: Research Company
    research_task = Task(
        description=f"""
        Based on the company name "{company}" and email domain, infer:
        1. Company industry
        2. Estimated company size
        3. Geographic location
        
        Return structured company information.
        """,
        agent=agents['company_researcher'],
        expected_output='JSON with industry, company_size, and location',
        context=[structure_task]
    )
    
    # Task 3: Score the Lead
    score_task = Task(
        description=f"""
        Score this lead using the same 100-point rubric:
        
        Target Criteria:
        - Target Industries: {', '.join(target_config['industries'])}
        - Target Company Sizes: {', '.join(target_config['company_sizes'])}
        - Target Regions: {', '.join(target_config['regions'])}
        
        Apply the scoring rubric and return detailed breakdown.
        """,
        agent=agents['lead_scorer'],
        expected_output='JSON with total_score, score_breakdown, and qualification_status',
        context=[structure_task, research_task]
    )
    
    # Task 4: Generate Recommendations
    recommendation_task = Task(
        description=f"""
        Provide actionable recommendations based on the qualification score.
        """,
        agent=agents['recommendation_agent'],
        expected_output='Detailed recommendations with next steps',
        context=[structure_task, research_task, score_task]
    )
    
    crew = Crew(
        agents=[
            agents['email_parser'],
            agents['company_researcher'],
            agents['lead_scorer'],
            agents['recommendation_agent']
        ],
        tasks=[structure_task, research_task, score_task, recommendation_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result


#--------------------------------#
#         Streamlit App          #
#--------------------------------#

st.set_page_config(
    page_title="CrewAI Lead Qualification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as original)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        font-weight: 700;
    }
    .main-header .gradient-text {
        background: linear-gradient(90deg, #4b2be3 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4b2be3, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(75, 43, 227, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🤖 Model Selection")
        model = st.selectbox(
            "Choose OpenAI Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0,
            help="Select the OpenAI model for CrewAI agents"
        )
        
        st.subheader("🎯 Lead Qualification Setup")
        
        target_industries = st.multiselect(
            "Target Industries",
            ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Education", "Consulting", "Real Estate"],
            default=["Technology", "Healthcare"]
        )
        
        target_company_sizes = st.multiselect(
            "Company Sizes",
            ["Startup (1-50)", "SMB (51-500)", "Enterprise (500+)"],
            default=["SMB (51-500)", "Enterprise (500+)"]
        )
        
        target_regions = st.multiselect(
            "Target Regions",
            ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"],
            default=["North America", "Europe"]
        )
        
        return {
            "model": model,
            "target_config": {
                "industries": target_industries,
                "company_sizes": target_company_sizes,
                "regions": target_regions
            }
        }

# Main UI
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">🎯 <span class="gradient-text">CrewAI Lead Qualification</span></h1>', unsafe_allow_html=True)

config = render_sidebar()

if not os.environ.get("OPENAI_API_KEY"):
    st.error("⚠️ Please set your OpenAI API key", icon="🚨")
    st.stop()

st.markdown("---")
st.markdown("### 📊 Lead Analysis with Multi-Agent AI")

tab1, tab2 = st.tabs(["📧 Email Analysis", "📝 Form Submission"])

with tab1:
    st.markdown("**Analyze leads from email content using CrewAI agents**")
    
    with st.form("email_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sender_email = st.text_input("Sender Email *", placeholder="john@company.com")
            email_subject = st.text_input("Subject *", placeholder="Inquiry about services")
        
        with col2:
            email_content = st.text_area("Email Content *", height=150, placeholder="Enter email content...")
        
        email_submitted = st.form_submit_button("🚀 Analyze with CrewAI", type="primary")

with tab2:
    st.markdown("**Analyze leads from form submissions using CrewAI agents**")
    
    with st.form("form_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            form_name = st.text_input("Name *", placeholder="John Doe")
            form_company = st.text_input("Company *", placeholder="ABC Corp")
            form_designation = st.text_input("Job Title", placeholder="Manager")
        
        with col2:
            form_email = st.text_input("Email *", placeholder="john@company.com")
            form_query = st.text_area("Query *", height=120, placeholder="Message...")
        
        form_submitted = st.form_submit_button("🚀 Analyze with CrewAI", type="primary")

# Process with CrewAI
if email_submitted or form_submitted:
    if email_submitted:
        if not all([sender_email, email_subject, email_content]):
            st.error("❌ Please fill all required fields")
            st.stop()
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', sender_email):
            st.error("❌ Invalid email address")
            st.stop()
        input_method = "email"
    else:
        if not all([form_name, form_company, form_email, form_query]):
            st.error("❌ Please fill all required fields")
            st.stop()
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', form_email):
            st.error("❌ Invalid email address")
            st.stop()
        input_method = "form"
    
    with st.status("🤖 CrewAI agents are analyzing...", expanded=True) as status:
        try:
            # Create CrewAI agents
            status.update(label="🔧 Initializing CrewAI agents...")
            agents = create_lead_qualification_crew(config['model'])
            
            # Run analysis
            if input_method == "email":
                status.update(label="📧 CrewAI processing email...")
                result = run_email_lead_qualification(
                    agents,
                    sender_email,
                    email_subject,
                    email_content,
                    config['target_config']
                )
            else:
                status.update(label="📝 CrewAI processing form...")
                result = run_form_lead_qualification(
                    agents,
                    form_name,
                    form_company,
                    form_designation,
                    form_email,
                    form_query,
                    config['target_config']
                )
            
            status.update(label="✅ Analysis complete!", state="complete")
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 CrewAI Analysis Results")
            
            st.success("✅ Multi-agent analysis completed successfully!")
            
            st.markdown("**🤖 Agent Workflow:**")
            st.info("""
            1️⃣ Email Parser Agent - Extracted contact information
            2️⃣ Company Researcher Agent - Gathered company intelligence
            3️⃣ Lead Scorer Agent - Calculated qualification score
            4️⃣ Recommendation Agent - Generated next steps
            """)
            
            st.markdown("**📄 Full CrewAI Output:**")
            st.code(str(result), language="text")
            
        except Exception as e:
            status.update(label="❌ Error occurred", state="error")
            st.error(f"Error: {str(e)}")

st.divider()
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <p style='color: #4b2be3; font-weight: 600;'>
        Powered by <strong>CrewAI</strong> Multi-Agent Framework
    </p>
</div>
""", unsafe_allow_html=True)
