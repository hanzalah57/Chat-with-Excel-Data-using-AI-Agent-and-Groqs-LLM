import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate 
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
import pandas as pd 
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

def main():
    load_dotenv()

    # Page Setup
    st.set_page_config(
        page_title="Chat with Excel Data using AI Agent & Groq's LLM", 
        page_icon="💬", 
        layout="wide"
    )
    
    st.title("💬📊 Chat with Excel Data using AI Agent & Groq's LLM")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False

    # Sidebar Configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        st.write("""
        **Steps to get started:**
        1. Enter your Groq API Key
        2. Select your preferred model and parameters
        3. Upload an Excel (.xlsx) or CSV file
        4. Start asking questions about your data
        """)

        st.header("🔑 API Key")
        api_key = st.text_input("Enter your Groq API key:", type="password")
        
        if not api_key:
            st.warning("⚠️ Please enter your Groq API Key to continue")
        elif api_key.startswith("gsk_") and len(api_key) > 10:
            st.success("✅ API Key validated!")
        else:
            st.error("❌ Invalid API Key format")

        st.header("⚙️ Model Settings")
        model = st.selectbox(
            "Choose LLM model",
            ["llama3-8b-8192", "gemma2-9b-It", "deepseek-r1-distill-llama-70b"],
            index=0
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        max_tokens = st.number_input("Max Tokens", min_value=100, max_value=8192, value=1000, step=100)

        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload your data file", 
            type=["csv", "xlsx"],
            help="Supported formats: CSV, Excel (.xlsx)"
        )

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main content area
    if not api_key:
        st.info("👈 Please enter a valid Groq API key in the sidebar to begin")
        return

    if uploaded_file is None:
        st.info("👈 Please upload a CSV or Excel file in the sidebar to start chatting with your data")
        return

    # Process uploaded file
    if uploaded_file is not None and not st.session_state.file_processed:
        process_file(uploaded_file, api_key, model, temperature, max_tokens)

    # Display file info if processed
    if st.session_state.df is not None:
        display_file_info()

    # Chat interface
    if st.session_state.agent is not None:
        display_chat_interface()

def process_file(uploaded_file, api_key, model, temperature, max_tokens):
    """Process the uploaded file and initialize the agent"""
    file_name = uploaded_file.name
    
    with st.spinner("📊 Processing your file..."):
        try:
            # Load the file
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                file_type = "CSV"
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
                file_type = "Excel"
            else:
                st.error("❌ Unsupported file type")
                return

            st.session_state.df = df
            
            # Initialize LLM
            llm = initialize_llm(api_key, model, temperature, max_tokens)
            if llm is None:
                return
                
            # Create agent
            agent = create_agent(llm, df)
            if agent is None:
                return
                
            st.session_state.agent = agent
            st.session_state.file_processed = True
            
            # Add welcome message
            welcome_msg = f"✅ {file_type} file '{file_name}' loaded successfully! I can now help you analyze your data. What would you like to know?"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": welcome_msg,
                "timestamp": datetime.now()
            })
            
            st.success(f"✅ {file_type} file processed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def initialize_llm(api_key, model, temperature, max_tokens):
    """Initialize the Groq LLM"""
    try:
        llm = ChatGroq(
            model=model,
            groq_api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            streaming=True
        )
        st.success("✅ LLM Initialized successfully")
        return llm
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {str(e)}")
        return None

def create_agent(llm, df):
    """Initialize the pandas dataframe agent"""
    try:
        prompt = """You are a data analysis assistant working with a pandas DataFrame.
                Answer user questions accurately and concisely using Python and pandas operations.
                Do not include extra commentary or internal thoughts.
                Only return the results of your analysis in a clean, readable format."""

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            max_iterations=15,
        )
        st.success("✅ Agent Initialized successfully")
        return agent
    except Exception as e:
        st.error(f"❌ Error creating agent: {str(e)}")
        return None

def display_file_info():
    """Display information about the loaded file"""
    if st.session_state.df is not None:
        with st.expander("📊 Dataset Overview", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(st.session_state.df))
            with col2:
                st.metric("Total Columns", len(st.session_state.df.columns))
            with col3:
                st.metric("Memory Usage", f"{st.session_state.df.memory_usage().sum() / 1024:.1f} KB")
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Data Type': st.session_state.df.dtypes.astype(str),
                'Non-Null Count': st.session_state.df.count(),
                'Null Count': st.session_state.df.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.subheader("Sample Data")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

def display_chat_interface():
    """Display the main chat interface"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "timestamp" in message:
                st.caption(f"_{message['timestamp'].strftime('%H:%M:%S')}_")

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"_{datetime.now().strftime('%H:%M:%S')}_")

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(), 
                expand_new_thoughts=True, 
                collapse_completed_thoughts=True
            )
            
            try:
                recent_context = []
                for msg in st.session_state.messages[-5:]:
                    recent_context.append(f"{msg['role']}: {msg['content']}")
                
                context_prompt = f"""
                Previous conversation context:
                {chr(10).join(recent_context)}
                
                Current question: {prompt}
                
                Please analyze the data to answer this question thoroughly.
                """
                
                with st.spinner("🤔 Analyzing your data..."):
                    response = st.session_state.agent.run(
                        context_prompt, 
                        callbacks=[st_cb]
                    )
                
                st.write("📊 Analysis Result:")
                st.write(response)
                
                timestamp = datetime.now()
                st.caption(f"_{timestamp.strftime('%H:%M:%S')}_")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": timestamp
                })
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error while analyzing your data: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": datetime.now()
                })

if __name__ == "__main__":
    main()
