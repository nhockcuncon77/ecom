import streamlit as st
import sys
import os

# Add the current directory to the path so we can import from chatbotv11
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the conversation history functions from chatbotv11
from chatbotv11 import store_assistant_response, capture_and_store_response

def test_conversation_history():
    """Test the conversation history functionality"""
    
    st.title("🧪 Conversation History Test")
    
    # Initialize conversation history if not exists
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Add clear conversation button
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.conversation_history = []
        st.rerun()
    
    # Display current conversation history
    st.subheader("📝 Current Conversation History")
    if st.session_state.conversation_history:
        for i, message in enumerate(st.session_state.conversation_history):
            st.write(f"**{message['role'].title()} {i+1}:** {message['content'][:100]}...")
    else:
        st.info("No conversation history yet.")
    
    # Test adding messages
    st.subheader("🧪 Test Adding Messages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add User Message"):
            st.session_state.conversation_history.append({
                "role": "user", 
                "content": f"Test user message {len(st.session_state.conversation_history)//2 + 1}"
            })
            st.rerun()
    
    with col2:
        if st.button("Add Assistant Message"):
            # Test the store_assistant_response function
            test_content = f"Test assistant response {len(st.session_state.conversation_history)//2 + 1}"
            store_assistant_response(test_content)
            st.rerun()
    
    # Test different response types
    st.subheader("🧪 Test Different Response Types")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Add Error Response"):
            store_assistant_response("This is a test error message", error=True)
            st.rerun()
    
    with col2:
        if st.button("Add Simple Response"):
            capture_and_store_response("This is a simple test response")
            st.rerun()
    
    with col3:
        if st.button("Add Dataframe Response"):
            import pandas as pd
            test_df = pd.DataFrame({
                'Column1': [1, 2, 3],
                'Column2': ['A', 'B', 'C']
            })
            store_assistant_response(
                "This is a test dataframe response",
                response_type="dataframe",
                dataframe=test_df,
                text="This is a test dataframe response"
            )
            st.rerun()
    
    # Display detailed conversation history
    st.subheader("📋 Detailed Conversation History")
    if st.session_state.conversation_history:
        for i, message in enumerate(st.session_state.conversation_history):
            with st.expander(f"Message {i+1} - {message['role'].title()}"):
                st.write(f"**Role:** {message['role']}")
                st.write(f"**Content:** {message['content']}")
                if 'response_type' in message:
                    st.write(f"**Response Type:** {message['response_type']}")
                if 'error' in message and message['error']:
                    st.write("**Error:** True")
                if 'dataframe' in message:
                    st.write("**Has Dataframe:** True")
                if 'source_file' in message:
                    st.write(f"**Source File:** {message['source_file']}")
    else:
        st.info("No conversation history to display.")

if __name__ == "__main__":
    test_conversation_history()

