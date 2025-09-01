#!/usr/bin/env python3
"""
Test script to verify the logging functionality
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the path so we can import from chatbotv11
sys.path.append('.')

# Import the logging functions
from chatbotv11 import log_conversation, read_conversation_logs, get_log_statistics

def test_logging_functionality():
    """Test the logging functionality"""
    print("🧪 Testing Logging Functionality\n")
    print("=" * 50)
    
    # Test 1: Log some sample conversations
    print("1. Testing conversation logging...")
    
    sample_conversations = [
        {
            "user_question": "Show me gross profit for all ASINs",
            "ai_response": "Here's your gross profit analysis for all ASINs...",
            "session_id": "test_session_001",
            "data_availability": {"metadata_cache": True, "order_data": False},
            "error_occurred": False
        },
        {
            "user_question": "What are my sales numbers?",
            "ai_response": "I haven't been fully trained yet - some required data files are missing...",
            "session_id": "test_session_001",
            "data_availability": {"metadata_cache": False, "order_data": False},
            "error_occurred": True
        },
        {
            "user_question": "How can I improve my Buy Box percentage?",
            "ai_response": "Here are some strategies to improve your Buy Box percentage...",
            "session_id": "test_session_002",
            "data_availability": {"metadata_cache": True, "order_data": True},
            "error_occurred": False
        }
    ]
    
    for conv in sample_conversations:
        log_conversation(
            user_question=conv["user_question"],
            ai_response=conv["ai_response"],
            session_id=conv["session_id"],
            data_availability=conv["data_availability"],
            error_occurred=conv["error_occurred"]
        )
        print(f"   ✅ Logged: {conv['user_question'][:50]}...")
    
    # Test 2: Read and display logs
    print("\n2. Testing log reading...")
    logs = read_conversation_logs(limit=10)
    print(f"   📊 Found {len(logs)} log entries")
    
    if logs:
        print("   Recent log entries:")
        for i, log in enumerate(logs[-3:], 1):  # Show last 3
            print(f"   {i}. {log.get('timestamp', 'Unknown')} - {log.get('user_question', 'No question')[:50]}...")
    
    # Test 3: Get statistics
    print("\n3. Testing log statistics...")
    stats = get_log_statistics()
    
    if stats:
        print("   📈 Log Statistics:")
        print(f"   - Total conversations: {stats.get('total_conversations', 0)}")
        print(f"   - Unique sessions: {stats.get('unique_sessions', 0)}")
        print(f"   - Error rate: {stats.get('error_rate', 0)}%")
        print(f"   - Average question length: {stats.get('avg_question_length', 0)}")
        print(f"   - Average response length: {stats.get('avg_response_length', 0)}")
        print(f"   - Date range: {stats.get('date_range', 'No data')}")
    else:
        print("   ❌ No statistics available")
    
    # Test 4: Check log file
    print("\n4. Checking log file...")
    log_file_path = os.path.join("logs", "chatbot_logs.jsonl")
    if os.path.exists(log_file_path):
        file_size = os.path.getsize(log_file_path)
        print(f"   ✅ Log file exists: {log_file_path}")
        print(f"   📁 File size: {file_size} bytes")
        
        # Show a sample log entry
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                try:
                    sample_log = json.loads(lines[-1].strip())
                    print(f"   📝 Sample log entry keys: {list(sample_log.keys())}")
                except json.JSONDecodeError:
                    print("   ⚠️ Could not parse sample log entry")
    else:
        print(f"   ❌ Log file not found: {log_file_path}")
    
    print("\n" + "=" * 50)
    print("✅ Logging functionality test complete!")

if __name__ == "__main__":
    test_logging_functionality()

