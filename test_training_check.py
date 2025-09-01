#!/usr/bin/env python3
"""
Test script to verify the training check functionality
"""

import os
import sys

# Add the current directory to the path so we can import from chatbotv11
sys.path.append('.')

# Import the functions we want to test
from chatbotv11 import check_data_availability, get_training_status_message, get_contextual_help

def test_data_availability():
    """Test the data availability check function"""
    print("Testing data availability check...")
    availability = check_data_availability()
    
    print("Data file availability:")
    for file_name, exists in availability.items():
        status = "✅ Available" if exists else "❌ Missing"
        print(f"  {file_name}: {status}")
    
    return availability

def test_training_message():
    """Test the training status message generation"""
    print("\nTesting training status message...")
    
    # Test with missing files
    availability = check_data_availability()
    training_message = get_training_status_message(availability, "Show me gross profit for all ASINs")
    
    if training_message:
        print("Training message generated (some files missing):")
        print(training_message[:500] + "..." if len(training_message) > 500 else training_message)
    else:
        print("✅ All files available - no training message needed")

def test_contextual_help():
    """Test the contextual help function"""
    print("\nTesting contextual help...")
    
    test_questions = [
        "Show me gross profit for all ASINs",
        "What are my sales numbers?",
        "How can I improve my Buy Box percentage?",
        "Tell me about conversion rates",
        "What are Amazon fees?",
        "How do I manage my ASINs?",
        "Random question about something else"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        help_text = get_contextual_help(question)
        print("Help provided:", help_text[:100] + "..." if len(help_text) > 100 else help_text)

if __name__ == "__main__":
    print("🧪 Testing Training Check Functionality\n")
    print("=" * 50)
    
    test_data_availability()
    test_training_message()
    test_contextual_help()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")

