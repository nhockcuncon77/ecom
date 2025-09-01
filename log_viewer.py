#!/usr/bin/env python3
"""
Simple log viewer for chatbot conversation logs
"""

import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter

def view_logs():
    """View and analyze conversation logs"""
    log_file_path = os.path.join("logs", "chatbot_logs.jsonl")
    
    if not os.path.exists(log_file_path):
        print("❌ No log file found. Run the chatbot first to generate logs.")
        return
    
    # Read logs
    logs = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
    
    if not logs:
        print("❌ No valid log entries found.")
        return
    
    print(f"📊 Found {len(logs)} conversation logs\n")
    
    # Basic statistics
    print("=" * 60)
    print("📈 BASIC STATISTICS")
    print("=" * 60)
    
    total_conversations = len(logs)
    unique_sessions = len(set(log.get('session_id') for log in logs))
    errors = sum(1 for log in logs if log.get('error_occurred', False))
    error_rate = (errors / total_conversations * 100) if total_conversations > 0 else 0
    
    print(f"Total Conversations: {total_conversations}")
    print(f"Unique Sessions: {unique_sessions}")
    print(f"Errors: {errors} ({error_rate:.1f}%)")
    
    # Date range
    timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
    if timestamps:
        try:
            dates = [datetime.fromisoformat(ts) for ts in timestamps]
            date_range = f"{min(dates).strftime('%Y-%m-%d %H:%M')} to {max(dates).strftime('%Y-%m-%d %H:%M')}"
            print(f"Date Range: {date_range}")
        except:
            print("Date Range: Unknown")
    
    # Average lengths
    avg_question_length = sum(log.get('question_length', 0) for log in logs) / total_conversations if total_conversations > 0 else 0
    avg_response_length = sum(log.get('response_length', 0) for log in logs) / total_conversations if total_conversations > 0 else 0
    print(f"Average Question Length: {avg_question_length:.1f} characters")
    print(f"Average Response Length: {avg_response_length:.1f} characters")
    
    # Data availability analysis
    print("\n" + "=" * 60)
    print("📁 DATA AVAILABILITY ANALYSIS")
    print("=" * 60)
    
    data_availability_counts = Counter()
    for log in logs:
        if log.get('data_availability'):
            for file_name, available in log['data_availability'].items():
                if available:
                    data_availability_counts[f"{file_name}_available"] += 1
                else:
                    data_availability_counts[f"{file_name}_missing"] += 1
    
    for file_name in ['metadata_cache', 'order_data', 'asin_report', 'brand_report', 'gm_report', 'underper_report', 'brand_map']:
        available = data_availability_counts.get(f"{file_name}_available", 0)
        missing = data_availability_counts.get(f"{file_name}_missing", 0)
        total = available + missing
        if total > 0:
            availability_rate = (available / total * 100) if total > 0 else 0
            print(f"{file_name}: {available}/{total} available ({availability_rate:.1f}%)")
    
    # Recent conversations
    print("\n" + "=" * 60)
    print("💬 RECENT CONVERSATIONS (Last 5)")
    print("=" * 60)
    
    recent_logs = logs[-5:] if len(logs) >= 5 else logs
    for i, log in enumerate(reversed(recent_logs), 1):
        timestamp = log.get('timestamp', 'Unknown')
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp[:19] if timestamp else 'Unknown'
        
        question = log.get('user_question', 'No question')
        response = log.get('ai_response', 'No response')
        error = log.get('error_occurred', False)
        
        print(f"\n{i}. {formatted_time}")
        print(f"   Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"   A: {response[:100]}{'...' if len(response) > 100 else ''}")
        if error:
            print("   ⚠️  ERROR OCCURRED")
    
    # Error analysis
    if errors > 0:
        print("\n" + "=" * 60)
        print("⚠️  ERROR ANALYSIS")
        print("=" * 60)
        
        error_logs = [log for log in logs if log.get('error_occurred', False)]
        for i, log in enumerate(error_logs, 1):
            timestamp = log.get('timestamp', 'Unknown')
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp[:19] if timestamp else 'Unknown'
            
            question = log.get('user_question', 'No question')
            print(f"{i}. {formatted_time}: {question[:80]}{'...' if len(question) > 80 else ''}")
    
    # Export option
    print("\n" + "=" * 60)
    print("💾 EXPORT OPTIONS")
    print("=" * 60)
    
    # Create DataFrame for export
    df = pd.DataFrame(logs)
    if not df.empty:
        # Clean up the data for better CSV export
        export_df = df.copy()
        
        # Convert data_availability dict to string for CSV
        if 'data_availability' in export_df.columns:
            export_df['data_availability'] = export_df['data_availability'].apply(
                lambda x: str(x) if x else '{}'
            )
        
        # Save to CSV
        csv_filename = f"conversation_logs_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_df.to_csv(csv_filename, index=False)
        print(f"✅ Logs exported to: {csv_filename}")
        
        # Show DataFrame info
        print(f"\n📋 DataFrame Info:")
        print(f"   Shape: {export_df.shape}")
        print(f"   Columns: {list(export_df.columns)}")
    else:
        print("❌ No data to export")

if __name__ == "__main__":
    print("🔍 Chatbot Log Viewer")
    print("=" * 60)
    view_logs()

