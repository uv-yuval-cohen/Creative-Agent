#!/usr/bin/env python3
"""
Orchestrator execution with LLM and context management:
"""

import json
import tempfile
import os
from context_manager import ContextManager
from claude_agent import ClaudeAgentManager
from utils import generate_session_id, create_session_directory

def run_orchestrator_flow(session_dir):
    """Execute orchestrator flow with context management"""

    print("Executing LLM Orchestrator and Context Flow")
    print("=" * 60)

    subject = "I need a creative idea for a project in course about 3D representations and generations with machine learning"

    print(f"Subject: {subject}")
    print(f"Session Directory: {session_dir}")
    print(f"Files will be saved and persist after execution")

    # Initialize managers
    print("\n1. Initializing Managers...")

    context_manager = ContextManager(session_dir, subject)
    agent_manager = ClaudeAgentManager(session_dir, context_manager)
    print("Managers initialized successfully")

    for iteration in range(1, 9):
        print(f"\nIteration {iteration}")
        print("=" * 40)
        # Get current context for orchestrator
        print(f"Context Length: {len(context_manager.context)} characters")

        # Call orchestrator
        print(f"Calling LLM Orchestrator for iteration {iteration}...")

        orchestrator_response, success, metadata = agent_manager.create_orchestrator(
            current_iteration=iteration,
        )

        if not success:
            print(f"Orchestrator failed: {metadata.get('error', 'Unknown error')}")
            continue

        print("Orchestrator executed successfully")
        print(f"Processing time: {metadata.get('processing_time', 0):.2f} seconds")
        print(f"Response length: {len(orchestrator_response)} characters")
        print("Response preview:")
        print(f"   {orchestrator_response[:200]}...")

        # Get context before adding this iteration
        context_before = context_manager.context
        context_before_length = len(context_before)

        # Add to context manager (triggers: summarize → emotion → append)
        print("\nProcessing through Context Manager...")
        print("   Step 1: Summarizing orchestrator response...")
        print("   Step 2: Adding emotional enhancement...")
        print("   Step 3: Appending to context...")

        context_success = context_manager.add_iteration_result(
            iteration_num=iteration,
            response=orchestrator_response,
            agent_data=metadata,
            processing_time=metadata.get('processing_time', 0),
            retry_count=0
        )

        # Show results
        context_after = context_manager.context
        context_after_length = len(context_after)
        added_length = context_after_length - context_before_length

        print(f"Context processing status: {'Successful' if context_success else 'Failed'}")
        print(f"Context growth: {context_before_length} → {context_after_length} (+{added_length} characters)")

        # Show what was actually added
        if context_before:
            added_content = context_after[len(context_before):].strip()
            print(f"Added to context ({len(added_content)} characters):")
            print(f"   {added_content}")
        else:
            print(f"Initial context created ({len(context_after)} characters):")
            print(f"   {context_after}")

        print("\nCurrent Session Statistics:")
        print(f"   Total iterations: {len(context_manager.context_history)}")
        print(f"   Total context length: {len(context_manager.context)}")
        print(f"   Compression events: {context_manager.compression_count}")

    # Final summary
    print("\nFinal Context Summary")
    print("=" * 50)
    print(context_manager.context)

    # Analytics
    print("\nSession Analytics")
    print("=" * 30)
    analytics = context_manager.get_analytics_summary()
    if "error" not in analytics:
        session_overview = analytics.get('session_overview', {})
        agent_distribution = analytics.get('agent_distribution', {})
        quality_trends = analytics.get('quality_trends', {})

        print(f"Duration: {session_overview.get('session_duration', 0):.1f} seconds")
        print(f"Agent calls: {agent_distribution}")
        print(f"Quality metrics: {quality_trends}")
    else:
        print(f"Analytics error: {analytics['error']}")

if __name__ == "__main__":
    try:
        # Check if API key is available
        from config import Config

        if not Config.ANTHROPIC_API_KEY:
            print("Error: ANTHROPIC_API_KEY not found in environment")
            print("Please set your API key in .env file")
            exit(1)

        session_id = generate_session_id()
        session_dir = create_session_directory(session_id)
        run_orchestrator_flow(session_dir)
        print("\nExecution Completed Successfully")
        print(f"Session files saved in: {session_dir}")
        print(f"Review the files to inspect the JSON data")

    except Exception as e:
        print(f"\nExecution Failed: {e}")
        import traceback
        traceback.print_exc()