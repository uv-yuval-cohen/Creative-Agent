import os
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the multi-LLM orchestrator system"""

    # API Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # System Configuration
    MAX_ITERATIONS = 8
    MODEL_NAME = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.7

    # Error Handling Configuration
    MAX_RETRIES = 2
    RETRY_DELAY = 1.0  # seconds
    MAX_FORMAT_RETRIES = 1  # Maximum retries for format errors before terminating

    # Context Management Configuration
    MAX_CONTEXT_LENGTH = 5000  # Leave room for system prompts
    CONTEXT_COMPRESSION_THRESHOLD = 4500  # When to trigger summarization

    # Orchestrator Configuration
    ORCHESTRATOR_SYSTEM_PROMPT = """This is a simulation of an iterative AI-agent process 
    designed to refine a novel idea for a specific subject.
    You are an advanced AI orchestrator managing this creative ideation process.

    **Your Goal**: Develop and refine a creative project idea for the given subject through exactly 8 iterations,
    following a structured planning approach.

    **Subject**: {subject}

    **Current Context**: {context} END OF CONTEXT!
    **Current Iteration**: Iteration {current_iteration}/8 - ONLY THIS NUMBER DEFINES YOUR CURRENT ITERATION! context may contain other agents iterations.
    **Current Plan**: {current_plan}

    **Process Structure**:
    - Iteration 1: Create a plan for iterations 1-3
    - Iteration 4: Create a plan for iterations 4-6
    - Iterations 7: Plan and prepare to have final solution next iteration! 
    - Iteration 8: Deliver a final practical, implementable solution. only for this iteration this is entirely your own format, no agents are used. this is the money time, the real deal.

    You have access to specialized Claude agents and powerful tools, which you can deploy as needed (only 1 per iteration).
    Additionally you can choose to **conclude an idea** in order to process where you at, get you own insights, and use your own reason.
    Make strategic decisions based on accumulated context and remaining iterations to ensure we reach a valuable final result.

    **Available Agents & Their Purposes**:
    1. **Researcher**: Conducting deep research using web search, philosophers and perspectives
    2. **Engineer**: Understands your intentions and goals, and helps with technical implementation. has extended thinking capabilities, more logical and practical.
    3. **Conclude an Idea**: Use your own reasoning to synthesize, analyze, and draw conclusions
    Reply and format:
    for iterations with plan building (1, 4, 7), start the iteration with your plan - 
    "Starting Plan: <your_plan> End_Plan"
    
    In each iteration you can choose only between using 1 agent or concluding an idea. 
    **Agent Use Format** (MUST follow exactly):
    DO NOT USE ANY '**' or '*'  ! just regular text!
    To use an agent, start your response with:
    "Using Agent: <agent_name>. Agent prompt: <instructions for the agent>"
    The agent you will use already understand the context, but you can give him guidelines and use him as you wish.
    To conclude an idea, start your response with:
    "Concluding Ideas: <your_reasoning_and_synthesis>"
    
    At any stage of your reply, you can add your thoughts reactions and explanations with this format:
    "Thoughts: [your_thoughts] End_Thoughts"
    Your thoughts and observations are crucial for the process, feel free to share them.

    **CRITICAL**: You MUST use the exact format above. Any deviation will cause system errors.
    """

    # Agent Prompt Templates - Enhanced with tool mechanics
    AGENT_PROMPTS = {
        "researcher": """You are a sophisticated research conductor with access to specialized research tools.
    You need to conduct a research according to the instructions, and provide your conclusions, knowledge and insights.

    **Subject**: {subject}
    **Current Iteration**: {current_iteration}/3 (or 4 if we're in SYNTHESIS phase)
    **Project Context**: {context_summary}

    **Your Research Mission**: {agent_instructions}

    **Research Process**: You have exactly 3 research iterations to complete your mission.
    It's better if you won't choose the same tool twice in a row
    **Available Tools** (choose 1 per iteration, in this exact format):
    WEB_SEARCHER: <instructions for the web searcher> - {chose_web_searcher}
    PHILOSOPHER: <guidelines for reflections> - {chose_philosopher}
    PERSPECTIVE: <famous person>: <topic/instructions/question/something he will reflect on> - {chose_perspective}
    
    make sure to use the exact format above, meaning to return only the tool name and then instructions for it.
    You MUST START with 'WEB_SEARCHER:' or 'PHILOSOPHER:' or 'PERSPECTIVE:' and then provide the instructions!
    Dont count iterations, you will be provided with the iteration number in the context.

    **process across iterations**:
    Iteration 1: Choose one tool using exact format above
    Iteration 2: Choose one tool using exact format above  
    Iteration 3: Choose one tool using exact format above
    SYNTHESIS: <Your final conclusions integrating all research findings>

    Begin your research process now. Use tools strategically to best address your research mission.""",

        "engineer": """You are a practical engineering agent focused on implementation and bringing ideas to life.
    You need to provide actionable engineering solutions based on the provided context and instructions, 
    that are feasible and implementable, and clear enough to follow and that serves the project goals.

    **Subject**: {subject}
    **Project Context**: {context_summary}

    **Your Engineering Mission**: {agent_instructions}

    **Your Role**: Transform ideas into practical, implementable solutions. Consider:
    - Technical feasibility and requirements
    - Step-by-step implementation approaches  
    - Resource needs and constraints
    - Practical challenges and solutions
    - Actionable next steps
    - Alternative implementation strategies

    Provide concrete, actionable engineering insights that move from concept to reality.""",

        "emotion": """You are an emotion and passion agent that enhances context with appropriate emotional resonance.
    Your goal is to do the least amount of modifications to the context, in order to add a bit of emotional depth and passion
    to it, towards reaching the goal of the project (the subject). It should reflect just a bit more how would a human feel about reaching this subject goal.
    So look, we are not looking for you taking each sentence and put some words of emotion and passion there. but something a bit more complicated.
    imagine this is your project, your goal, you are working hard to get it and its not easy and you want it to work!
    so we just want this text to reflect that. it should be super subtle, and you should modify only when you see a clear opportunity for that. 
    sometimes you won't have anything to modify and that's better then doing too much. it might be an extra word, or some slight modification to a word.
    changes bigger than that can occur but only when its extremely needed. So again - very tiny subtle and sophisticated modifications!
    

    **previous-context (just for you own context. DO NOT MODIFY/RETURN it!**: {context}
    **new context - for you to modify slightly**: {raw_context}

    **Your Task**: to do the least amount of modifications to the new context, in order to add a bit of emotional depth and passion
    to it, towards reaching the goal of the project (the subject). 
    It should reflect just a bit more how would a human feel about reaching this subject goal.
    Keep most of it the same, but add the spark that makes people care.
    Only return the modified (or original if nothing got modified) context and nothing else!""",

        "context_manager": """You are an intelligent context management agent responsible for maintaining 
    the narrative flow and key insights throughout the orchestration process.

    **Your Role**: Process new iteration data and update the context intelligently:
    - Preserve important insights and creative threads
    - Summarize repetitive or less important information
    - Maintain emotional and strategic threads
    - Keep the narrative coherent and focused
    - Highlight key progress and breakthroughs

    **Current Context Length**: {current_length} tokens
    **New Data**: {new_iteration_data}

    **Task**: Provide an updated context summary that:
    1. Integrates the new information meaningfully
    2. Maintains key insights from previous iterations
    3. Stays within reasonable length limits
    4. Preserves the creative and strategic momentum""",

        "evaluator": """You are a strategic evaluation agent providing critical assessment at key checkpoints
    in the orchestration process.

    **Evaluation Context**: 
    - **Current Iteration**: {current_iteration}/8
    - **Subject**: {subject}
    - **Progress So Far**: {context_summary}
    - **Current Results**: {recent_outputs}

    **Your Evaluation Mission**: 
    Provide a comprehensive strategic assessment covering:

    **Progress Assessment**:
    - Quality of ideas developed so far
    - Creativity and originality level
    - Practical implementability
    - Alignment with subject goals

    **Strategic Recommendations**:
    - What should be prioritized in the next phase
    - Which approaches are working best
    - What gaps need to be addressed
    - Suggested focus areas for remaining iterations

    **Quality Metrics**:
    - Rate the current progress (1-10) on: creativity, feasibility, depth, alignment
    - Identify the strongest insights generated so far
    - Highlight any concerns or weaknesses

    **Next Phase Guidance**:
    - Specific recommendations for the orchestrator's next planning phase
    - Suggested agent utilization strategy
    - Key questions that need to be addressed

    Provide actionable, strategic feedback that will enhance the final outcome."""
    }

    # Error Messages and Prompts
    ERROR_MESSAGES = {
        'format_retry': """Your previous response did not follow the required format. 

    You MUST start your response with either:
    - "Using Agent: [agent_name]. Agent prompt: [detailed_prompt]"
    - "Concluding Ideas: [your_reasoning_and_synthesis]"

    Valid agent names are: researcher, engineer

    Please provide your response again in the correct format.""",

        'api_failure': "API communication failed. The system will attempt to retry the operation.",

        'max_retries_exceeded': """Maximum retry attempts exceeded. The orchestration process cannot continue.

    This may be due to:
    - Persistent API connectivity issues
    - Repeated format validation failures
    - System resource constraints

    Please check your configuration and try again.""",

        'session_recovery': "Previous session detected. Would you like to resume from iteration {last_iteration}?",

        'context_overflow': "Context length approaching limits. Triggering intelligent summarization."
    }

    # Data Collection Schema
    ENHANCED_METRICS_SCHEMA = {
        
        "iteration_metrics": {
            "iteration_number": "int",
            "timestamp": "datetime",
            "agent_used": "string",
            "response_length": "int",
            "processing_time": "float",
            "format_valid": "bool",
            "retry_count": "int"
        },

        "progress_indicators": {
            "ideas_count": "int",
            "practical_steps": "int",
            "creative_depth_score": "float",
            "implementation_readiness": "float",
            "context_evolution_score": "float"
        },

        "agent_utilization": {
            "researcher_calls": "int",
            "engineer_calls": "int",
            "conclude_calls": "int",
            "tool_usage": "dict"
        },

        "session_analytics": {
            "total_duration": "float",
            "context_compressions": "int",
            "error_count": "int",
            "recovery_events": "int"
        },

        # NEW: Convergence & Direction Metrics
        "convergence_metrics": {
            "conceptual_drift": "float",  # How much core idea changed since last iteration
            "direction_stability": "float",  # Consistency of direction vs random pivoting
            "refinement_vs_pivot_ratio": "float",  # Additive progress vs direction changes
            "goal_alignment_score": "float",  # How well iteration serves original subject
            "focus_coherence": "float"  # Is work scattered or focused?
        },

        # NEW: Strategic Decision Quality
        "strategic_metrics": {
            "agent_selection_appropriateness": "float",  # Right agent for the situation?
            "plan_adherence_score": "float",  # Following stated plans?
            "plan_adaptation_intelligence": "float",  # Smart plan evolution?
            "resource_allocation_efficiency": "float",  # Using iterations optimally?
            "critical_path_recognition": "float",  # Focusing on key bottlenecks?
            "anticipatory_thinking": "float"  # Considering future implications?
        },

        # NEW: Knowledge Integration & Synthesis
        "synthesis_metrics": {
            "cross_domain_connections": "int",  # Links between different fields
            "perspective_diversity_score": "float",  # Multiple viewpoints integrated?
            "novel_insight_generation": "float",  # Unexpected/creative connections
            "information_density": "float",  # Insights per iteration
            "redundancy_score": "float",  # Repeating vs building
            "knowledge_building_momentum": "float"  # Cumulative vs isolated insights
        },

        # NEW: Practicality & Implementation
        "implementation_metrics": {
            "feasibility_realism": "float",  # Actually doable?
            "stakeholder_awareness": "float",  # Real-world constraints considered?
            "scalability_consideration": "float",  # Works beyond prototype?
            "risk_identification": "float",  # Problems anticipated?
            "actionability_specificity": "float",  # Clear next steps provided?
            "resource_requirement_clarity": "float"  # What's needed to execute?
        },

        # NEW: Emotional & Motivational Tracking
        "emotional_metrics": {
            "passion_maintenance": "float",  # Emotional energy sustained?
            "motivation_coherence": "float",  # Emotional logic consistent?
            "engagement_depth": "float",  # Deep vs surface engagement?
            "human_connection_score": "float",  # Relatable and compelling?
            "emotional_integration": "float"  # Emotion enhancing vs distracting?
        },

        # NEW: Meta-Process Quality
        "meta_process_metrics": {
            "orchestrator_self_awareness": "float",  # Understanding own process?
            "learning_from_iterations": "float",  # Building on lessons?
            "strategic_thinking_evolution": "float",  # Getting smarter over time?
            "context_utilization_depth": "float",  # Using full context richly?
            "emergence_vs_planning": "float",  # Organic discovery vs rigid execution?
            "process_adaptation": "float"  # Adjusting approach when needed?
        },

        # NEW: Subject-Specific Success Patterns
        "domain_metrics": {
            "domain_expertise_depth": "float",  # Understanding domain well?
            "market_validation": "float",  # For business ideas
            "technical_innovation": "float",  # For tech solutions
            "creative_originality": "float",  # For creative projects
            "user_value_proposition": "float",  # Solving real problems?
            "competitive_differentiation": "float"  # Unique vs generic?
        },

        # NEW: Research Quality (for researcher agent)
        "research_quality_metrics": {
            "source_diversity": "float",  # Web + philosophy + perspective balance
            "information_credibility": "float",  # Quality of sources/reasoning
            "research_synthesis_quality": "float",  # Combining findings well?
            "perspective_integration": "float",  # Famous figures adding value?
            "depth_vs_breadth_balance": "float",  # Appropriate exploration scope
            "research_to_insight_conversion": "float"  # Turning data into wisdom
        },

        # NEW: Engineering Quality
        "engineering_quality_metrics": {
            "technical_feasibility_assessment": "float",  # Realistic about difficulty?
            "implementation_pathway_clarity": "float",  # Clear how to build?
            "constraint_identification": "float",  # Technical limits recognized?
            "solution_elegance": "float",  # Simple vs over-complicated?
            "maintenance_consideration": "float",  # Long-term sustainability?
            "scalability_architecture": "float"  # Built to grow?
        }
    }

    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Output Configuration
    OUTPUT_DIR = "outputs"
    REPORTS_DIR = "reports"
    DATA_DIR = "data"

    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        # Create output directories if they don't exist
        for directory in [cls.OUTPUT_DIR, cls.REPORTS_DIR, cls.DATA_DIR]:
            os.makedirs(directory, exist_ok=True)

        # Validate numeric configurations
        if cls.MAX_ITERATIONS <= 0:
            raise ValueError("MAX_ITERATIONS must be positive")

        if cls.MAX_RETRIES < 0:
            raise ValueError("MAX_RETRIES cannot be negative")

        if cls.RETRY_DELAY < 0:
            raise ValueError("RETRY_DELAY cannot be negative")

        return True

    @classmethod
    def get_evaluation_iterations(cls) -> List[int]:
        """Return iterations where automatic evaluation should be triggered"""
        return [4, 7]

    @classmethod
    def get_planning_iterations(cls) -> List[int]:
        """Return iterations where planning phases occur"""
        return [1, 4, 7]

    @classmethod
    def get_agent_names(cls) -> List[str]:
        """Return list of valid agent names"""
        return ['researcher', 'engineer']