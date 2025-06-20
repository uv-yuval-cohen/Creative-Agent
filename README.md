# Multi-Agent Creative Ideation Orchestrator

An AI orchestrator system that uses specialized Claude agents to iteratively develop and refine creative project ideas through structured, intelligent collaboration.

## 🎯 Overview

This system creates a dynamic, multi-agent workflow where an orchestrator Claude agent manages specialized sub-agents (Researcher, Engineer, Emotion Agent) to develop creative solutions through 8 structured iterations. The system maintains context intelligence, tracks progress analytically, and produces implementable outcomes.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Orchestrator  │    │  Context Manager │    │ Agent Manager   │
│     (Claude)    │◄──►│   (Enhanced)     │◄──►│   (Multi-LLM)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Strategic Plans │    │ Intelligent      │    │ Specialized     │
│ Agent Selection │    │ Summarization    │    │ Agents:         │
│ Iteration Logic │    │EmotionEnhancement│    │ • Researcher    │
└─────────────────┘    │ Analytics        │    │ • Engineer      │
                       └──────────────────┘    │ • Emotion       │
                                               └─────────────────┘
```

## 🚀 Key Features

### Intelligent Orchestration
- **Strategic Planning**: Creates plans for iterations 1-3, 4-6, and 7-8
- **Dynamic Agent Selection**: Chooses appropriate specialists based on current needs
- **Self-Reflection**: Can conclude ideas using own reasoning and synthesis

### Specialized Agents
- **Researcher Agent**: 3-iteration research process with tools:
  - Web Search for current information
  - Philosophy mode for deep conceptual analysis  
  - Perspective mode channeling famous experts
- **Engineer Agent**: Technical implementation and feasibility analysis
- **Emotion Agent**: Subtle emotional enhancement to maintain human connection

### Advanced Context Management
- **Intelligent Summarization**: Compresses context while preserving key insights
- **Emotional Integration**: Adds appropriate passion and motivation
- **Analytics Tracking**: Comprehensive metrics on quality, progress, and strategy

### Robust Data Collection
- **Enhanced Metrics**: 60+ metrics across 12 categories including convergence, strategy, synthesis, implementation, and domain-specific success patterns
- **Session Analytics**: Performance tracking, error handling, and recovery
- **Iteration Snapshots**: Detailed context preservation at key points

## 📁 Project Structure

```
orchestrator/
├── main.py                 # Entry point and execution flow
├── config.py              # Configuration and prompts
├── claude_agent.py        # Multi-agent management system
├── context_manager.py     # Intelligent context handling
├── emotion_agent.py       # Emotional enhancement agent
├── utils.py               # Utilities and helpers
├── .env                   # Environment variables (API keys)
└── outputs/               # Generated session data
    └── session_YYYYMMDD_HHMMSS/
        ├── iterations/    # Individual iteration responses
        ├── context/       # Context history and snapshots
        ├── reports/       # Final reports and analytics
        ├── agents/        # Agent-specific outputs
        ├── errors/        # Error logs for debugging
        └── analytics/     # Performance metrics
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Anthropic API key with Claude access

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd orchestrator
   ```

2. **Install dependencies**
   ```bash
   pip install anthropic python-dotenv
   ```

3. **Configure environment**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
   ```

4. **Run the system**
   ```bash
   python main.py
   ```

## 🎯 Usage

### Basic Usage

```python
from main import run_orchestrator_flow
from utils import generate_session_id, create_session_directory

# Generate session
session_id = generate_session_id()
session_dir = create_session_directory(session_id)

# Run orchestration (8 iterations)
run_orchestrator_flow(session_dir)
```

### Customizing the Subject

Modify the subject in `main.py`:

```python
subject = "Your creative project goal here"
```

### Advanced Configuration

Customize behavior in `config.py`:
- `MAX_ITERATIONS`: Number of orchestration cycles (default: 8)
- `TEMPERATURE`: Creative vs. focused responses (default: 0.7)
- `MAX_CONTEXT_LENGTH`: Context management threshold
- Agent prompts and system behaviors

## 📊 Output & Analytics

### Session Files
- **Iterations**: Individual orchestrator responses in markdown format
- **Context History**: Complete conversation and summarization chain
- **Analytics**: Comprehensive metrics and quality scores
- **Agent Responses**: Detailed outputs from each specialized agent

### Analytics Dashboard
The system tracks 60+ metrics across 12 categories:

- **Convergence Metrics**: Goal alignment, direction stability, focus coherence
- **Strategic Metrics**: Agent selection quality, plan adherence, resource efficiency
- **Synthesis Metrics**: Cross-domain connections, novel insights, knowledge building
- **Implementation Metrics**: Feasibility, stakeholder awareness, actionability
- **Emotional Metrics**: Passion maintenance, motivation coherence
- **Meta-Process Metrics**: Self-awareness, learning, adaptation

## 🔧 Architecture Deep Dive

### Orchestrator Flow
1. **Iteration Planning** (1, 4, 7): Creates strategic plans for upcoming phases
2. **Agent Execution**: Selects and deploys appropriate specialist agent
3. **Context Integration**: Summarizes, enhances, and maintains narrative flow
4. **Progress Analysis**: Tracks quality metrics and strategic decisions
5. **Adaptive Planning**: Adjusts strategy based on progress and evaluation

### Context Management Pipeline
```
Raw Response → Summarization → Emotion Enhancement → Context Append
     ↓              ↓               ↓                    ↓
Length Check → LLM Analysis → Subtle Passion → Updated Context
```

### Research Agent Workflow
```
Iteration 1-3: Tool Selection (Web Search | Philosophy | Perspective)
     ↓
Tool Execution & Results Integration  
     ↓
Synthesis: Final conclusions and insights
```

## 🎛️ Configuration Options

### Model Settings
```python
MODEL_NAME = "claude-sonnet-4-20250514"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
```

### Context Management
```python
MAX_CONTEXT_LENGTH = 3500
CONTEXT_COMPRESSION_THRESHOLD = 3000
```

### Error Handling
```python
MAX_RETRIES = 2
RETRY_DELAY = 1.0
MAX_FORMAT_RETRIES = 1
```

## 📈 Performance & Quality Metrics

### Quality Indicators
- **Goal Alignment**: How well iterations serve the original subject
- **Creative Depth**: Level of originality and insight generation
- **Implementation Readiness**: Actionability and practical feasibility  
- **Strategic Coherence**: Quality of decision-making and planning

### Success Patterns
- **Domain Expertise**: Understanding of subject area
- **Market Validation**: Real-world applicability (for business ideas)
- **Technical Innovation**: Novel approaches and solutions
- **User Value**: Problem-solving relevance

## 🔄 Error Handling & Recovery

### Robust Design
- **Automatic Retries**: API failures with exponential backoff
- **Format Validation**: Response structure verification and correction
- **Session Recovery**: Resume from last completed iteration
- **Error Logging**: Comprehensive debugging information

### Recovery Process
```python
can_recover, last_iteration, session_state = check_session_recovery(session_dir)
if can_recover:
    # Resume from last_iteration + 1
```

## 🎨 Customization Examples

### Custom Subject Types
```python
# Business/Startup
subject = "Innovative sustainable food delivery service"

# Creative/Artistic  
subject = "Interactive art installation using AI and sensors"

# Technical/Software
subject = "Mobile app for real-time language learning"

# Social Impact
subject = "Community platform for local environmental action"
```

### Agent Behavior Modification
Customize agent prompts in `config.py` for different approaches:
- More technical focus vs. creative exploration
- Domain-specific expertise (business, art, technology)
- Different emotional tones and engagement levels

## 📝 Example Session Output

```
Session: session_20241220_143022
Subject: "Creative idea for 3D ML project"

Iteration 1: [PLAN] Research current 3D ML trends
Iteration 2: [RESEARCHER] Web search on latest techniques  
Iteration 3: [RESEARCHER] Philosophy on 3D representation
Iteration 4: [EVALUATION] Progress assessment + new plan
Iteration 5: [ENGINEER] Technical implementation strategy
Iteration 6: [RESEARCHER] Expert perspectives synthesis
Iteration 7: [EVALUATION] Final preparation + plan
Iteration 8: [CONCLUDE] Final concept and implementation roadmap

Final Output: Comprehensive project concept with technical details, 
creative elements, and step-by-step implementation guide.
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Getting Started
Run `python main.py` and watch specialized AI agents collaborate to develop your ideas into implementable solutions.
