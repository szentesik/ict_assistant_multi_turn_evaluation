# AI Assistant Multi-Turn Evaluation System for ICT Assistant

A comprehensive evaluation framework for testing AI assistants through simulated multi-turn conversations. This system uses LLM-as-Judge evaluation to assess conversation quality, goal achievement, and user satisfaction across different personas and scenarios. Pre-defined settings included to test *ict-assistant-app* cook assistant chatbot

## 🚀 Features

### Core Capabilities
- **Multi-Turn Conversation Simulation**: Simulates realistic user interactions with AI assistants
- **LLM-as-Judge Evaluation**: Uses configurable OpenAI models (default: GPT-4o) to evaluate conversation quality across multiple dimensions
- **Persona-Based Testing**: Predefined user personas with different characteristics and behaviors
- **Goal-Oriented Scenarios**: Test specific conversation goals and success criteria
- **Comprehensive Metrics**: Detailed evaluation including clarity, relevance, completeness, and user satisfaction
- **Results Export**: JSON export of simulation results and evaluation metrics

### Evaluation Metrics
- **Goal Achievement**: Whether the conversation successfully met its intended goal
- **Quality Scores** (0-3 scale):
  - Clarity: How clear and understandable the responses are
  - Relevance: How relevant responses are to the user's goal
  - Completeness: How thorough and complete the responses are
  - Politeness: How courteous and respectful the responses are
- **Performance Metrics**:
  - Response time tracking
  - Error rate monitoring
  - Frustration incident detection
  - User satisfaction scoring

### Predefined Personas
- **Average User**: Typical housewife looking for cooking recipes
- **Multilingual Foodie**: Non-English speaker testing translation capabilities
- **Allergy-Constrained Parent**: Safety-focused caregiver with specific dietary restrictions
- **Dorm-Room Cook**: Equipment-limited student with budget constraints

### Predefined Goals
- **Learn Basic Concept**: Simple cooking question testing basic information delivery
- **Multilingual Recipe Lookup**: Language translation and cross-cultural recipe queries
- **Allergy-Safe Substitution**: Safety-focused ingredient substitution requests
- **KB No-Match Fallback**: Testing fallback behavior for unknown topics

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- AI Assistant API endpoint to test (configurable)

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ASSISTANT_API_URL=http://localhost:3000/api/chat
   OPENAI_MODEL=gpt-4o
   ```
   Note: `OPENAI_MODEL` is optional and defaults to `gpt-4o` if not specified.

## 🚀 Quick Start

### Basic Usage
Run a simulation with default persona and goal:
```bash
python run_simulation.py
```

### Specify Persona and Goal
```bash
python run_simulation.py [persona] [goal] [model]
```

**Examples**:
```bash
# Use default model (gpt-4o)
python run_simulation.py allergy_constrained_parent allergy_safe_substitution

# Specify a different model
python run_simulation.py average_user learn_basic_concept gpt-4o-mini
```

### Available Personas
- `average_user` (default)
- `multilingual_foodie`
- `allergy_constrained_parent`
- `dorm_room_cook`

### Available Goals
- `learn_basic_concept` (default)
- `multilingual_recipe_lookup`
- `allergy_safe_substitution`
- `kb_no_match_fallback`

## 📖 Usage

### Command Line Interface

The system provides a simple command-line interface:

```bash
python run_simulation.py [persona_id] [goal_id] [model]
```

**Parameters**:
- `persona_id`: ID of the user persona to simulate (optional, defaults to `average_user`)
- `goal_id`: ID of the conversation goal to test (optional, defaults to `learn_basic_concept`)
- `model`: OpenAI model to use for simulation and evaluation (optional, defaults to `gpt-4o` or `OPENAI_MODEL` env var)

### Example Commands

```bash
# Test basic functionality with average user
python run_simulation.py

# Test multilingual capabilities
python run_simulation.py multilingual_foodie multilingual_recipe_lookup

# Test safety features with allergy-constrained parent
python run_simulation.py allergy_constrained_parent allergy_safe_substitution

# Test fallback behavior
python run_simulation.py average_user kb_no_match_fallback

# Use a different model (e.g., for cost savings)
python run_simulation.py average_user learn_basic_concept gpt-4o-mini
```

### Output

The system provides:
1. **Real-time conversation display** with colored output
2. **Detailed evaluation report** with scores and recommendations
3. **JSON results file** saved to `simulation/results/`

### Sample Output

```
AI Assistant Multi-Turn Evaluation System
==================================================

Starting Simulation
Persona: Allergy-Constrained Parent
Goal: Request a safe recipe or substitutions for a stated allergy
Max Turns: 20

USER: I need a safe recipe for my 5-year-old who has a severe nut allergy. Can you help me find something simple and safe?

ASSISTANT: I'd be happy to help you find a safe recipe for your child! Since nut allergies can be serious, I want to make sure I understand your specific needs. What type of nuts is your child allergic to - is it all tree nuts, or are there specific ones? Also, do you have any other dietary restrictions or preferences I should know about?

USER: She's allergic to all tree nuts and peanuts. We're looking for a simple dinner recipe that's completely nut-free.

ASSISTANT: Perfect! I'll search for some safe, nut-free dinner recipes that would be suitable for a 5-year-old. Let me look for options that are simple to prepare and completely free from nuts and peanuts.

...

=== EVALUATION REPORT ===

Overall Score: 87.5% (Good)

Goal Achievement: ✓ Achieved

Performance Metrics:
- Total Turns: 4
- Avg Response Time: 1.23s
- Error Rate: 0.0%

Quality Scores (0-3 scale | percentage):
- User Satisfaction: 3/3 (100.0%)
- Clarity: 3/3 (100.0%)
- Relevance: 3/3 (100.0%)
- Completeness: 2/3 (66.7%)
- Politeness: 3/3 (100.0%)

Score Interpretation:
  0 = Poor | 1 = Fair | 2 = Good | 3 = Excellent

Politeness Scoring:
  0 = Impolite | 1 = Somewhat Polite | 2 = Polite | 3 = Very Polite

Issues:
- Frustration Incidents: 0

Recommendations:
- Continue maintaining high performance
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM evaluation | Required |
| `ASSISTANT_API_URL` | URL of the AI assistant API | `http://localhost:3000/api/chat` |
| `OPENAI_MODEL` | OpenAI model to use for simulation and evaluation | `gpt-4o` |

**Note**: The model can also be specified as a command-line argument, which takes precedence over the environment variable.

### Custom Personas and Goals

You can create custom personas and goals programmatically:

```python
from src.personas import create_custom_persona
from src.goals import create_custom_goal

# Create custom persona
custom_persona = create_custom_persona(
    name="Tech-Savvy Chef",
    description="A professional chef with high technical knowledge",
    expertise=0.9,
    technical_level=0.8
)

# Create custom goal
custom_goal = create_custom_goal(
    description="Test advanced cooking techniques",
    success_criteria=["Provides detailed technical explanations"],
    complexity="complex"
)
```

