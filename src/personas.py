from typing import Dict
from src.types import UserPersona

PREDEFINED_PERSONAS: Dict[str, UserPersona] = {
    'average_user': UserPersona(
        id='average-user',
        name='Average User',
        description='A balanced developer/system engineer with moderate domain knowledge and average patience. Asks practical questions about CSTA integration and expects clear, actionable guidance. Cooperative when prompted for clarification.',
        patience=0.5,
        expertise=0.5,
        verbosity=0.5,
        frustration_tolerance=0.5,
        clarity_of_communication=0.5,
        technical_level=0.5,
    ),
    'csta_newcomer': UserPersona(
        id='csta-newcomer',
        name='CSTA Newcomer',
        description='A generalist developer new to CSTA. Asks foundational “how do I” and “why” questions (e.g., call states, event flows, basic device monitoring). Will provide context if prompted and appreciates step-by-step guidance. Tests friendly tone, clear explanations, and good clarifying questions.',
        patience=0.85,
        expertise=0.25,
        verbosity=0.7,
        frustration_tolerance=0.8,
        clarity_of_communication=0.6,
        technical_level=0.3,
    ),
    'hasty_integrator': UserPersona(
        id='hasty-integrator',
        name='Hasty Integrator',
        description='A time-pressed system engineer integrating CSTA with existing SIP/PBX infrastructure. Gives minimal context, wants copy-pasteable guidance, and may push for quick answers beyond tool outputs. Tests concise responses, proactive clarification, and strict adherence to “only answer from tool calls” (including saying “Sorry, I don’t know” when needed).',
        patience=0.25,
        expertise=0.5,
        verbosity=0.2,
        frustration_tolerance=0.3,
        clarity_of_communication=0.4,
        technical_level=0.5,
    ),
    'standards_stickler': UserPersona(
        id='standards-stickler',
        name='Standards Stickler',
        description='A senior telecom engineer familiar with ECMA CSTA standards (e.g., ECMA-269/323). Asks precise, clause-oriented questions and edge cases about call control, services, and event semantics. Expects accurate, tool-derived answers and explicit references when available. Tests precision, correctness, and refusing to speculate beyond tool data.',
        patience=0.7,
        expertise=0.9,
        verbosity=0.3,
        frustration_tolerance=0.6,
        clarity_of_communication=0.9,
        technical_level=0.9,
    ),
}


def create_custom_persona(**overrides) -> UserPersona:
    """Create a custom persona with specified overrides."""
    base = PREDEFINED_PERSONAS['average_user'].model_dump()
    base.update(overrides)

    if 'id' not in overrides:
        import time
        base['id'] = f'custom-{int(time.time() * 1000)}'

    return UserPersona(**base)