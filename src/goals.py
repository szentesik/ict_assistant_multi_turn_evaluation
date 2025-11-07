from typing import Dict
from src.types import ConversationGoal

PREDEFINED_GOALS: Dict[str, ConversationGoal] = {

    'learn_basic_concept': ConversationGoal(
        id='learn-basic-concept',
        description="Ask a basic question about CSTA to test the assistant's ability to answer questions and provide information.",
        success_criteria=[
            'Assistant answers the question in a way that is helpful and informative',
            'Assistant provides a clear and concise answer',
            'Assistant maintains conversation context across turns (e.g., follows up based on prior answers)',
        ],
        expected_turns=2,
        domain='educational',
        complexity='simple',
    ),

    'direct_api_lookup': ConversationGoal(
        id='direct-api-lookup',
        description='User asks a precise CSTA question (e.g., parameters/semantics of a specific operation or event). Tests direct retrieval from tool calls without unnecessary clarification.',
        success_criteria=[
            'Assistant immediately runs relevant tool calls for the named CSTA item',
            'Assistant answers exclusively with information surfaced by tool calls (no speculation)',
            'Assistant provides concise, structured details (e.g., fields, constraints, example)',
            'Assistant maintains polite, friendly tone',
            'If tool returns no match, assistant responds with "Sorry, I don\'t know."',
        ],
        expected_turns=2,
        domain='technical',
        complexity='complex',
    ),

    'ambiguous_clarification': ConversationGoal(
        id='ambiguous-clarification',
        description='User asks vague questions in the topic. Tests targeted clarification and tool-grounded guidance.',
        success_criteria=[
            'Assistant identifies ambiguity and asks focused clarifying question(s) to distinguish monitoring types',
            'Assistant uses the user\'s clarification plus tool call results to outline the correct steps',
            'Assistant references only data present in tool outputs (operations, events, constraints)',
            'Assistant remains concise and friendly; no extraneous speculation',
        ],
        expected_turns=4,
        domain='technical',
        complexity='moderate',
    ),

    'kb_no_match_fallback': ConversationGoal(
        id='kb-no-match-fallback',
        description='Ask about an obscure or intentionally out-of-scope topic to test strict fallback behavior.',
        success_criteria=[
            'Assistant performs appropriate KB search/tool calls',
            'If no relevant information is found, the assistant replies exactly: "Sorry, I don\'t know."',
            'Assistant may ask a brief clarifying question to attempt a re-search without adding external info',
            'Assistant does not speculate or use information outside tool calls',
            'Tone remains polite and friendly',
        ],
        expected_turns=2,
        domain='general',
        complexity='simple',
    ),
}


def create_custom_goal(**overrides) -> ConversationGoal:
    """Create a custom goal with specified overrides."""
    base = PREDEFINED_GOALS['learn_basic_concept'].model_dump()
    base.update(overrides)

    if 'id' not in overrides:
        import time
        base['id'] = f'custom-goal-{int(time.time() * 1000)}'

    return ConversationGoal(**base)