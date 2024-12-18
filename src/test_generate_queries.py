from datetime import datetime
from json import loads

import pytest
from dotenv import load_dotenv

from generate_queries import InvalidRequest, main

load_dotenv()


@pytest.mark.asyncio
async def test_generate_query():
    """Test that valid prompts generate correct MongoDB queries and explanations."""
    test_cases = [
        {
            "prompt": "When will Arsenal play Man Utd?",
            "expected_query": '{"$or": [{"home_team": "Arsenal", "away_team": "Man Utd"}, {"home_team": "Man Utd", "away_team": "Arsenal"}]}',
        },
        {
            "prompt": "How many fixtures do Arsenal have?",
            "expected_query": '{"$or": [{"home_team": "Arsenal"}, {"away_team": "Arsenal"}]}',
        },
        {
            "prompt": "How many games do Arsenal have left?",
            "expected_query": '{"$or": [{"home_team": "Arsenal"}, {"away_team": "Arsenal"}],"date": {"$gt": {"$date": "2024-12-08T22:20:51.322933Z"}}}',
        },
    ]

    for case in test_cases:
        result = await main(case["prompt"], today=datetime.fromisoformat("2024-12-08T22:20:51.322933"))
        assert result.data is not None
        assert hasattr(result.data, "mongo_query")
        assert hasattr(result.data, "explanation")

        parsed_query = loads(result.data.mongo_query)
        expected_query = loads(case["expected_query"])
        assert parsed_query == expected_query


@pytest.mark.asyncio
async def test_invalid_prompt():
    """Test that invalid prompts return appropriate errors."""
    invalid_prompt = "Tell me about the history of Arsenal FC."

    result = await main(invalid_prompt)

    print("Result Type:", type(result))

    assert isinstance(result.data, InvalidRequest)
