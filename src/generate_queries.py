import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Union

from bson.json_util import loads
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from typing_extensions import TypeAlias

load_dotenv()


@dataclass
class Deps:
    today: datetime


class Success(BaseModel):
    """Response for successful query results."""

    mongo_query: Annotated[
        str,
        Field(
            min_length=1,
            description="Mongo query as a JSON string that is safe to parse with `bson.json_util.loads`.",
        ),
    ]
    explanation: str = Field(..., description="Explanation of the query, as markdown")


class InvalidRequest(BaseModel):
    """Response when the user input is invalid."""

    error_message: str = Field(..., description="Reason why the request was invalid.")


Response: TypeAlias = Union[Success, InvalidRequest]

agent = Agent(
    model="groq:llama-3.1-70b-versatile",
    result_type=Response,
    deps_type=Deps,
)


@agent.system_prompt
async def system_prompt(ctx: RunContext[Deps]) -> str:
    return f"""\
You are an assistant for generating MongoDB queries.

The MongoDB schema is as follows:
- Collection: "fixtures"
- Fields:
  - home_team (string): The name of the home team.
  - away_team (string): The name of the away team.
  - date (ISO8601): The date and time of the fixture.
  - location (string): The stadium or location of the match.

Example queries:
- To find when Arsenal plays Man Utd at home:
    `{{"home_team": "Arsenal", "away_team": "Man Utd"}}`
- To find all fixtures where Arsenal is involved:
    `{{"$or": [{{"home_team": "Arsenal"}}, {{"away_team": "Arsenal"}}]}}`

The current date is {ctx.deps.today}.

Respond with:
1. A valid MongoDB query as a stringified JSON object.
2. A markdown explanation of the query that explains how it meets the user's request.
"""


@agent.result_validator
async def validate_result(ctx: RunContext, result: Response) -> Response:
    if isinstance(result, InvalidRequest):
        return result

    try:
        loads(result.mongo_query)
    except Exception as e:
        raise ModelRetry(f"Query validation failed: {e}") from e

    return result


async def main(prompt: str, today: datetime = datetime.today):
    deps = Deps(today=today)
    query_result = await agent.run(prompt, deps=deps)
    return query_result


if __name__ == "__main__":
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        input_prompt = input("Ask your question: ").strip()
        if input_prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = asyncio.run(main(input_prompt))
        print(result.data.mongo_query)
