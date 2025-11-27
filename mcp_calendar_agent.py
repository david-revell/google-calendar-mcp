"""
Attempt 4 — Calendar Agent (v1)
- True OpenAI Agents SDK agent
- Uses MCP Google Calendar server tools: list_events, create_event, update_event
- Phoenix tracing via tracer_config.tracer
- Agent does think → act (tool) → think ... until final JSON answer
"""

import asyncio
from typing import Optional

from openai import OpenAI  # not strictly needed, but matches qna_agent pattern
from agents import Agent, Runner, function_tool
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from tracer_config import tracer

client = OpenAI()

# ---------------------------------------------------------------------
# MCP bridge (async → sync) 
# ---------------------------------------------------------------------

SERVER_COMMAND = "python"
SERVER_ARGS = ["calendar_mcp_server.py"]  # same dir as this file


async def _call_mcp(tool_name: str, args: dict) -> str:
    """Low-level MCP call, returns plain text for the agent."""
    params = StdioServerParameters(
        command=SERVER_COMMAND,
        args=SERVER_ARGS,
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)

    # Try to collapse the MCP response into a simple string
    try:
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                parts = []
                for c in content:
                    # Text blocks usually have .text or are simple strings
                    text = getattr(c, "text", None)
                    parts.append(text if text is not None else str(c))
                return "\n".join(parts)
            return str(content)
        return str(result)
    except Exception:
        return str(result)


async def call_mcp(tool_name: str, args: dict) -> str:
    """Async wrapper so tools can await MCP without blocking the event loop."""
    return await _call_mcp(tool_name, args)

# ---------------------------------------------------------------------
# Tools exposed to the Agent (SDK @function_tool)
# ---------------------------------------------------------------------

@function_tool
async def list_calendar_events(date_start: str, date_end: Optional[str] = None) -> str:
    """
    List calendar events between date_start and date_end (inclusive).
    Dates can be 'today', 'tomorrow', 'next Monday', '2025-11-26', etc.
    """
    with tracer.start_as_current_span("tool_list_calendar_events") as span:
        span.set_attribute("date_start", date_start)
        if date_end:
            span.set_attribute("date_end", date_end)

        text = await call_mcp(
            "list_events",
            {
                "date_start": date_start,
                "date_end": date_end,
            },
        )
        span.set_attribute("mcp_result_preview", text[:200])
        return text


@function_tool
async def create_calendar_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[str] = None,
) -> str:
    """
    Create a new calendar event via MCP.
    Datetimes can be ISO8601 or natural language like 'tomorrow 3pm'.
    Attendees is an optional comma-separated list of emails.
    """
    with tracer.start_as_current_span("tool_create_calendar_event") as span:
        span.set_attribute("summary", summary)
        span.set_attribute("start_datetime", start_datetime)
        span.set_attribute("end_datetime", end_datetime)

        args = {
            "summary": summary,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": description,
            "location": location,
            "attendees": attendees,
        }

        text = await call_mcp("create_event", args)
        span.set_attribute("mcp_result_preview", text[:200])
        return text


@function_tool
async def update_calendar_event(
    event_id: str,
    summary: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """
    Update an existing calendar event by Google Calendar event_id.
    Any field left as None will be left unchanged.
    """
    with tracer.start_as_current_span("tool_update_calendar_event") as span:
        span.set_attribute("event_id", event_id)
        if summary:
            span.set_attribute("summary", summary)
        if start_datetime:
            span.set_attribute("start_datetime", start_datetime)
        if end_datetime:
            span.set_attribute("end_datetime", end_datetime)

        args = {
            "event_id": event_id,
            "summary": summary,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "description": description,
            "location": location,
        }

        text = await call_mcp("update_event", args)
        span.set_attribute("mcp_result_preview", text[:200])
        return text


# ---------------------------------------------------------------------
# Agent definition (true SDK agent, with tools + JSON final output)
# ---------------------------------------------------------------------

calendar_agent = Agent(
    name="GoogleCalendarAgent",
    model="gpt-5-nano",
    instructions="""
You are an autonomous calendar agent for David's Google Calendar.

You have three tools:
- list_calendar_events(date_start, date_end?)
- create_calendar_event(summary, start_datetime, end_datetime, description?, location?, attendees?)
- update_calendar_event(event_id, summary?, start_datetime?, end_datetime?, description?, location?)

General behaviour:
- Think step-by-step: PLAN → use tools → observe results → PLAN again until the task is complete.
- You may call tools multiple times in one run.
- Use natural language time like "today", "tomorrow", "next Thursday" when helpful; the MCP server can parse them.

Guidelines:
- When the user wants to SEE events, call list_calendar_events with an appropriate range.
- When the user wants to CREATE an event:
  - If any essential detail is missing (title, start, end), ask the user to clarify before calling the tool.
  - Otherwise call create_calendar_event.
- When the user wants to UPDATE an event:
  - If the event_id is unknown, first call list_calendar_events (for the relevant day or range),
    show options, and ask the user which Event ID to use.
  - Then call update_calendar_event with the chosen event_id and new fields.

Final answer format (always):
- When you are finished (no more tool calls needed), respond with VALID JSON only, no extra text.
- The JSON MUST have exactly these keys:
  {
    "final_answer": "<natural language summary to the user>",
    "reasoning": "<short explanation of the tools used and why>"
  }
- final_answer should be one or two short sentences, suitable to show directly to the user.
- reasoning can mention which tools were called and any important decisions.
""",
    tools=[list_calendar_events, create_calendar_event, update_calendar_event],
)


# ---------------------------------------------------------------------
# Simple REPL entrypoint (like qna_agent_phoenix_v5)
# ---------------------------------------------------------------------

@tracer.agent
def main():
    print("\n=== Google Calendar MCP Agent (Attempt 4 v1) ===")
    user_input = input("You: ").strip()
    if not user_input:
        print("No input provided.")
        return

    result = Runner.run_sync(calendar_agent, user_input)

    # Trace final output for Phoenix
    with tracer.start_as_current_span("calendar_agent_final_response") as span:
        preview = str(result.final_output)[:200] if hasattr(result, "final_output") else ""
        span.set_attribute("user_input", user_input)
        span.set_attribute("final_output_preview", preview)

    print("\n=== Agent final_output ===")
    print(result.final_output)


if __name__ == "__main__":
    main()
