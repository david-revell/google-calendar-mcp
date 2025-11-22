# ============================================================
# mcp_calendar_agent_v1.py
#
# TRUE MINIMAL VERSION
# - Natural language → JSON via Agent SDK
# - MCP server call
# - Only one Phoenix trace span (top-level)
# - No clarifying logic
# - No helpers
# - No nested spans
# ============================================================

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
from openai import OpenAI
from agents import Agent, Runner

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tracer_config import tracer


client = OpenAI()

# -----------------------------------------------------------
# Agent definition
# -----------------------------------------------------------
calendar_agent = Agent(
    name="CalendarAgent",
    model="gpt-5-nano",
    instructions=(
        "You are a calendar assistant. "
        "Understand natural language and decide which calendar operation is needed "
        "(list events, create event, update event). "
        "Extract all relevant details (dates, times, title, attendees). "
        "Respond ONLY with valid JSON. "
        "Do NOT write sentences, explanations, or comments. "
        "If you do NOT know the event_id, you MUST leave the event_id field out entirely. "
        "NEVER invent or guess an event_id. "
        "NEVER use placeholders such as <id>, next_meeting, id, or similar. "
        "Valid example: {\"tool\": \"list_events\", \"args\": {\"date_start\": \"today\"}} "
        "Valid example: {\"tool\": \"create_event\", \"args\": {\"summary\": \"Meeting with Nicolas\", \"start_datetime\": \"<parsed>\", \"end_datetime\": \"<parsed_plus_1h>\"}} "
        "Valid example: {\"tool\": \"update_event\", \"args\": {\"event_id\": \"abc123\", \"start_datetime\": \"<parsed>\", \"end_datetime\": \"<parsed>\"}} "
        "Invalid example (because event_id is unknown): {\"tool\": \"update_event\", \"args\": {\"event_id\": \"<id>\", ...}} "
        "If the user wants to move, change, shift, or reschedule a meeting, you MUST choose the update_event tool. "
        "You must NEVER respond with questions or clarification prompts. You must ALWAYS return a JSON tool call, even if the user query is ambiguous."
    ),
)


# -----------------------------------------------------------
# MCP call function (no Phoenix tracing inside)
# -----------------------------------------------------------
async def call_mcp(tool: str, args: dict):
    server_params = StdioServerParameters(
        command="python",
        args=["calendar_mcp_server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return result


# -----------------------------------------------------------
# NL → (tool, args)
# -----------------------------------------------------------
def interpret_nl(query: str):
    out = Runner.run_sync(calendar_agent, query)
    tool_call = out.final_output

    if isinstance(tool_call, str):
        tool_call = json.loads(tool_call)

    return tool_call["tool"], tool_call["args"]


# -----------------------------------------------------------
# MAIN LOOP — only one Phoenix span
# -----------------------------------------------------------
def main():
    print("\n=== Calendar MCP Agent (v1 Minimal) ===")
    query = input("Ask your calendar anything: ")

    with tracer.start_as_current_span("calendar_agent_run") as span:
        try:
            tool, args = interpret_nl(query)
            print("AGENT OUTPUT:", tool, args)

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(call_mcp(tool, args))

            print("\n=== MCP Result ===")
            print(result)

            span.set_attribute("status", "success")

        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error_message", str(e))
            print("Runtime error:", e)


# -----------------------------------------------------------
# REPL loop
# -----------------------------------------------------------
if __name__ == "__main__":
    while True:
        main()
