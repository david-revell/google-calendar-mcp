# ============================================================
# mcp_calendar_agent_v6.py
#
# v6 = remember original intent across clarifications
# ============================================================

import asyncio
import json
import re

from openai import OpenAI
from agents import Agent, Runner
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tracer_config import tracer

# Session memory (per run)
session_state = {
    "matches": None,
    "pending_query": None,  # Stores the original query needing clarification
    "pending_tool": None,
    "pending_args": None,
}

client = OpenAI()

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
        "You must NEVER respond with questions or clarification prompts. You must ALWAYS return a JSON tool call, even if the user query is ambiguous. All clarifications are handled outside the model."
    ),
)

async def call_mcp(tool: str, args: dict):
    with tracer.start_as_current_span("call_mcp") as span:
        span.set_attribute("tool", tool)
        span.set_attribute("args_preview", json.dumps(args)[:200])
        server_params = StdioServerParameters(
            command="python",
            args=["calendar_mcp_server.py"],
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
                span.set_attribute("result_preview", str(result)[:200])
                return result

def interpret_nl(query: str):
    with tracer.start_as_current_span("interpret_nl") as span:
        out = Runner.run_sync(calendar_agent, query)
        tool_call = out.final_output
        if isinstance(tool_call, str):
            tool_call = json.loads(tool_call)
        span.set_attribute("resolved_tool", tool_call.get("tool"))
        span.set_attribute("args_preview", json.dumps(tool_call.get("args"))[:200])
        return tool_call["tool"], tool_call["args"]

def extract_event_id(plaintext: str):
    for line in plaintext.splitlines():
        if line.lower().startswith("event id:"):
            return line.split(":", 1)[1].strip()
    return None

def detect_multiple_matches(plaintext: str):
    matches = []
    current_summary = None
    current_id = None
    for line in plaintext.splitlines():
        stripped = line.strip()
        if stripped.startswith("Event:"):
            current_summary = stripped
        if stripped.startswith("Event ID:"):
            current_id = stripped.split(":", 1)[1].strip()
        if current_summary and current_id:
            matches.append((current_id, current_summary))
            current_summary = None
            current_id = None
    return matches

def main():
    print("\n=== Calendar MCP Agent v6 ===")
    while True:
        query = input("Ask your calendar anything: ").strip()
        # If we're awaiting clarification from the user
        if session_state["matches"] and session_state["pending_query"]:
            reply = query.lower()
            for eid, summary in session_state["matches"]:
                s = summary.lower().replace("event:", "").strip()
                if eid.lower() in reply or s in reply:
                    # Reconstruct the original tool call with clarified event_id
                    tool = session_state["pending_tool"]
                    args = dict(session_state["pending_args"])  # copy
                    args["event_id"] = eid
                    print(f"Clarified event_id: {eid}")
                    # Clear session state
                    session_state["matches"] = None
                    session_state["pending_query"] = None
                    session_state["pending_tool"] = None
                    session_state["pending_args"] = None
                    # Call MCP tool
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(call_mcp(tool, args))
                    print("\n=== MCP Result ===")
                    print(result)
                    break
            else:
                print("I didn't recognise which meeting you meant. Please try again or be more specific.")
            continue

        # Normal flow: interpret NL
        try:
            tool, args = interpret_nl(query)
            # Normalize attendees
            if "attendees" in args and isinstance(args["attendees"], list):
                args["attendees"] = ", ".join(args["attendees"])
            # Safety checks
            if not tool:
                raise ValueError("Agent returned no tool name.")
            if not isinstance(args, dict):
                raise ValueError("Agent returned invalid args structure.")
            # Fix invalid attendee formats
            if "attendees" in args:
                att = args["attendees"]
                if isinstance(att, str) and "@" not in att:
                    del args["attendees"]

            # If update_event is missing event_id, start clarification flow
            missing_id = (
                tool == "update_event" and (
                    "event_id" not in args or
                    str(args.get("event_id", "")).strip() in ["<id>", "id", "None", ""]
                )
            )
            if missing_id:
                print("\nFetching events to identify the correct event...")
                ql = query.lower()
                date_args = {"date_start": "tomorrow"} if "tomorrow" in ql else {"date_start": "today"}
                loop = asyncio.get_event_loop()
                events_text = loop.run_until_complete(call_mcp("list_events", date_args))
                raw_text = events_text.content[0].text if hasattr(events_text, "content") and events_text.content else ""
                matches = detect_multiple_matches(raw_text)
                if len(matches) > 1:
                    session_state["matches"] = matches
                    session_state["pending_query"] = query
                    session_state["pending_tool"] = tool
                    session_state["pending_args"] = args
                    options = ", ".join([f"{sid} ({summary})" for sid, summary in matches])
                    print(f"\nMultiple events match your query: {options}")
                    print("Please specify which meeting you meant (by event ID or summary).")
                    continue  # Wait for user clarification
                # If only one match, auto-select
                event_id = extract_event_id(raw_text)
                if not event_id:
                    raise RuntimeError("Could not automatically detect an eventId.")
                args["event_id"] = event_id
                print(f"Selected eventId = {event_id}")

            # Call MCP tool
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(call_mcp(tool, args))
            print("\n=== MCP Result ===")
            print(result)

        except Exception as e:
            print("Runtime error:", e)

if __name__ == "__main__":
    main()