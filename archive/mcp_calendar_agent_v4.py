# ============================================================
# mcp_calendar_agent_v4.py
#
# v1  = first working MCP round-trip
# v2  = natural-language interpretation via the Agent SDK
# v3  = Phoenix tracing
# v4  = eventId isn't needed to update an event
# ============================================================

import asyncio
# Windows requires a specific event-loop policy for subprocess I/O.
# Without this, async pipes on Windows can deadlock or throw cryptic errors.
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
from openai import OpenAI
from agents import Agent, Runner
# OpenAI client + Agents SDK:
#   Agent  = the definition of model + instructions
#   Runner = small helper to run the agent synchronously (no async needed here)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# MCP SDK:
#   StdioServerParameters = describes how to launch an MCP server subprocess
#   stdio_client          = opens the pipes to that subprocess
#   ClientSession         = handles JSON-RPC messaging over those pipes

from tracer_config import tracer
# Phoenix tracer

client = OpenAI()
# Creates the OpenAI client. Required by the Agents SDK even if the script
# does not call the OpenAI API directly at this stage.

MCP_CMD = ["python", "calendar_mcp_server.py"]
# Historical leftover from v1.
# No longer used because StdioServerParameters handles the launch.
# Kept only for context.

# ---------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------
calendar_agent = Agent(
    name="CalendarAgent",
    model="gpt-5-nano",
    instructions=(
        # These instructions tell the model exactly how to behave.
        # It must:
        #   - interpret the user's natural language
        #   - choose list_events / create_event / update_event
        #   - extract structured details
        #   - output ONLY valid JSON (no sentences, no commentary)
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
        "If the user wants to move, change, shift, or reschedule a meeting, you MUST choose the update_event tool."
     ),
)

# ---------------------------------------------------------------------
# call_mcp(): launch MCP server + call tool + return result
# ---------------------------------------------------------------------
async def call_mcp(tool: str, args: dict):
    """
    Responsible for:
      1. Launching the MCP server as a subprocess
      2. Opening STDIO pipes to it
      3. Starting a JSON-RPC session via ClientSession
      4. Calling the chosen tool with the provided args
      5. Returning whatever the server returns

    This is the central MCP interaction layer.
    """
    with tracer.start_as_current_span("call_mcp") as span:
        span.set_attribute("tool", tool)
        span.set_attribute("args_preview", json.dumps(args)[:200])

        # StdioServerParameters describes exactly:
        #   - which executable to run
        #   - which script to pass
        #   - which environment to inherit
        server_params = StdioServerParameters(
            command="python",                # Use local Python interpreter
            args=["calendar_mcp_server.py"], # The actual MCP server file
            env=None,                        # Inherit parent environment
        )

        # stdio_client launches the server and yields:
        #   read  = async stream for server stdout
        #   write = async stream for server stdin
        with tracer.start_as_current_span("mcp_server_init"):
            async with stdio_client(server_params) as (read, write):

                # ClientSession handles:
                #   - JSON-RPC message construction
                #   - sending tool calls
                #   - receiving responses
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    with tracer.start_as_current_span("mcp_session_initialize"):
                        await session.initialize()

                    with tracer.start_as_current_span("mcp_tool_call") as tool_span:
                        tool_span.set_attribute("tool", tool)
                        tool_span.set_attribute("args_preview", json.dumps(args)[:200])

                        result = await session.call_tool(tool, args)

                        tool_span.set_attribute("result_preview", str(result)[:200])

                    span.set_attribute("result_preview", str(result)[:200])
                    return result


# ---------------------------------------------------------------------
# interpret_nl(): natural language → JSON {tool, args}
# ---------------------------------------------------------------------
def interpret_nl(query: str):
    with tracer.start_as_current_span("interpret_nl") as span:
        span.set_attribute("nl_query", query[:200])

        with tracer.start_as_current_span("agent_run_sync"):
            out = Runner.run_sync(calendar_agent, query)

        tool_call = out.final_output

        if isinstance(tool_call, str):
            tool_call = json.loads(tool_call)

        span.set_attribute("resolved_tool", tool_call.get("tool"))
        span.set_attribute("args_preview", json.dumps(tool_call.get("args"))[:200])

        return tool_call["tool"], tool_call["args"]
    
# ---------------------------------------------------------------------
# extract_event_id(): helper function to read eventId
# ---------------------------------------------------------------------

def extract_event_id(plaintext: str, summary_hint: str = None, time_hint: str = None):
    for line in plaintext.splitlines():
        if line.lower().startswith("event id:"):
            return line.split(":", 1)[1].strip()
    return None

# ---------------------------------------------------------------------
# main(): top-level entry
# ---------------------------------------------------------------------
def main():
    print("\n=== Calendar MCP Agent ===")

    query = input("Ask your calendar anything: ")

    with tracer.start_as_current_span("calendar_agent_run") as span:
        try:
            # Step 1: interpret NL
            tool, args = interpret_nl(query)
            print("AGENT OUTPUT:", tool, args)

            # Normalise list-of-names → comma-separated string
            if "attendees" in args and isinstance(args["attendees"], list):
                args["attendees"] = ", ".join(args["attendees"])

            # Safety checks
            if not tool:
                raise ValueError("Agent returned no tool name.")
            if not isinstance(args, dict):
                raise ValueError("Agent returned invalid args structure.")

            # Fix invalid attendee formats (names instead of emails)
            if "attendees" in args:
                att = args["attendees"]

                # If it's a single string with no '@', remove the field
                if isinstance(att, str) and "@" not in att:
                    del args["attendees"]

            # Treat '<id>' and similar placeholders as missing (DR - note - this is a bit dubious in my opinion)
            missing_id = ("event_id" not in args) or (str(args.get("event_id", "")).strip() in ["<id>", "id", "None", ""])
            if tool == "update_event" and missing_id:

                print("\nFetching events to identify the correct event...")
                # Determine which day to list based on the user's query
                query_lower = query.lower()
                if "tomorrow" in query_lower:
                    date_start = "tomorrow"
                    date_end = "tomorrow"
                else:
                    date_start = "today"
                    date_end = "7d"

                events_text = asyncio.run(call_mcp(
                    "list_events",
                    {"date_start": date_start, "date_end": date_end}
                ))

                print("DEBUG RAW EVENTS:", repr(events_text))
                # Extract raw plaintext from the CallToolResult object
                raw_text = events_text.content[0].text if events_text.content else ""
                event_id = extract_event_id(raw_text)

                if not event_id:
                    raise RuntimeError("Could not automatically detect an eventId.")

                args["event_id"] = event_id
                print(f"Selected eventId = {event_id}")


            # Step 2: call MCP
            events = asyncio.run(call_mcp(tool, args))

            # Step 3: final output span
            with tracer.start_as_current_span("final_output") as span_out:
                span_out.set_attribute("result_preview", str(events)[:200])

            print("\n=== MCP Result ===")
            print(events)

            # Mark success
            span.set_attribute("status", "success")

        except Exception as e:
            # Mark failure
            span.set_attribute("status", "error")
            span.set_attribute("error_message", str(e))
            raise

# ---------------------------------------------------------------------
# Windows cleanup block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Runtime error:", e)
    finally:
        # Without this, Windows may emit
        # "Event loop is closed" warnings on exit.
        try:
            asyncio.get_event_loop().close()
        except Exception:
            pass
