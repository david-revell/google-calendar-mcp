# ============================================================
# mcp_calendar_agent_v1.py  --  FIRST WORKING STDIO MCP AGENT
# This file is frozen. Do NOT modify. Future versions go in:
#     mcp_calendar_agent.py  (active dev)
#     mcp_calendar_agent_v2.py, v3.py, etc.
# ============================================================

import asyncio
# Windows needs a special event loop policy for async subprocess I/O.
# This line avoids known Windows issues with pipes and prevents random failures.
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
from openai import OpenAI
from agents import Agent, Runner
# NOTE: 'openai' and 'agents' here are from the OpenAI Agents SDK.
# Agent = defines your instructions + model
# Runner = runs that agent synchronously for simple one-shot queries

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# These are the official MCP Python SDK imports.
# ClientSession  = manages JSON-RPC sessions over any transport (stdio/SSE/etc.)
# StdioServerParameters = describes how to launch a subprocess MCP server
# stdio_client   = opens the STDIO connection to the MCP server


client = OpenAI()
# Creates an OpenAI client instance.
# Even though we do not call OpenAI endpoints in Version 1,
# the Agents SDK requires this object to exist.

MCP_CMD = ["python", "calendar_mcp_server.py"]
# Old variable kept for reference only.
# It is NOT used anymore.
# The real process command is now encoded inside StdioServerParameters.
# Keeping this here preserves historical context for Version 1.


# ---------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------
calendar_agent = Agent(
    name="CalendarAgent",
    model="gpt-5-nano",   # cheapest OpenAI model
    instructions=(
        # These instructions force the Agent to output structured JSON.
        #
        # IMPORTANT:
        # This is NOT a conversational agent yet.
        # It is deliberately restrictive so we get a known-stable baseline.
        #
        # Example output:
        # {"tool": "list_events", "args": {"date_start": "today"}}
        "When the user asks for events, output ONLY JSON:\n"
        '{"tool": "list_events", "args": {"date_start": "<date>"}}'
    ),
)


# ---------------------------------------------------------------------
# call_mcp(): handles the full interaction with your MCP server
# ---------------------------------------------------------------------
async def call_mcp(tool: str, args: dict):
    """
    This function:
        - launches the MCP server as a subprocess
        - opens STDIO streams to it
        - initializes a JSON-RPC session
        - calls the specified tool with the provided arguments
        - returns the raw tool result

    This is the *core* of Version 1.
    """

    # REQUIRED FOR THE NEW MCP SDK:
    # The server MUST be described using a StdioServerParameters object.
    # This avoids the previous errors ("list has no attribute 'command'").
    server_params = StdioServerParameters(
        command="python",                  # actual executable to run
        args=["calendar_mcp_server.py"],   # script that launches your MCP server
        env=None,                          # inherit parent process environment
    )

    # stdio_client(server_params):
    #   - launches your MCP server
    #   - returns (read_stream, write_stream)
    async with stdio_client(server_params) as (read, write):

        # ClientSession(read, write):
        #   - sends JSON-RPC messages to the server
        #   - receives JSON-RPC responses back
        async with ClientSession(read, write) as session:

            # MUST be called before first tool call.
            # Establishes handshake + advertises available tools.
            await session.initialize()

            # Now we call an actual tool:
            # e.g., tool="list_events", args={"date_start":"today"}
            result = await session.call_tool(tool, args)

            return result


# ---------------------------------------------------------------------
# main(): top-level entry point
# ---------------------------------------------------------------------
def main():
    print("\n=== Calendar MCP Agent (STDIO) ===")

    # HARD-WIRED query for Version 1
    # Later versions will accept real natural language.
    query = "list today's events"

    # Let the OpenAI Agent produce a JSON tool request.
    # Runner.run_sync() = blocking wrapper for the async agent run.
    out = Runner.run_sync(calendar_agent, query)

    # Extract the final agent output.
    tool_call = out.final_output

    # The agent may return a Python string containing JSON.
    # We parse it into a dict for safe access.
    if isinstance(tool_call, str):
        tool_call = json.loads(tool_call)

    # The JSON expected from the instructions:
    #   {"tool": "...", "args": {...}}
    tool = tool_call["tool"]
    args = tool_call["args"]

    # Now run the tool call through the MCP server.
    events = asyncio.run(call_mcp(tool, args))

    print("\n=== MCP Result ===")
    print(events)   # Will show a ContentBlock or structuredContent


# ---------------------------------------------------------------------
# REQUIRED Windows cleanup block
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Runtime error:", e)
    finally:
        # This avoids "event loop is closed" noise on Windows.
        try:
            asyncio.get_event_loop().close()
        except Exception:
            pass
