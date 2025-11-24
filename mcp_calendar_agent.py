# ============================================================
# mcp_calendar_agent_v4_1.py
#
# Version 4.1:
# - Behaviour identical to v4 (clarifying questions + auto update_event)
# - Natural language → JSON via Agent SDK
# - MCP server call (calendar_mcp_server.py)
# - Plain-text chat history (messages[])
# - Auto update_event:
#     * LLM parses new datetime
#     * Agent calls list_events for that date
#     * Parses MCP text output for title + event_id
#     * Matches by title and calls update_event with event_id
# - Clarifying questions when information is missing
# - HIERARCHICAL Phoenix tracing:
#     * calendar_agent_run       (top-level agent span)
#         * interpret_nl             (chain)
#         * ask_clarifying_question  (agent, when used)
#         * process_tool_call        (chain, when JSON)
#             * call_mcp             (tool, for each MCP call)
#         * state_transition         (metadata, whenever state changes)
# ============================================================

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
from datetime import datetime, timedelta

from agents import Agent, Runner
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tracer_config import tracer

# ------------------------------------------------------------
# Simple workflow states (kept for future)
# ------------------------------------------------------------
STATE_IDLE = "IDLE"
STATE_WAITING_FOR_EVENT_SELECTION = "WAITING_FOR_EVENT_SELECTION"
STATE_WAITING_FOR_UPDATE_DETAILS = "WAITING_FOR_UPDATE_DETAILS"

session_state = {
    "state": STATE_IDLE,
    "pending_update_args": None,
    "candidate_events": None,
    "selected_event_id": None,
    "last_list_date": None,   # for possible fallback use
}

# ------------------------------------------------------------
# Chat history for semantic memory (plain text)
# ------------------------------------------------------------
messages: list[str] = []


# ------------------------------------------------------------
# Helper: explicit state transition span
# ------------------------------------------------------------
def transition_state(new_state: str, reason: str | None = None):
    global session_state
    old_state = session_state["state"]
    session_state["state"] = new_state

    with tracer.start_as_current_span("state_transition") as s_span:
        s_span.set_attribute("component", "metadata")
        s_span.set_attribute("from_state", old_state)
        s_span.set_attribute("to_state", new_state)
        s_span.set_attribute("changed", old_state != new_state)
        if reason:
            s_span.set_attribute("reason", reason)


# ------------------------------------------------------------
# Helper: ask a clarifying question (no tool call)
# ------------------------------------------------------------
def ask(question: str):
    # This helper gets its own span so we can see clarifications separately.
    with tracer.start_as_current_span("ask_clarifying_question") as span:
        span.set_attribute("component", "agent")
        print(question)
        messages.append(f"Assistant: {question}")


# ------------------------------------------------------------
# Agent definition (fixed instructions)
# ------------------------------------------------------------
calendar_agent = Agent(
    name="CalendarAgent",
    model="gpt-5-nano",
    instructions=(
        "You are a calendar assistant for a single user.\n\n"
        "Your job is to:\n"
        "1) When you have enough information, output ONLY valid JSON representing a tool call:\n"
        "   list_events, create_event, or update_event.\n"
        "2) When essential information is missing, respond with a short clarifying question in "
        "natural language and do NOT output JSON.\n\n"
        "JSON OUTPUT RULES:\n"
        "- Output ONLY valid JSON (no comments, no extra text) when you choose to call a tool.\n"
        "- Use REAL ISO 8601 datetime strings for all dates and times.\n"
        "- NEVER output placeholders such as <parsed>, <parsed_plus_1h>, <unknown>, or similar.\n"
        "- NEVER guess or invent an event_id. If you do not know it, omit it entirely.\n"
        "- If the user wants to move/reschedule an event but does not specify an event_id, "
        "produce update_event WITHOUT an event_id, but INCLUDE a 'summary' field if the user "
        "identified the meeting by title.\n\n"
        "VALID JSON EXAMPLES:\n"
        "{\"tool\": \"list_events\", \"args\": {\"date_start\": \"today\"}}\n"
        "{\"tool\": \"create_event\", \"args\": {"
        "\"summary\": \"Meeting with Nicolas\", "
        "\"start_datetime\": \"2025-11-24T16:00:00\", "
        "\"end_datetime\": \"2025-11-24T17:00:00\"}}\n"
        "{\"tool\": \"update_event\", \"args\": {"
        "\"event_id\": \"abc123\", "
        "\"summary\": \"Meeting with Nicolas\", "
        "\"start_datetime\": \"2025-11-25T15:00:00\", "
        "\"end_datetime\": \"2025-11-25T16:00:00\"}}\n"
        "{\"tool\": \"update_event\", \"args\": {"
        "\"summary\": \"Meeting with Nicolas\", "
        "\"start_datetime\": \"2025-11-25T15:00:00\", "
        "\"end_datetime\": \"2025-11-25T16:00:00\"}}\n\n"
        "INVALID JSON EXAMPLES (DO NOT DO THIS):\n"
        "{\"start_datetime\": \"<parsed>\"}\n"
        "{\"event_id\": \"<id>\"}\n"
        "{\"tool\": \"update_event\", \"args\": {\"event_id\": \"next_meeting\"}}\n\n"
        "CLARIFYING QUESTIONS:\n"
        "- If you lack an essential detail (e.g. which meeting, which date, what change), "
        "respond with a brief, precise question in natural language.\n"
        "- When asking a question, DO NOT output JSON.\n"
        "- Never fabricate details. If you are unsure, ask.\n"
    ),
)


# ------------------------------------------------------------
# MCP call function (no Phoenix inside; tracing is at call sites)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# NL → (call_type, tool_or_question, args) using full chat history
# call_type = "json"      → tool_or_question = tool name, args = dict
# call_type = "question"  → tool_or_question = question text, args = None
# ------------------------------------------------------------
def interpret_nl(query: str):
    messages.append(f"User: {query}")
    full_context = "\n".join(messages)

    with tracer.start_as_current_span("interpret_nl") as span:
        span.set_attribute("component", "chain")
        span.set_attribute("history_length", len(messages))

        out = Runner.run_sync(calendar_agent, full_context)
        raw = out.final_output

        if isinstance(raw, str):
            stripped = raw.strip()
        else:
            stripped = str(raw).strip()

        # Try JSON first.
        try:
            tool_call = json.loads(stripped)
            messages.append(f"Assistant: {stripped}")
            span.set_attribute("output_type", "json")
            span.set_attribute("tool", tool_call.get("tool"))
            return "json", tool_call["tool"], tool_call["args"]

        except Exception:
            # Clarifying question path.
            question = stripped
            messages.append(f"Assistant: {question}")
            span.set_attribute("output_type", "question")
            return "question", question, None


# ------------------------------------------------------------
# Helpers for parsing MCP list_events output
# ------------------------------------------------------------
def normalise_title(title: str) -> str:
    return title.strip().casefold()


def parse_time_to_iso(date_str: str, time_str: str) -> str:
    # date_str: 'YYYY-MM-DD', time_str like '04:00 PM'
    dt_time = datetime.strptime(time_str.strip(), "%I:%M %p").time()
    dt = datetime.fromisoformat(date_str + "T00:00:00").replace(
        hour=dt_time.hour, minute=dt_time.minute, second=0, microsecond=0
    )
    return dt.isoformat()


def parse_events_from_result(result, date_str: str):
    """
    Parse events from MCP list_events result.

    Expected text format (per event):

        Event: Test create
        Event ID: gj5k5...
        Time: 04:00 PM - 05:00 PM
        Location: ...
        Description: ...
        Attendees: ...

    Blank line between events.
    """
    text_block = None

    # Try structuredContent['result'] first
    if hasattr(result, "structuredContent"):
        sc = result.structuredContent
        if isinstance(sc, dict) and "result" in sc:
            text_block = sc["result"]

    # Fallback: first content.text
    if not text_block and hasattr(result, "content"):
        try:
            first = result.content[0]
            text_block = getattr(first, "text", None)
        except Exception:
            pass

    # Last resort: string of whole object
    if text_block is None:
        text_block = str(result)

    events = []
    current = None

    for raw_line in text_block.splitlines():
        line = raw_line.strip()

        if not line:
            # blank line → end of current event
            if current:
                events.append(current)
                current = None
            continue

        if line.startswith("Event:"):
            if current:
                events.append(current)
            title = line[len("Event:"):].strip()
            current = {"title": title, "event_id": None, "start": None, "end": None}

        elif line.startswith("Event ID:") and current is not None:
            eid = line[len("Event ID:"):].strip()
            current["event_id"] = eid

        elif line.startswith("Time:") and current is not None:
            time_part = line[len("Time:"):].strip()  # '04:00 PM - 05:00 PM'
            try:
                start_str, end_str = [p.strip() for p in time_part.split("-")]
                current["start"] = parse_time_to_iso(date_str, start_str)
                current["end"] = parse_time_to_iso(date_str, end_str)
            except Exception:
                # Leave as None if parsing fails
                pass

    if current:
        events.append(current)

    return events


def extract_date_from_list_args(args: dict) -> str | None:
    ds = args.get("date_start")
    if not ds:
        return None

    if ds == "today":
        return datetime.today().date().isoformat()
    if ds == "tomorrow":
        return (datetime.today().date() + timedelta(days=1)).isoformat()

    if "T" in ds:
        return ds.split("T")[0]

    if len(ds) == 10 and ds[4] == "-" and ds[7] == "-":
        return ds

    return None


def extract_date_from_iso_datetime(dt_str: str | None) -> str | None:
    if not dt_str:
        return None
    if "T" in dt_str:
        return dt_str.split("T")[0]
    return None


# ------------------------------------------------------------
# Core turn handler (top-level span + nested spans)
# ------------------------------------------------------------
def handle_turn(user_input: str):
    global session_state

    with tracer.start_as_current_span("calendar_agent_run") as span:
        span.set_attribute("component", "agent")
        span.set_attribute("current_state", session_state["state"])

        try:
            state = session_state["state"]
            loop = asyncio.get_event_loop()

            call_type, tool_or_question, args = interpret_nl(user_input)

            # Clarifying question path
            if call_type == "question":
                span.set_attribute("decision", "clarifying_question")
                ask(tool_or_question)
                span.set_attribute("status", "clarifying_question")
                return

            # From here on, we have a JSON tool call.
            tool = tool_or_question
            print("AGENT OUTPUT:", tool, args)

            with tracer.start_as_current_span("process_tool_call") as process_span:
                process_span.set_attribute("component", "chain")
                process_span.set_attribute("tool", tool)
                process_span.set_attribute("state_at_start", state)

                # Only IDLE is really used in this version
                if state == STATE_IDLE:
                    # ---------- update_event with automatic lookup ----------
                    if tool == "update_event":
                        # Work out the target date from the parsed datetime,
                        # or fall back to last list date.
                        target_date = extract_date_from_iso_datetime(
                            args.get("start_datetime")
                        )
                        if not target_date:
                            target_date = session_state.get("last_list_date")

                        if not target_date:
                            print(
                                "\nCannot determine the date of the meeting to update.\n"
                                "Please either specify the date explicitly (e.g. 'tomorrow' or a date),\n"
                                "or run a 'list my events ...' command first."
                            )
                            process_span.set_attribute(
                                "status", "missing_date_for_update"
                            )
                            span.set_attribute("status", "missing_date_for_update")
                            return

                        title = args.get("summary")
                        if not title:
                            print(
                                "\nThe update_event call does not include a 'summary' field.\n"
                                "Please refer to the meeting by its title so the agent can match it."
                            )
                            process_span.set_attribute(
                                "status", "missing_summary_for_update"
                            )
                            span.set_attribute("status", "missing_summary_for_update")
                            return

                        # If event_id already present (user provided it), just call MCP directly.
                        if args.get("event_id"):
                            with tracer.start_as_current_span("call_mcp") as m_span:
                                m_span.set_attribute("component", "tool")
                                m_span.set_attribute("tool", "update_event")
                                m_span.set_attribute("arg_keys", list(args.keys()))
                                result = loop.run_until_complete(
                                    call_mcp("update_event", args)
                                )
                            print("\n=== MCP Result (update_event) ===")
                            print(result)

                            transition_state(
                                STATE_IDLE,
                                reason="update_event_direct",
                            )
                            process_span.set_attribute(
                                "status", "success_update_event_direct"
                            )
                            span.set_attribute("status", "success_update_event_direct")
                            return

                        # Otherwise, automatically list events on that date and match by title.
                        list_args = {
                            "date_start": f"{target_date}T00:00:00",
                            "date_end": f"{target_date}T23:59:59",
                        }
                        with tracer.start_as_current_span("call_mcp") as m_span:
                            m_span.set_attribute("component", "tool")
                            m_span.set_attribute("tool", "list_events")
                            m_span.set_attribute("args_summary", list_args)
                            list_result = loop.run_until_complete(
                                call_mcp("list_events", list_args)
                            )

                        events = parse_events_from_result(list_result, target_date)
                        norm_title = normalise_title(title)
                        matches = [
                            e for e in events
                            if normalise_title(e["title"]) == norm_title
                        ]

                        if len(matches) == 0:
                            print(
                                "\nNo matching event found for title "
                                f"'{title}' on {target_date}.\n"
                                "Please check the title or date and try again."
                            )
                            process_span.set_attribute(
                                "status", "no_matching_event_for_update"
                            )
                            span.set_attribute(
                                "status", "no_matching_event_for_update"
                            )
                            return

                        if len(matches) > 1:
                            print(
                                "\nMultiple events found with the title "
                                f"'{title}' on {target_date}. "
                                "Automatic disambiguation is not implemented yet.\n"
                                "Matching events:\n"
                            )
                            for e in matches:
                                print(
                                    f"- {e['title']} (Event ID: {e['event_id']}, "
                                    f"start: {e['start']}, end: {e['end']})"
                                )
                            print(
                                "\nPlease rerun your command and specify which event ID to use, "
                                "or adjust the title to be unique."
                            )
                            process_span.set_attribute(
                                "status", "multiple_matching_events_for_update"
                            )
                            span.set_attribute(
                                "status", "multiple_matching_events_for_update"
                            )
                            return

                        # Exactly one match → attach event_id and call update_event
                        match = matches[0]
                        args["event_id"] = match["event_id"]

                        with tracer.start_as_current_span("call_mcp") as m_span:
                            m_span.set_attribute("component", "tool")
                            m_span.set_attribute("tool", "update_event")
                            m_span.set_attribute("arg_keys", list(args.keys()))
                            result = loop.run_until_complete(
                                call_mcp("update_event", args)
                            )

                        print("\n=== MCP Result (update_event) ===")
                        print(result)

                        transition_state(
                            STATE_IDLE,
                            reason="update_event_auto_resolved",
                        )
                        process_span.set_attribute(
                            "status", "success_update_event_auto"
                        )
                        span.set_attribute("status", "success_update_event_auto")
                        return

                    # ---------- Non-update tools ----------
                    if tool == "list_events":
                        list_date = extract_date_from_list_args(args)
                        if list_date:
                            session_state["last_list_date"] = list_date

                    with tracer.start_as_current_span("call_mcp") as m_span:
                        m_span.set_attribute("component", "tool")
                        m_span.set_attribute("tool", tool)
                        m_span.set_attribute("arg_keys", list(args.keys()))
                        result = loop.run_until_complete(call_mcp(tool, args))

                    print("\n=== MCP Result ===")
                    print(result)

                    transition_state(
                        STATE_IDLE,
                        reason=f"{tool}_completed",
                    )
                    process_span.set_attribute("status", "success")
                    span.set_attribute("status", "success")
                    return

                elif state == STATE_WAITING_FOR_EVENT_SELECTION:
                    print(
                        "\n[STATE_WAITING_FOR_EVENT_SELECTION]\n"
                        "Selection flow not implemented in this version."
                    )
                    process_span.set_attribute(
                        "status", "unimplemented_selection_state"
                    )
                    span.set_attribute("status", "unimplemented_selection_state")
                    return

                elif state == STATE_WAITING_FOR_UPDATE_DETAILS:
                    print(
                        "\n[STATE_WAITING_FOR_UPDATE_DETAILS]\n"
                        "Detailed update flow not implemented in this version."
                    )
                    process_span.set_attribute(
                        "status", "unimplemented_update_details_state"
                    )
                    span.set_attribute("status", "unimplemented_update_details_state")
                    return

        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error_message", str(e))
            print("Runtime error:", e)


# ------------------------------------------------------------
# REPL loop
# ------------------------------------------------------------
def main():
    print("\n=== Google Calendar MCP Agent (v4.1: tracing upgrade + state spans) ===")
    print("Type Ctrl-C to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            handle_turn(user_input)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
