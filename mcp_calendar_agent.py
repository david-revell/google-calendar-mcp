#!/usr/bin/env python
# ============================================================
# Calendar Agent v3 — v2 architecture + LIST/CREATE/UPDATE
# ============================================================

import asyncio
import json
from openai import OpenAI
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from tracer_config import tracer

client = OpenAI()


class CalendarAgent:
    def __init__(self):
        self.memory = []

    def remember(self, role: str, msg: str):
        self.memory.append({"role": role, "content": msg})

    def llm(self, messages):
        resp = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

    # ---------------------------------------------------------
    # PLANNER (v2 extended: LIST / CREATE / UPDATE / ASK <q>)
    # ---------------------------------------------------------
    @tracer.chain()
    def plan(self, user_message: str) -> str:
        prompt = (
            "You are the planner for a calendar agent.\n"
            f"User said: \"{user_message}\"\n\n"
            "You MUST respond with exactly ONE of:\n"
            "PLAN:LIST\n"
            "PLAN:CREATE\n"
            "PLAN:UPDATE\n"
            "PLAN:ASK <question>\n"
            "Nothing else.\n\n"
            "- LIST → user wants to view events.\n"
            "- CREATE → user clearly wants a new event.\n"
            "- UPDATE → user clearly wants to change an existing event.\n"
            "- ASK → essential details missing.\n"
        )
        return self.llm([{"role": "user", "content": prompt}])

    # ---------------------------------------------------------
    # ARG EXTRACTORS (LLM returns tool + args + reasoning)
    # ---------------------------------------------------------
    @tracer.chain()
    def extract_create_args(self, user_message: str):
        system = (
            "You extract event-creation details for a calendar tool.\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "tool": "create_event",\n'
            '  "args": {\n'
            '    "summary": "<title>",\n'
            '    "start_datetime": "<ISO8601>",\n'
            '    "end_datetime": "<ISO8601>"\n'
            "  },\n"
            '  "reasoning": "<short natural language explanation>"\n'
            "}\n\n"
            "Rules:\n"
            "- Use REAL ISO 8601 datetimes (e.g. 2025-11-26T18:00:00).\n"
            "- NEVER invent attendees; do NOT include any attendees field.\n"
            "- Do NOT include placeholders like <parsed>, <time>, or similar.\n"
            "- No extra fields at top level besides tool, args, reasoning.\n"
            "- Output MUST be valid JSON parsable by json.loads.\n"
        )
        user = f"User request: {user_message}"
        raw = self.llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        try:
            obj = json.loads(raw)
            if obj.get("tool") != "create_event":
                return None
            args = obj.get("args") or {}
            reasoning = obj.get("reasoning") or ""
            args["reasoning"] = reasoning
            return args
        except Exception:
            return None

    @tracer.chain()
    def extract_update_args(self, user_message: str):
        system = (
            "You extract event-update details for a calendar tool.\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "tool": "update_event",\n'
            '  "args": {\n'
            '    "event_id": "<id or null>",\n'
            '    "summary": "<new title or null>",\n'
            '    "start_datetime": "<ISO8601 or null>",\n'
            '    "end_datetime": "<ISO8601 or null>"\n'
            "  },\n"
            '  "reasoning": "<short natural language explanation>"\n'
            "}\n\n"
            "Rules:\n"
            "- If the user mentions an event ID, copy it exactly into event_id.\n"
            "- If no event ID is given, set event_id to null. NEVER invent IDs.\n"
            "- Use REAL ISO 8601 datetimes.\n"
            "- Do NOT include attendees.\n"
            "- No placeholders like <parsed>.\n"
            "- No extra fields at top level besides tool, args, reasoning.\n"
        )
        user = f"User request: {user_message}"
        raw = self.llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        try:
            obj = json.loads(raw)
            if obj.get("tool") != "update_event":
                return None
            args = obj.get("args") or {}
            reasoning = obj.get("reasoning") or ""
            args["reasoning"] = reasoning
            return args
        except Exception:
            return None

    # ---------------------------------------------------------
    # MCP TOOL CALL (same idea as v2)
    # ---------------------------------------------------------
    @tracer.tool()
    async def call_mcp(self, tool: str, args: dict):
        params = StdioServerParameters(
            command="python",
            args=["calendar_mcp_server.py"],
            env=None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, args)

    # ---------------------------------------------------------
    # FINALISER (natural language, same concept as v2)
    # ---------------------------------------------------------
    @tracer.chain()
    def finalise(self, tool_result) -> str:
        system = (
            "You are the finaliser for a calendar agent.\n"
            "You get a raw tool result and must reply with ONE short sentence "
            "explaining what happened in natural language. No JSON."
        )
        user = f"Tool result:\n{tool_result}"
        return self.llm(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    # ---------------------------------------------------------
    # HANDLERS (modular, called from run_turn)
    # ---------------------------------------------------------
    async def handle_list(self, user_message: str) -> str:
        result = await self.call_mcp(
            "list_events",
            {
                "date_start": "today",
                "reasoning": "Planner chose LIST for the user's request.",
            },
        )
        return self.finalise(result)

    async def handle_create(self, user_message: str) -> str:
        args = self.extract_create_args(user_message)
        if not args:
            return "I couldn’t extract the event details. Please restate with title, start time, and end time."

        result = await self.call_mcp("create_event", args)

        # === FIXED ERROR CHECK ===
        if result.isError == True:
            return f"The calendar rejected the event: {result}"

        return self.finalise(result)


    async def handle_update(self, user_message: str) -> str:
        args = self.extract_update_args(user_message)
        if not args:
            return "I couldn’t understand which event to update. Please include the event ID and new times."

        event_id = args.get("event_id")
        start_dt = args.get("start_datetime")
        end_dt = args.get("end_datetime")

        if not event_id:
            return "To update an event, I need its event ID. Please tell me the event ID you want to change."
        if not start_dt or not end_dt:
            return "Please specify the new start and end times for the event."

        result = await self.call_mcp("update_event", args)

        # === FIXED ERROR CHECK ===
        if result.isError == True:
            return f"The calendar rejected the update: {result}"

        return self.finalise(result)

    # ---------------------------------------------------------
    # MAIN AGENT LOOP (thin, v2-style)
    # ---------------------------------------------------------
    @tracer.agent()
    async def run_turn(self, user_message: str) -> str:
        self.remember("user", user_message)
        plan = self.plan(user_message)

        # ASK
        if plan.startswith("PLAN:ASK"):
            q = plan[len("PLAN:ASK "):]
            self.remember("assistant", q)
            return q

        # LIST
        if plan.startswith("PLAN:LIST"):
            reply = await self.handle_list(user_message)
            self.remember("assistant", reply)
            return reply

        # CREATE
        if plan.startswith("PLAN:CREATE"):
            reply = await self.handle_create(user_message)
            self.remember("assistant", reply)
            return reply

        # UPDATE
        if plan.startswith("PLAN:UPDATE"):
            reply = await self.handle_update(user_message)
            self.remember("assistant", reply)
            return reply

        # UNKNOWN
        fallback = "I didn’t understand what to do with your request."
        self.remember("assistant", fallback)
        return fallback


# ============================================================
# REPL
# ============================================================
async def main():
    agent = CalendarAgent()
    print("Calendar Agent v3 ready.")

    while True:
        msg = input("You: ").strip()
        if not msg:
            continue
        reply = await agent.run_turn(msg)
        print("Agent:", reply)

if __name__ == "__main__":
    asyncio.run(main())
