import asyncio

from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def main():
    # 1. Connect to MCP server
    mcp = BasicMCPClient("http://127.0.0.1:8020/sse")

    # 2. Get MCP tools
    tool_spec = McpToolSpec(client=mcp)
    tools = await tool_spec.to_tool_list_async()

    # 3. LLM (cheapest)
    llm = OpenAI(model="gpt-5-nano")

    # 4. Direct tool call (0.14.8 has no agents)
    for tool in tools:
        if tool.metadata.name == "list_events":
            result = await tool.acall(date_start="today")

            print(result)


asyncio.run(main())
