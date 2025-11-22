import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


SERVER_URL = "http://127.0.0.1:8020/sse"


async def main():
    print("Connecting to MCP server at", SERVER_URL)

    async with sse_client(url=SERVER_URL) as streams:
        async with ClientSession(*streams) as session:
            # Handshake
            init_result = await session.initialize()
            print("Initialized. Server capabilities:", init_result.capabilities)

            # List tools exposed by the server
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print("Tools available:", tool_names)


if __name__ == "__main__":
    asyncio.run(main())
