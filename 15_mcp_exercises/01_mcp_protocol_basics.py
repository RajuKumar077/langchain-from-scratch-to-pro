"""
Lesson 01: MCP Protocol Fundamentals (Direct ClientSession)
===========================================================
This lesson demonstrates how Model Context Protocol (MCP) works at the
wire protocol level using the official Python `mcp` SDK directly.

Key Concepts Learned:
1. `StdioServerParameters`: Defines how the client launches the MCP server process.
2. `stdio_client`: Manages stdin/stdout pipes for JSON-RPC 2.0 communication.
3. `ClientSession`: The client protocol handler.
4. Protocol Handshake: `await session.initialize()`.
5. Discovering Capabilities:
   - `session.list_tools()`: Functions the server can execute.
   - `session.list_resources()`: Documents/data the server can read.
   - `session.list_prompts()`: Prompt templates the server provides.
6. Direct Invocation:
   - `session.call_tool(...)`
   - `session.read_resource(...)`
   - `session.get_prompt(...)`

No LLMs are involved yet—this is pure protocol exploration!
"""

import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import central environment setup
import config

async def run_mcp_protocol_demo():
    print("=" * 65)
    print(" Lesson 01: Model Context Protocol (MCP) Raw Client Session")
    print("=" * 65)

    server_script = Path(config.SERVERS_DIR) / "math_weather_server.py"
    if not server_script.exists():
        raise FileNotFoundError(f"Server script not found at {server_script}")

    # 1. Define how to launch the MCP server as a subprocess
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script.resolve())],
        env=None,
    )

    print(f"[1] Spawning MCP Server Subprocess:")
    print(f"    Command: {server_params.command}")
    print(f"    Script : {server_script.name}")
    print()

    # 2. Establish stdio connection and start MCP ClientSession
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 3. Protocol handshake: exchange capabilities and version
            print("[2] Initializing MCP Session Handshake...")
            init_result = await session.initialize()
            print(f"    Protocol Version : {init_result.protocolVersion}")
            print(f"    Server Name      : {init_result.serverInfo.name}")
            print(f"    Server Version   : {init_result.serverInfo.version}")
            print()

            # 4. Discover exposed Tools
            print("[3] Listing Server Tools:")
            tools_response = await session.list_tools()
            for idx, tool in enumerate(tools_response.tools, 1):
                print(f"    {idx}. {tool.name}")
                print(f"       Description : {tool.description}")
                print(f"       Parameters  : {list(tool.inputSchema.get('properties', {}).keys())}")
            print()

            # 5. Discover exposed Resources
            print("[4] Listing Server Resources:")
            resources_response = await session.list_resources()
            for idx, res in enumerate(resources_response.resources, 1):
                print(f"    {idx}. URI  : {res.uri}")
                print(f"       Name : {res.name}")
                print(f"       Desc : {res.description}")
            print()

            # 6. Discover exposed Prompts
            print("[5] Listing Server Prompts:")
            prompts_response = await session.list_prompts()
            for idx, prompt in enumerate(prompts_response.prompts, 1):
                print(f"    {idx}. Name : {prompt.name}")
                print(f"       Desc : {prompt.description}")
            print()

            # 7. Execute a Tool via MCP Protocol
            print("[6] Calling Tool: 'add' with args {'a': 45, 'b': 55}...")
            call_res = await session.call_tool("add", arguments={"a": 45, "b": 55})
            print(f"    Tool Result Content : {[c.text for c in call_res.content]}")
            print()

            print("[7] Calling Tool: 'get_weather' with args {'city': 'Tokyo'}...")
            weather_res = await session.call_tool("get_weather", arguments={"city": "Tokyo"})
            print(f"    Tool Result Content : {[c.text for c in weather_res.content]}")
            print()

            # 8. Read a Resource via MCP Protocol
            print("[8] Reading Resource: 'company://policies/travel'...")
            resource_data = await session.read_resource("company://policies/travel")
            for item in resource_data.contents:
                print(f"    Resource Content:\n{item.text.strip()}")
            print()

            # 9. Get a Prompt Template via MCP Protocol
            print("[9] Getting Prompt: 'weather_assistant' for 'Paris'...")
            prompt_data = await session.get_prompt("weather_assistant", arguments={"city": "Paris"})
            for msg in prompt_data.messages:
                print(f"    Prompt Message [{msg.role}]: {msg.content.text}")

    print()
    print("=" * 65)
    print(" [SUCCESS] Lesson 01 completed successfully!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_mcp_protocol_demo())
