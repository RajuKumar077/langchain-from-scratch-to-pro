"""
Lesson 02: LangChain MCP Adapters (Bridging MCP to LangChain BaseTool)
======================================================================
In this lesson, we introduce `langchain-mcp-adapters`, the official bridge
between the Model Context Protocol (MCP) and LangChain.

Key Concepts Learned:
1. `MultiServerMCPClient`: High-level client managing multiple MCP server connections.
2. `StdioConnection`: Configuration dict declaring command, args, and transport='stdio'.
3. Automatic Tool Conversion: `await client.get_tools()` converts remote MCP tools into
   native LangChain `BaseTool` objects.
4. Schema & Metadata Inspection: Inspecting `.name`, `.description`, and `.args`.
5. Native LangChain Invocations: Calling `await tool.ainvoke(...)` directly.
"""

import asyncio
import sys
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Environment configuration
import config

async def run_langchain_mcp_adapters_demo():
    print("=" * 65)
    print(" Lesson 02: LangChain MCP Adapters & MultiServerMCPClient")
    print("=" * 65)

    server_script = Path(config.SERVERS_DIR) / "math_weather_server.py"

    # 1. Define MultiServerMCPClient configuration
    client = MultiServerMCPClient({
        "local_math_weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_script.resolve())],
        }
    })

    print("[1] Connecting to MCP Server via MultiServerMCPClient...")
    
    # 2. Automatically load all tools as LangChain BaseTool instances
    tools: list[BaseTool] = await client.get_tools()
    print(f"[2] Successfully loaded {len(tools)} LangChain BaseTools:\n")

    # 3. Inspect each tool's LangChain metadata
    for idx, tool in enumerate(tools, 1):
        print(f"    Tool {idx}:")
        print(f"      Name        : {tool.name}")
        print(f"      Type        : {type(tool).__name__} (isinstance BaseTool: {isinstance(tool, BaseTool)})")
        print(f"      Description : {tool.description.strip().splitlines()[0]}")
        print(f"      Arguments   : {tool.args}")
        print("-" * 50)
    print()

    # 4. Map tools by name for easy execution
    tool_map = {t.name: t for t in tools}

    # 5. Execute tools directly using LangChain's ainvoke()
    print("[3] Testing Direct Invocations via LangChain BaseTool.ainvoke():")
    
    # Test 'add'
    add_tool = tool_map["add"]
    add_input = {"a": 128, "b": 256}
    print(f"    Executing {add_tool.name}{add_input}...")
    add_result = await add_tool.ainvoke(add_input)
    print(f"    Result: {add_result}\n")

    # Test 'multiply'
    mult_tool = tool_map["multiply"]
    mult_input = {"a": 15, "b": 12}
    print(f"    Executing {mult_tool.name}{mult_input}...")
    mult_result = await mult_tool.ainvoke(mult_input)
    print(f"    Result: {mult_result}\n")

    # Test 'get_weather'
    weather_tool = tool_map["get_weather"]
    weather_input = {"city": "London"}
    print(f"    Executing {weather_tool.name}{weather_input}...")
    weather_result = await weather_tool.ainvoke(weather_input)
    print(f"    Result: {weather_result}\n")

    # Test 'get_system_time'
    time_tool = tool_map["get_system_time"]
    print(f"    Executing {time_tool.name}{{}}...")
    time_result = await time_tool.ainvoke({})
    print(f"    Result:\n{time_result}\n")

    print("=" * 65)
    print(" [SUCCESS] Lesson 02 completed successfully!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_langchain_mcp_adapters_demo())
