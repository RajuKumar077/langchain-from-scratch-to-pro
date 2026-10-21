"""
Local MCP Server: Math, Weather & System Tools
==============================================
This server is built using Anthropic's official `FastMCP` framework.
It exposes:
1. Tools:
   - `add(a, b)`: Adds two numbers.
   - `multiply(a, b)`: Multiplies two numbers.
   - `get_weather(city)`: Simulates real-time weather information for any city.
   - `get_system_time()`: Retrieves current system time, UTC offset, and day.
2. Resources:
   - `company://policies/travel`: Corporate travel and expense policy document.
3. Prompts:
   - `weather_assistant(city)`: Pre-packaged prompt template.

Usage:
  - stdio mode (default when spawned by client):
      python math_weather_server.py
  - SSE web mode:
      python math_weather_server.py --transport sse --port 8000
"""

import sys
import datetime
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("MathWeatherServer")

# ==============================================================================
# 1. MCP Tools (Callable Functions)
# ==============================================================================

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers together. Use for exact summation."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers together. Use for exact multiplication."""
    return a * b

@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather conditions, temperature in Celsius, and humidity for a specified city.
    
    Args:
        city: The name of the city (e.g. 'Tokyo', 'London', 'New York', 'Paris', 'Bengaluru').
    """
    mock_weather = {
        "tokyo": "Sunny, 22°C (71.6°F), Humidity: 45%, Wind: 8 km/h NW",
        "london": "Light Rain, 14°C (57.2°F), Humidity: 82%, Wind: 15 km/h SW",
        "new york": "Partly Cloudy, 18°C (64.4°F), Humidity: 55%, Wind: 12 km/h E",
        "paris": "Clear, 19°C (66.2°F), Humidity: 50%, Wind: 10 km/h NE",
        "bengaluru": "Pleasant, 24°C (75.2°F), Humidity: 60%, Wind: 7 km/h SE",
        "san francisco": "Foggy, 16°C (60.8°F), Humidity: 78%, Wind: 18 km/h W",
    }
    key = city.strip().lower()
    info = mock_weather.get(key, f"Partly cloudy, 20°C (68°F), Humidity: 50% (Standard estimate for {city})")
    return f"Weather report for {city.title()}: {info}"

@mcp.tool()
def get_system_time() -> str:
    """Get the current system local time, UTC time, and ISO timestamp."""
    now = datetime.datetime.now()
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    return (
        f"Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.astimezone().tzname()})\n"
        f"UTC Time:   {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Day:        {now.strftime('%A')}"
    )

# ==============================================================================
# 2. MCP Resources (Static or Dynamic Documents)
# ==============================================================================

@mcp.resource("company://policies/travel")
def get_travel_policy() -> str:
    """Corporate travel, lodging, and meal expense reimbursement policy."""
    return """# Corporate Travel & Expense Policy (v2026.1)

1. Flight Booking: Economy class mandatory for domestic flights under 6 hours.
2. Lodging Allowance: Up to $250/night in tier-1 metro areas (Tokyo, London, NYC, SF).
3. Per Diem Meals: $75/day standard allowance with receipts required for purchases over $25.
4. Currency & Approvals: All foreign currency receipts must be converted at spot rate on the date of transaction.
5. Approvals: Pre-approval from department VP required for international travel.
"""

# ==============================================================================
# 3. MCP Prompts (Reusable Prompt Templates)
# ==============================================================================

@mcp.prompt()
def weather_assistant(city: str) -> str:
    """Generate a prompt instructing the LLM on how to deliver a weather briefing."""
    return (
        f"You are a friendly local travel advisor. Please look up the current weather "
        f"for {city} using the get_weather tool, and provide 3 tailored packing tips "
        f"based on the temperature and conditions."
    )

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Check if run with --transport sse
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport_type = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "sse"
    else:
        transport_type = "stdio"

    port = 8000
    if "--port" in sys.argv:
        p_idx = sys.argv.index("--port")
        if p_idx + 1 < len(sys.argv):
            port = int(sys.argv[p_idx + 1])

    if transport_type == "sse":
        print(f"[MathWeatherServer] Starting FastMCP Server on SSE http://127.0.0.1:{port}/sse ...", file=sys.stderr)
        mcp.settings.port = port
        mcp.settings.host = "127.0.0.1"
        mcp.run(transport="sse")
    else:
        # Default: stdio transport for subprocess IPC
        mcp.run(transport="stdio")
