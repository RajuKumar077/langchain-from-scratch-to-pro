"""
Remote / Online MCP Server: Financial Services & Market Data (SSE)
==================================================================
This server runs as a standalone network service over HTTP/SSE (Server-Sent Events).
It simulates a real-world online/remote microservice providing financial market tools.

Exposed Tools:
1. `convert_currency(amount, from_curr, to_curr)`: Real-time FX currency converter.
2. `get_stock_quote(symbol)`: Real-time stock quote lookup (AAPL, NVDA, MSFT, etc.).
3. `calculate_compound_interest(principal, rate_pct, years)`: Investment calculation.

Exposed Resource:
- `market://crypto/snapshot`: Live crypto snapshot (BTC, ETH, SOL).

Usage:
  Run this server in a separate terminal or as a background service:
      python online_mock_server.py --port 8080
  Access SSE endpoint at:
      http://127.0.0.1:8080/sse
"""

import sys
import os
from mcp.server.fastmcp import FastMCP

# Default port
port = int(os.getenv("MCP_SSE_PORT", "8080"))
host = os.getenv("MCP_SSE_HOST", "127.0.0.1")

if "--port" in sys.argv:
    idx = sys.argv.index("--port")
    if idx + 1 < len(sys.argv):
        port = int(sys.argv[idx + 1])

# Create FastMCP instance configured for network hosting
mcp = FastMCP(
    name="OnlineFinanceServer",
    host=host,
    port=port,
)

# ==============================================================================
# Financial Market Tools
# ==============================================================================

@mcp.tool()
def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
    """Convert an amount from one fiat currency to another using current exchange rates.
    
    Args:
        amount: Numerical amount to convert.
        from_curr: Source 3-letter currency code (e.g. USD, EUR, GBP, JPY, INR).
        to_curr: Target 3-letter currency code (e.g. USD, EUR, GBP, JPY, INR).
    """
    rates_to_usd = {
        "usd": 1.0,
        "eur": 1.08,
        "gbp": 1.28,
        "jpy": 0.0067,
        "inr": 0.012,
    }
    src = from_curr.strip().lower()
    tgt = to_curr.strip().lower()

    if src not in rates_to_usd:
        return f"Error: Unsupported source currency '{from_curr}'. Supported: USD, EUR, GBP, JPY, INR."
    if tgt not in rates_to_usd:
        return f"Error: Unsupported target currency '{to_curr}'. Supported: USD, EUR, GBP, JPY, INR."

    # Convert to USD first, then to target
    amount_in_usd = amount * rates_to_usd[src]
    converted = amount_in_usd / rates_to_usd[tgt]
    return f"{amount:,.2f} {src.upper()} = {converted:,.2f} {tgt.upper()} (Spot rate: 1 {src.upper()} = {(rates_to_usd[src]/rates_to_usd[tgt]):.4f} {tgt.upper()})"

@mcp.tool()
def get_stock_quote(symbol: str) -> str:
    """Get the current stock price, day change, and 52-week range for a given ticker symbol.
    
    Args:
        symbol: Ticker symbol (e.g. AAPL, MSFT, NVDA, GOOGL, TSLA, AMZN).
    """
    mock_quotes = {
        "AAPL": {"price": 234.50, "change": "+1.25%", "volume": "48.2M", "high52": 237.23, "low52": 164.08},
        "MSFT": {"price": 448.10, "change": "+0.85%", "volume": "22.1M", "high52": 468.35, "low52": 309.45},
        "NVDA": {"price": 138.25, "change": "+3.40%", "volume": "75.8M", "high52": 140.76, "low52": 39.23},
        "GOOGL": {"price": 182.40, "change": "-0.32%", "volume": "18.9M", "high52": 191.75, "low52": 115.83},
        "TSLA": {"price": 220.70, "change": "+2.15%", "volume": "62.4M", "high52": 271.00, "low52": 138.80},
    }
    sym = symbol.strip().upper()
    quote = mock_quotes.get(sym)
    if not quote:
        return f"Quote for '{sym}': Price $150.00 (Estimated), Change +0.00%, Volume 5.0M."

    return (
        f"Stock Quote for {sym}:\n"
        f" - Price:       ${quote['price']:.2f}\n"
        f" - Change:      {quote['change']}\n"
        f" - Volume:      {quote['volume']}\n"
        f" - 52-Wk Range: ${quote['low52']:.2f} - ${quote['high52']:.2f}"
    )

@mcp.tool()
def calculate_compound_interest(principal: float, annual_rate_pct: float, years: int) -> str:
    """Calculate compound interest and final future investment balance.
    
    Args:
        principal: Initial invested balance (e.g. 10000).
        annual_rate_pct: Annual interest percentage (e.g. 7.5 for 7.5%).
        years: Investment duration in years.
    """
    rate = annual_rate_pct / 100.0
    future_value = principal * ((1.0 + rate) ** years)
    interest_earned = future_value - principal
    return (
        f"Compound Interest Projection ({years} years @ {annual_rate_pct}%):\n"
        f" - Initial Principal : ${principal:,.2f}\n"
        f" - Interest Earned   : ${interest_earned:,.2f}\n"
        f" - Final Total Value : ${future_value:,.2f}"
    )

# ==============================================================================
# Financial Market Resource
# ==============================================================================

@mcp.resource("market://crypto/snapshot")
def get_crypto_snapshot() -> str:
    """Live crypto market benchmark prices."""
    return """# Cryptocurrency Market Snapshot
- Bitcoin (BTC):  $67,450.00 (+2.1%)
- Ethereum (ETH): $2,640.50 (+1.4%)
- Solana (SOL):   $155.20 (+4.8%)
Updated: Real-time simulated feed
"""

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    print(f"[OnlineFinanceServer] Starting Remote SSE MCP Server on http://{host}:{port}/sse ...", file=sys.stderr)
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="sse")
