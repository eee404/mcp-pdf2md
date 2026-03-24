from .server import mcp


def main():
    """PDF to Markdown Conversion Service - Provides MCP service for converting PDF files to Markdown"""

    # Check API key
    from .server import MISTRAL_API_KEY
    if not MISTRAL_API_KEY:
        print("Warning: API key not set, please set the MISTRAL_API_KEY environment variable")

    # Run MCP server
    mcp.run()

__all__ = ['main']
