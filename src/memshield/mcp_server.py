"""MemShield MCP server — exposes audit inspection tools to MCP clients."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def create_server(audit_db: str, knowledge_base_id: str = "default") -> Any:
    """Create and return the MCP server instance."""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError(
            "mcp is required for the MCP server. Install with: pip install memshield[mcp]"
        ) from None

    from memshield.audit.config import AuditConfig
    from memshield.audit.log import AuditLog

    config = AuditConfig(store=audit_db, knowledge_base_id=knowledge_base_id, tsa_url=None)
    log = AuditLog(config)
    server = Server("memshield")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="audit_inspect",
                description="Inspect a single MemShield audit record by inference ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "inference_id": {
                            "type": "string",
                            "description": "The inference ID to inspect",
                        }
                    },
                    "required": ["inference_id"],
                },
            ),
            Tool(
                name="audit_verify",
                description="Verify the integrity of the MemShield audit chain",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="audit_export",
                description="Export audit records, optionally filtered by date",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "from_date": {
                            "type": "string",
                            "description": "ISO 8601 date (optional)",
                        }
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "audit_inspect":
            record = log.get_record(arguments["inference_id"])
            if record is None:
                return [
                    TextContent(
                        type="text",
                        text=f"No record found for inference_id={arguments['inference_id']}",
                    )
                ]
            return [TextContent(type="text", text=json.dumps(record.to_dict(), indent=2))]

        elif name == "audit_verify":
            is_valid, errors = log.verify_chain()
            result = {"valid": is_valid, "errors": errors}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "audit_export":
            records = log.export(from_date=arguments.get("from_date"))
            return [TextContent(type="text", text=json.dumps(records, indent=2))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server, log


def main() -> None:
    """Entry point for `memshield mcp` CLI command."""
    parser = argparse.ArgumentParser(description="MemShield MCP server")
    parser.add_argument("--audit-db", required=True, help="Path to audit.db")
    parser.add_argument("--knowledge-base-id", default="default")
    args = parser.parse_args()

    try:
        from mcp.server.stdio import stdio_server
        import asyncio
    except ImportError:
        print(
            "mcp package required. Install with: pip install memshield[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    server, _ = create_server(args.audit_db, args.knowledge_base_id)

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
