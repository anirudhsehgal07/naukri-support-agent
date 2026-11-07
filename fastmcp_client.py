

import asyncio
from fastmcp import Client


async def main():

    # Connect to the MCP server using HTTP
    async with Client("http://127.0.0.1:8001/mcp") as client:

        application_ids = ["APP0001", "APP0002"]

        for record_id in application_ids:

            print(f"\nChecking application: {record_id}")

            result = await client.call_tool(
                "get_application_status",
                {"record_id": record_id}
            )

            print(result)


if __name__ == "__main__":
    asyncio.run(main())
