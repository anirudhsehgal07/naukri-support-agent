
from fastmcp import FastMCP
from application_lookup import check_job_application_status

# Create the MCP server
mcp = FastMCP("Naukri Support Server")


@mcp.tool()
def get_application_status(record_id: str) -> dict:
    """
    Check a job application's status using its application record ID.
    """
    return check_job_application_status(record_id.upper())


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001
    )
