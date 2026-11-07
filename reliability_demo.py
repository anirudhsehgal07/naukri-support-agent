
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# ==========================================
# SHARED WORKFLOW STATE
# ==========================================

class DemoState(TypedDict, total=False):
    message: str
    result: str


# ==========================================
# (A) RETRY WITH EXPONENTIAL BACKOFF
# ==========================================

retry_counter = {"attempts": 0}


async def unreliable_node(state: DemoState) -> dict:

    retry_counter["attempts"] += 1
    attempt = retry_counter["attempts"]

    print(f"Unreliable node: attempt {attempt}")

    # Simulate temporary failures on first 2 calls
    if attempt <= 2:
        print("Temporary failure!")
        raise ConnectionError("Simulated temporary failure")

    print("Node recovered successfully!")

    return {
        "result": "Retry operation succeeded on attempt 3"
    }


def build_retry_graph():

    builder = StateGraph(DemoState)

    # Explicit exponential-backoff retry configuration
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=4.0,
        jitter=True,
        retry_on=ConnectionError
    )

    builder.add_node(
        "unreliable_node",
        unreliable_node,
        retry_policy=retry_policy
    )

    builder.add_edge(START, "unreliable_node")
    builder.add_edge("unreliable_node", END)

    return builder.compile()


async def retry_demo():

    print("\n======================================")
    print("A. RETRY DEMONSTRATION")
    print("======================================")

    retry_counter["attempts"] = 0

    app = build_retry_graph()

    try:
        result = await app.ainvoke({
            "message": "Test retry recovery"
        })

        print("\nFinal result:", result["result"])
        print("Total attempts:", retry_counter["attempts"])

        if retry_counter["attempts"] == 3:
            print("PASS: Retry recovered within 3 attempts.")

    except Exception as error:
        print("Retry demonstration failed:", repr(error))


# ==========================================
# (B) PER-NODE TIMEOUT
# ==========================================

async def slow_node(state: DemoState) -> dict:

    print("Slow node started...")
    
    # Simulate a call taking 3 seconds
    await asyncio.sleep(3)

    return {
        "result": "Slow node completed"
    }


def build_node_timeout_graph():

    builder = StateGraph(DemoState)

    builder.add_node(
        "slow_node",
        slow_node,
        timeout=1.0
    )

    builder.add_edge(START, "slow_node")
    builder.add_edge("slow_node", END)

    return builder.compile()


async def node_timeout_demo():

    print("\n======================================")
    print("B. PER-NODE TIMEOUT DEMONSTRATION")
    print("======================================")

    app = build_node_timeout_graph()

    try:
        result = await app.ainvoke({
            "message": "Test node timeout"
        })

        print("Unexpected result:", result)

    except Exception as error:

        print("\nNode stopped cleanly.")
        print("Error type:", type(error).__name__)
        print("Error details:", str(error))

        if "timeout" in type(error).__name__.lower():
            print("PASS: Per-node timeout fired correctly.")
        else:
            print("Check the error and LangGraph version.")


# ==========================================
# (C) GLOBAL WORKFLOW TIMEOUT
# ==========================================

async def workflow_node_one(state: DemoState) -> dict:

    print("Global workflow: node 1 started.")

    await asyncio.sleep(1)

    print("Global workflow: node 1 completed.")

    return {}


async def workflow_node_two(state: DemoState) -> dict:

    print("Global workflow: node 2 started.")

    await asyncio.sleep(3)

    print("Global workflow: node 2 completed.")

    return {}


def build_global_timeout_graph():

    builder = StateGraph(DemoState)

    builder.add_node("node_one", workflow_node_one)
    builder.add_node("node_two", workflow_node_two)

    builder.add_edge(START, "node_one")
    builder.add_edge("node_one", "node_two")
    builder.add_edge("node_two", END)

    return builder.compile()


async def global_timeout_demo():

    print("\n======================================")
    print("C. GLOBAL TIMEOUT DEMONSTRATION")
    print("======================================")

    app = build_global_timeout_graph()

    try:

        # Overall graph execution limit: 1.5 seconds
        result = await asyncio.wait_for(
            app.ainvoke({
                "message": "Test global timeout"
            }),
            timeout=1.5
        )

        print("Unexpected result:", result)

    except asyncio.TimeoutError:

        print("\nGlobal timeout fired!")
        print("The complete workflow was cancelled.")
        print("PASS: Global timeout worked correctly.")


# ==========================================
# MAIN FUNCTION
# ==========================================

async def main():

    await retry_demo()

    await node_timeout_demo()

    await global_timeout_demo()

    print("\n======================================")
    print("ALL RELIABILITY TESTS FINISHED")
    print("======================================")


if __name__ == "__main__":
    asyncio.run(main())
