
from langgraph.checkpoint.sqlite import SqliteSaver
from agent_workflow import workflow


def main():

    database_file = "checkpoints.sqlite"

    # Same thread ID is used for pause and resume
    config = {
        "configurable": {
            "thread_id": "task15-demo-001"
        }
    }

    with SqliteSaver.from_conn_string(
        database_file
    ) as checkpointer:

        print("\n========== STEP 1: START WORKFLOW ==========")

        # Compile graph to pause before response_node
        paused_app = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["response_node"]
        )

        # Execute workflow
        first_result = paused_app.invoke(
            {
                "question": "How can I reschedule an interview?",
                "history": []
            },
            config=config
        )

        print("\nWorkflow paused before response_node.")

        # Read saved checkpoint
        saved_state = paused_app.get_state(config)

        print("Saved thread ID: task15-demo-001")
        print("Completed node results:")
        print("Intent:", saved_state.values.get("intent"))
        print("RAG result saved:", "result" in saved_state.values)
        print("Next node:", saved_state.next)

        print("\n========== STEP 2: RESUME WORKFLOW ==========")

        # Compile normal graph using the SAME checkpointer
        resumed_app = workflow.compile(
            checkpointer=checkpointer
        )

        # Resume with None input and the same thread ID
        final_result = resumed_app.invoke(
            None,
            config=config
        )

        print("\nWorkflow resumed successfully.")

        print("\nFinal response:")
        print(final_result.get("response"))

        print(
            "\nValidation status:",
            final_result.get("validation_status")
        )


if __name__ == "__main__":
    main()
