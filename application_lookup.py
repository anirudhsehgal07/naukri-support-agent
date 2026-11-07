
from dataset import generate_dataset

# Generate the same deterministic dataset used in Task 1
JOB_APPLICATIONS = generate_dataset()

# Configuration
MAX_AGE_DAYS = 30
ESCALATION_THRESHOLD = 0.60


def check_job_application_status(record_id: str) -> dict:
    """
    Find an application and return its status,
    expected salary, and escalation score.
    """

    # Find the application using its record ID
    application = next(
        (
            record for record in JOB_APPLICATIONS
            if record["record_id"] == record_id
        ),
        None
    )

    # Handle an unknown application ID
    if application is None:
        return {
            "record_id": record_id,
            "error": "Application record not found."
        }

    # Convert the priority flag into a number
    priority_score = (
        1 if application["flagged_priority_review"] else 0
    )

    # Newer applications receive a higher recency score
    recency_score = (
        MAX_AGE_DAYS - application["days_since_created"]
    ) / MAX_AGE_DAYS

    # Calculate the combined escalation score
    escalation_score = (
        0.60 * priority_score
        + 0.40 * recency_score
    )

    # Decide whether escalation is required
    escalation_required = (
        escalation_score >= ESCALATION_THRESHOLD
    )

    return {
        "record_id": application["record_id"],
        "status": application["status"],
        "expected_salary_inr": application["expected_salary_inr"],
        "escalation_score": round(escalation_score, 3),
        "escalation_required": escalation_required
    }


# Run tests when this file is executed directly
if __name__ == "__main__":

    # Test a valid application ID
    print("VALID APPLICATION:")
    print(check_job_application_status("APP0001"))

    # Test an invalid application ID
    print("\nINVALID APPLICATION:")
    print(check_job_application_status("APP9999"))

    # Analyse escalation scores for all applications
    print("\nESCALATION SCORE ANALYSIS:")

    for application in JOB_APPLICATIONS:

        result = check_job_application_status(
            application["record_id"]
        )

        print(
            application["record_id"],
            "Priority:", application["flagged_priority_review"],
            "Age:", application["days_since_created"],
            "Score:", result["escalation_score"],
            "Escalate:", result["escalation_required"]
        )
