
import random
from collections import Counter

# -------------------------------
# 1. Dataset configuration
# -------------------------------

TOTAL_RECORDS = 50
DATASET_SEED = 2026
PRIORITY_SEED = 42

CATEGORIES = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate"
]

STATUSES = [
    "Applied",
    "Screening",
    "Interview Scheduled",
    "Offered",
    "Rejected"
]

# Annual expected salary ranges in INR
# These ranges are role-specific.

SALARY_RANGES = {
    "Software Engineer": (400000, 1800000),
    "Data Analyst": (350000, 1200000),
    "Product Manager": (700000, 2000000),
    "HR Executive": (250000, 800000),
    "Sales Associate": (240000, 900000)
}


# -------------------------------
# 2. Generate job applications
# -------------------------------

def generate_dataset():

    rng = random.Random(DATASET_SEED)

    # Separate seeded generator for priority flags
    priority_rng = random.Random(PRIORITY_SEED)

    # Every category gets 10 records
    category_list = CATEGORIES * 10

    # Every status gets 10 records
    status_list = STATUSES * 10

    rng.shuffle(category_list)
    rng.shuffle(status_list)

    # Generate priority flags randomly
    priority_flags = [
        priority_rng.random() < 0.20
        for _ in range(TOTAL_RECORDS)
    ]

    JOB_APPLICATIONS = []

    for i in range(TOTAL_RECORDS):

        category = category_list[i]

        min_salary, max_salary = SALARY_RANGES[category]

        application = {
            "record_id": f"APP{i + 1:04d}",

            "category": category,

            "status": status_list[i],

            "expected_salary_inr": rng.randrange(
                min_salary, max_salary + 1, 1000
            ),

            "days_since_created": rng.randint(0, 30),

            "flagged_priority_review": priority_flags[i]
        }

        JOB_APPLICATIONS.append(application)

    return JOB_APPLICATIONS


# -------------------------------
# 3. Validate dataset
# -------------------------------

def validate_dataset(JOB_APPLICATIONS):

    assert len(JOB_APPLICATIONS) >= 40

    category_counts = Counter(
        record["category"] for record in JOB_APPLICATIONS
    )

    status_counts = Counter(
        record["status"] for record in JOB_APPLICATIONS
    )

    # Check required categories
    for category in CATEGORIES:
        assert category_counts[category] >= 3

    # Check required statuses
    for status in STATUSES:
        assert status_counts[status] >= 1

    # Check record fields and values
    for record in JOB_APPLICATIONS:

        assert set(record.keys()) == {
            "record_id",
            "category",
            "status",
            "expected_salary_inr",
            "days_since_created",
            "flagged_priority_review"
        }

        assert 0 <= record["days_since_created"] <= 30

        assert isinstance(record["flagged_priority_review"], bool)

        assert record["expected_salary_inr"] > 0

    # Calculate priority percentage
    priority_count = sum(
        record["flagged_priority_review"]
        for record in JOB_APPLICATIONS
    )

    priority_percentage = (
        priority_count / len(JOB_APPLICATIONS)
    ) * 100

    assert 10 <= priority_percentage <= 30, (
        "Priority percentage is outside the required range."
    )

    return category_counts, status_counts, priority_percentage


# -------------------------------
# 4. Main execution
# -------------------------------

if __name__ == "__main__":

    JOB_APPLICATIONS = generate_dataset()

    category_counts, status_counts, priority_percentage = (
        validate_dataset(JOB_APPLICATIONS)
    )

    print("\n===== DATASET SUMMARY =====")

    print("Total Records:", len(JOB_APPLICATIONS))

    print("\nCategory-wise Count:")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")

    print("\nStatus-wise Count:")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    print("\nPriority Review Percentage:")
    print(f"{priority_percentage:.2f}%")

    print("\nFirst 5 Job Application Records:")
    for record in JOB_APPLICATIONS[:5]:
        print(record)

    print("\nDataset validation successful!")
