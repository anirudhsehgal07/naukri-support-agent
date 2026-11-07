
from pathlib import Path
import re

KNOWLEDGE_BASE = {
    "eligibility.txt": """
Candidates must meet the education, skills, and experience requirements mentioned in the job description. Recruiters evaluate applicants based on relevant qualifications, professional experience, and required skills. Candidates should provide accurate information in their resumes and applications.
""",

    "interview_scheduling.txt": """
Candidates who successfully complete the screening stage may be invited for an interview. The recruiter contacts candidates through their registered email address or phone number with the interview date, time, format, and other instructions. Candidates should confirm their availability or request rescheduling in advance.
""",

    "offer_negotiation.txt": """
Selected candidates receive an offer containing details about the role, compensation, benefits, and joining conditions. Candidates may discuss salary expectations and other employment terms with the recruiter. Any changes to the offer must be approved by the authorized hiring team.
""",

    "background_verification.txt": """
Background verification may be conducted for selected candidates before their employment is finalized. The process may include verification of education, previous employment, identity, and professional references. Candidates must provide genuine documents, and discrepancies may require additional review.
""",

    "notice_period.txt": """
Candidates should disclose their current employment notice period during the recruitment process. The expected joining date is discussed with the recruiter while considering existing employment commitments. Candidates may discuss early release or notice-period reduction with their current employer.
""",

    "referral_bonus.txt": """
Employees may refer qualified candidates for open positions through the approved referral process. A referral bonus may be provided when a referred candidate is hired and completes the required minimum employment period. Employees must follow the organization's referral guidelines to qualify.
""",

    "internal_transfer.txt": """
Existing employees may apply for internal job opportunities when they meet the eligibility requirements. Internal applications are evaluated based on skills, performance, experience, business requirements, and necessary manager approvals. Employees should discuss their interest with their manager and follow the internal application process.
""",

    "probation.txt": """
New employees may be placed on a probation period when they join the organization. During probation, the organization evaluates work quality, attendance, professional conduct, and role responsibilities. Based on the evaluation, employment may be confirmed, probation may be extended, or other actions may be taken according to employment terms.
""",

    "remote_work.txt": """
Remote-work eligibility depends on the job role, team requirements, and management approval. Employees working remotely must maintain productivity, attend required meetings, communicate effectively, and follow information-security policies. Remote-work arrangements may change according to business requirements and organizational policies.
""",

    "diversity_hiring.txt": """
The recruitment process aims to provide equal employment opportunities to candidates from different backgrounds. Hiring decisions should be based on job-related qualifications, relevant experience, and required skills rather than unrelated personal characteristics. Recruiters should follow fair evaluation practices and avoid discrimination.
""",

    "exit_interview.txt": """
Employees leaving the organization may be invited to participate in an exit interview. The discussion may cover their employment experience, reasons for leaving, feedback, and suggestions for improvement. Feedback is handled confidentially and used by authorized teams to improve workplace practices.
""",

    "data_retention.txt": """
Applicant information is stored securely and accessed only by authorized personnel. Candidate records are retained according to the organization's data-retention schedule and applicable privacy requirements. Information should be securely deleted or anonymized after the retention period unless a valid reason requires continued storage.
"""
}


BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"


def validate_document(filename, content):
    sentences = [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', content.strip())
        if sentence.strip()
    ]

    count = len(sentences)

    if not 2 <= count <= 5:
        raise ValueError(
            f"{filename} has {count} sentences. "
            "Each document must contain 2-5 sentences."
        )

    return count


def create_knowledge_base():
    KB_DIR.mkdir(exist_ok=True)

    print("\nCREATING RECRUITMENT KNOWLEDGE BASE")
    print("=" * 45)

    for filename, content in KNOWLEDGE_BASE.items():

        sentence_count = validate_document(filename, content)

        file_path = KB_DIR / filename

        file_path.write_text(
            content.strip() + "\n",
            encoding="utf-8"
        )

        print(
            f"Created: {filename} | "
            f"Sentences: {sentence_count}"
        )

    print("=" * 45)
    print(f"Total documents created: {len(KNOWLEDGE_BASE)}")
    print("All documents contain 2-5 sentences.")
    print("Knowledge Base created successfully!")


if __name__ == "__main__":
    create_knowledge_base()
