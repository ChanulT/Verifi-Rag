#!/usr/bin/env python3
"""
Verifi-RAG Truth Script - Automated Evaluation

This script runs automated questions against ingested medical documents
and evaluates if the answers match expected ground truth values.

Requirements:
- Ingestion service running at http://localhost:8002
- Chat service running at http://localhost:8000
- At least one medical PDF ingested

Usage:
    # With default test data
    python truth_script.py

    # With custom ground truth file
    python truth_script.py --ground-truth my_ground_truth.json

    # With specific PDF to ingest first
    python truth_script.py --pdf blood_test_2023.pdf --ground-truth ground_truth.json
"""

import asyncio
import json
import re
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

import httpx

# =============================================================================
# Configuration
# =============================================================================

INGESTION_SERVICE_URL = "http://localhost:8002"
CHAT_SERVICE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120.0  # seconds
POLLING_INTERVAL = 2.0  # seconds


class EvalResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    NO_ANSWER = "NO_ANSWER"


@dataclass
class TestCase:
    """A single test case with question and expected answer."""
    id: str
    question: str
    expected_answer: str
    expected_values: List[str] = field(default_factory=list)  # Specific values to check
    expected_source: Optional[str] = None  # Expected source filename
    allow_partial: bool = True  # Pass if at least some expected values found


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    question: str
    expected: str
    actual_answer: str
    citations: List[Dict]
    result: EvalResult
    score: float  # 0.0 to 1.0
    details: str
    response_time_ms: float
    confidence: float


@dataclass
class EvaluationReport:
    """Overall evaluation report."""
    total_tests: int
    passed: int
    failed: int
    partial: int
    errors: int
    overall_score: float
    avg_response_time_ms: float
    avg_confidence: float
    results: List[TestResult]


# =============================================================================
# Default Test Cases (for a typical blood test report)
# =============================================================================

DEFAULT_TEST_CASES = [
    TestCase(
        id="TC001",
        question="What is the hemoglobin level in the blood test report?",
        expected_answer="The hemoglobin level should be a numeric value with units (g/dL)",
        expected_values=["hemoglobin", "g/dL", "Hgb", "HGB"],
        allow_partial=True,
    ),
    TestCase(
        id="TC002",
        question="What are the cholesterol levels mentioned in the report?",
        expected_answer="Should mention total cholesterol, LDL, HDL, or triglycerides",
        expected_values=["cholesterol", "LDL", "HDL", "triglycerides", "mg/dL"],
        allow_partial=True,
    ),
    TestCase(
        id="TC003",
        question="What is the fasting blood glucose level?",
        expected_answer="Blood glucose value in mg/dL",
        expected_values=["glucose", "blood sugar", "mg/dL", "fasting"],
        allow_partial=True,
    ),
    TestCase(
        id="TC004",
        question="What medications is the patient currently taking?",
        expected_answer="Should say 'I don't know' or similar if not in the document",
        expected_values=["don't know", "not found", "no information", "not mentioned", "cannot find"],
        allow_partial=True,
    ),
    TestCase(
        id="TC005",
        question="What is the patient's blood type?",
        expected_answer="Blood type (A, B, AB, O with Rh factor) or 'not found'",
        expected_values=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "don't know", "not found", "not mentioned"],
        allow_partial=True,
    ),
]


# =============================================================================
# Service Clients
# =============================================================================

class IngestionClient:
    """Client for the ingestion service."""

    def __init__(self, base_url: str = INGESTION_SERVICE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def health_check(self) -> bool:
        """Check if ingestion service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def ingest_pdf(self, file_path: Path) -> str:
        """
        Ingest a PDF file and return job_id.
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            response = await self.client.post(
                f"{self.base_url}/ingest",
                files=files,
            )
            response.raise_for_status()
            return response.json()["job_id"]

    async def wait_for_job(self, job_id: str, max_wait: float = 300.0) -> Dict:
        """
        Poll job status until completion.
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            response = await self.client.get(f"{self.base_url}/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()

            status = job.get("status", "unknown")
            progress = job.get("progress", 0)

            print(f"  Job {job_id[:8]}... status={status}, progress={progress}%")

            if status == "completed":
                return job
            elif status == "failed":
                raise Exception(f"Job failed: {job.get('error_message', 'Unknown error')}")

            await asyncio.sleep(POLLING_INTERVAL)

        raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")

    async def close(self):
        await self.client.aclose()


class ChatClient:
    """Client for the chat service."""

    def __init__(self, base_url: str = CHAT_SERVICE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def health_check(self) -> bool:
        """Check if chat service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def ask_question(
            self,
            question: str,
            session_id: Optional[str] = None,
            document_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Send a question to the chat service.
        """
        payload = {
            "message": question,
        }

        if session_id:
            payload["session_id"] = session_id

        if document_ids:
            payload["document_ids"] = document_ids

        start_time = time.time()

        response = await self.client.post(
            f"{self.base_url}/chat",
            json=payload,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if response.status_code != 200:
            return {
                "answer": f"Error: {response.status_code}",
                "citations": [],
                "status": "error",
                "confidence": 0.0,
                "response_time_ms": elapsed_ms,
            }

        result = response.json()
        result["response_time_ms"] = elapsed_ms
        return result

    async def close(self):
        await self.client.aclose()


# =============================================================================
# Evaluation Logic
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation except for common units
    text = re.sub(r'[^\w\s/\-\+\.]', '', text)
    return text.strip()


def check_expected_values(answer: str, expected_values: List[str]) -> tuple[bool, List[str], List[str]]:
    """
    Check if expected values are present in the answer.

    Returns:
        (all_found, found_values, missing_values)
    """
    normalized_answer = normalize_text(answer)

    found = []
    missing = []

    for value in expected_values:
        normalized_value = normalize_text(value)
        if normalized_value in normalized_answer:
            found.append(value)
        else:
            missing.append(value)

    all_found = len(missing) == 0
    return all_found, found, missing


def check_citations_present(response: Dict) -> bool:
    """Check if the response includes citations."""
    citations = response.get("citations", [])
    return len(citations) > 0


def check_grounded_response(response: Dict) -> bool:
    """Check if the response is grounded (not a hallucination)."""
    status = response.get("status", "")
    confidence = response.get("confidence", 0.0)

    # A grounded response either:
    # 1. Has success status with citations
    # 2. Has "no_relevant_context" status (correctly says "I don't know")

    if status == "success" and confidence > 0.5:
        return True
    if status == "no_relevant_context":
        return True  # Correctly acknowledged lack of information

    return False


def evaluate_test_case(test_case: TestCase, response: Dict) -> TestResult:
    """
    Evaluate a single test case against the response.
    """
    answer = response.get("answer", "")
    citations = response.get("citations", [])
    confidence = response.get("confidence", 0.0)
    status = response.get("status", "unknown")
    response_time_ms = response.get("response_time_ms", 0.0)

    # Check for error
    if status == "error":
        return TestResult(
            test_id=test_case.id,
            question=test_case.question,
            expected=test_case.expected_answer,
            actual_answer=answer,
            citations=citations,
            result=EvalResult.ERROR,
            score=0.0,
            details=f"Service returned error: {answer}",
            response_time_ms=response_time_ms,
            confidence=confidence,
        )

    # Check expected values
    all_found, found_values, missing_values = check_expected_values(
        answer,
        test_case.expected_values
    )

    # Calculate score
    if len(test_case.expected_values) > 0:
        score = len(found_values) / len(test_case.expected_values)
    else:
        score = 1.0 if answer else 0.0

    # Determine result
    if all_found:
        result = EvalResult.PASS
        details = f"All expected values found: {found_values}"
    elif found_values and test_case.allow_partial:
        result = EvalResult.PARTIAL
        details = f"Found: {found_values}. Missing: {missing_values}"
    elif found_values:
        result = EvalResult.FAIL
        details = f"Only found: {found_values}. Missing: {missing_values}"
    else:
        result = EvalResult.FAIL
        details = f"None of expected values found. Expected any of: {test_case.expected_values}"

    # Check for proper "I don't know" responses
    i_dont_know_keywords = ["don't know", "not found", "no information", "cannot find", "not mentioned"]
    answer_says_dont_know = any(kw in normalize_text(answer) for kw in i_dont_know_keywords)
    test_expects_dont_know = any(kw in test_case.expected_values for kw in i_dont_know_keywords)

    if test_expects_dont_know and answer_says_dont_know:
        result = EvalResult.PASS
        score = 1.0
        details = "Correctly responded with 'I don't know' (no hallucination)"

    # Bonus: Check if citations are present when answer is given
    if result == EvalResult.PASS and not answer_says_dont_know:
        if not citations:
            details += " [Warning: No citations provided]"
            score *= 0.9  # Small penalty for missing citations

    return TestResult(
        test_id=test_case.id,
        question=test_case.question,
        expected=test_case.expected_answer,
        actual_answer=answer[:500] + "..." if len(answer) > 500 else answer,
        citations=citations,
        result=result,
        score=score,
        details=details,
        response_time_ms=response_time_ms,
        confidence=confidence,
    )


async def run_evaluation(
        test_cases: List[TestCase],
        chat_client: ChatClient,
        session_id: Optional[str] = None,
) -> EvaluationReport:
    """
    Run all test cases and generate evaluation report.
    """
    results = []

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case.id}: {test_case.question[:60]}...")

        try:
            response = await chat_client.ask_question(
                question=test_case.question,
                session_id=session_id,
            )

            result = evaluate_test_case(test_case, response)
            results.append(result)

            # Print immediate result
            emoji = {
                EvalResult.PASS: "✅",
                EvalResult.PARTIAL: "🟡",
                EvalResult.FAIL: "❌",
                EvalResult.ERROR: "💥",
                EvalResult.NO_ANSWER: "⚠️",
            }[result.result]

            print(f"  Result: {emoji} {result.result.value}")
            print(f"  Score: {result.score:.2f}")
            print(f"  Time: {result.response_time_ms:.0f}ms")
            print(f"  Details: {result.details[:80]}...")

        except Exception as e:
            print(f"  Error: {e}")
            results.append(TestResult(
                test_id=test_case.id,
                question=test_case.question,
                expected=test_case.expected_answer,
                actual_answer=f"Exception: {str(e)}",
                citations=[],
                result=EvalResult.ERROR,
                score=0.0,
                details=str(e),
                response_time_ms=0.0,
                confidence=0.0,
            ))

    # Calculate summary statistics
    passed = sum(1 for r in results if r.result == EvalResult.PASS)
    failed = sum(1 for r in results if r.result == EvalResult.FAIL)
    partial = sum(1 for r in results if r.result == EvalResult.PARTIAL)
    errors = sum(1 for r in results if r.result == EvalResult.ERROR)

    overall_score = sum(r.score for r in results) / len(results) if results else 0.0
    avg_response_time = sum(r.response_time_ms for r in results) / len(results) if results else 0.0
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0

    return EvaluationReport(
        total_tests=len(test_cases),
        passed=passed,
        failed=failed,
        partial=partial,
        errors=errors,
        overall_score=overall_score,
        avg_response_time_ms=avg_response_time,
        avg_confidence=avg_confidence,
        results=results,
    )


def print_report(report: EvaluationReport):
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         SUMMARY                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total Tests:        {report.total_tests:>5}                         ║
║  Passed:             {report.passed:>5} ✅                           ║
║  Partial:            {report.partial:>5} 🟡                           ║
║  Failed:             {report.failed:>5} ❌                           ║
║  Errors:             {report.errors:>5} 💥                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Overall Score:      {report.overall_score:>5.1%}                    ║
║  Avg Response Time:  {report.avg_response_time_ms:>5.0f}ms           ║
║  Avg Confidence:     {report.avg_confidence:>5.1%}                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    print("\nDETAILED RESULTS:")
    print("-" * 70)

    for result in report.results:
        emoji = {
            EvalResult.PASS: "✅",
            EvalResult.PARTIAL: "🟡",
            EvalResult.FAIL: "❌",
            EvalResult.ERROR: "💥",
            EvalResult.NO_ANSWER: "⚠️",
        }[result.result]

        print(f"\n{emoji} [{result.test_id}] {result.result.value} (Score: {result.score:.2f})")
        print(f"   Question: {result.question}")
        print(f"   Expected: {result.expected[:100]}...")
        print(f"   Actual:   {result.actual_answer[:100]}...")
        print(f"   Details:  {result.details}")
        if result.citations:
            print(f"   Citations: {len(result.citations)} sources cited")
            for c in result.citations[:3]:
                print(f"     - [{c.get('number', '?')}] {c.get('source', 'unknown')}, Page {c.get('page', '?')}")

    print("\n" + "=" * 70)

    # Final verdict
    if report.overall_score >= 0.8:
        print("🎉 EVALUATION: PASSED (Score >= 80%)")
    elif report.overall_score >= 0.6:
        print("🟡 EVALUATION: PARTIAL PASS (Score >= 60%)")
    else:
        print("❌ EVALUATION: FAILED (Score < 60%)")

    print("=" * 70)


def save_report(report: EvaluationReport, output_path: Path):
    """Save evaluation report to JSON file."""
    report_dict = {
        "summary": {
            "total_tests": report.total_tests,
            "passed": report.passed,
            "failed": report.failed,
            "partial": report.partial,
            "errors": report.errors,
            "overall_score": report.overall_score,
            "avg_response_time_ms": report.avg_response_time_ms,
            "avg_confidence": report.avg_confidence,
        },
        "results": [
            {
                "test_id": r.test_id,
                "question": r.question,
                "expected": r.expected,
                "actual_answer": r.actual_answer,
                "citations": r.citations,
                "result": r.result.value,
                "score": r.score,
                "details": r.details,
                "response_time_ms": r.response_time_ms,
                "confidence": r.confidence,
            }
            for r in report.results
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nReport saved to: {output_path}")


def load_test_cases(file_path: Path) -> List[TestCase]:
    """Load test cases from JSON file."""
    with open(file_path) as f:
        data = json.load(f)

    test_cases = []
    for item in data["test_cases"]:
        test_cases.append(TestCase(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_values=item.get("expected_values", []),
            expected_source=item.get("expected_source"),
            allow_partial=item.get("allow_partial", True),
        ))

    return test_cases


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Verifi-RAG Truth Script - Automated Evaluation"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to PDF file to ingest before testing",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Path to ground truth JSON file with test cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_report.json"),
        help="Path to save evaluation report (default: evaluation_report.json)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion step (assume documents already ingested)",
    )

    args = parser.parse_args()

    # Initialize clients
    ingestion_client = IngestionClient()
    chat_client = ChatClient()

    try:
        # Health checks
        print("Checking service health...")

        if not await ingestion_client.health_check():
            print(f"❌ Ingestion service not available at {INGESTION_SERVICE_URL}")
            print("   Please start the service: uvicorn app.main:app --port 8002")
            sys.exit(1)
        print(f"✅ Ingestion service: {INGESTION_SERVICE_URL}")

        if not await chat_client.health_check():
            print(f"❌ Chat service not available at {CHAT_SERVICE_URL}")
            print("   Please start the service: uvicorn app.main:app --port 8000")
            sys.exit(1)
        print(f"✅ Chat service: {CHAT_SERVICE_URL}")

        # Ingest PDF if provided
        if args.pdf and not args.skip_ingest:
            if not args.pdf.exists():
                print(f"❌ PDF file not found: {args.pdf}")
                sys.exit(1)

            print(f"\nIngesting PDF: {args.pdf}")
            job_id = await ingestion_client.ingest_pdf(args.pdf)
            print(f"  Job ID: {job_id}")

            job = await ingestion_client.wait_for_job(job_id)
            print(f"  ✅ Ingestion complete: {job.get('chunks_created', 0)} chunks created")

        # Load test cases
        if args.ground_truth:
            if not args.ground_truth.exists():
                print(f"❌ Ground truth file not found: {args.ground_truth}")
                sys.exit(1)
            test_cases = load_test_cases(args.ground_truth)
            print(f"\nLoaded {len(test_cases)} test cases from {args.ground_truth}")
        else:
            test_cases = DEFAULT_TEST_CASES
            print(f"\nUsing {len(test_cases)} default test cases")

        # Run evaluation
        report = await run_evaluation(test_cases, chat_client)

        # Print and save report
        print_report(report)
        save_report(report, args.output)

        # Return exit code based on result
        if report.overall_score >= 0.6:
            sys.exit(0)
        else:
            sys.exit(1)

    finally:
        await ingestion_client.close()
        await chat_client.close()


if __name__ == "__main__":
    asyncio.run(main())