import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from p1_report import kickoff_row  # noqa: E402


def test_kickoff_row_reports_a_missing_kickoff():
    # The plan: "Kickoff bulunamazsa ... rapor bunu yazar" (review finding 7).
    row = kickoff_row(2, None, 113742)
    assert "bulunamadı" in row and "113742" in row


def test_kickoff_row_signed_difference():
    assert kickoff_row(2, 113743, 113742) == "| 2. yarı kickoff | 113743 | 113742 | +1 kare |"
