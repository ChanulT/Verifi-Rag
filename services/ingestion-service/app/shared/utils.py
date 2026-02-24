"""
Date Extraction Utility for Ingestion Service.

Extracts document date/year from:
1. Filename patterns
2. Document content (looks for medical-specific prefixes)
3. PDF metadata
"""

import re
import logging
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["DateExtractor", "extract_document_date"]


class DateExtractor:
    """
    Extract date information from documents.

    Priority:
    1. Explicit date in filename
    2. Date patterns in content (first 3000-5000 chars)
    3. Year in filename
    """

    # Full dates patterns (Standard, US, EU, Written)
    DATE_PATTERNS = [
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'mdy'),
        (r'(\d{1,2})-(\d{1,2})-(\d{4})', 'dmy'),

        # NEW Pattern: Handle "20-Feb-2023" or "20 Feb 2023"
        (r'(\d{1,2})[\s\-]+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s\-]+(\d{4})',
         'written_eu'),

        # Existing US Written: "Dec 02, 2023"
        (r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})',
         'written'),
    ]

    YEAR_PATTERNS = [
        r'[_\-\s](\d{4})[_\-\s\.]',
        r'(\d{4})[_\-]',
        r'[_\-](\d{4})',
        r'[^\d](\d{4})[^\d]',
    ]

    # UPDATED: Medical report specific date patterns
    MEDICAL_DATE_PATTERNS = [
        # Standard headers
        r'(?:Date|DATE|Report Date|Collection Date|Test Date|Sample Date)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
        r'(?:Date|DATE|Report Date)[:\s]+(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})',
        r'(?:Collected|Reported|Tested)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',

        # NEW: Specifically for lab report status/lifecycle dates
        r'(?:Collected\s+on|COLLECTED\s+ON)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
        r'(?:Registered\s+on|REGISTERED\s+ON)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
        r'(?:Approved\s+on|APPROVED\s+ON)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
        r'(?:Sample\s+Collected\s+at|Collected\s+at)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})',
    ]

    MONTH_MAP = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
        'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }

    def __init__(self):
        self.current_year = datetime.now().year

    def extract_from_filename(self, filename: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract date or year from filename string."""
        base_name = Path(filename).stem
        for pattern, fmt in self.DATE_PATTERNS:
            match = re.search(pattern, base_name, re.IGNORECASE)
            if match:
                try:
                    date_str = self._parse_date_match(match, fmt)
                    if date_str:
                        year = int(date_str[:4])
                        if 1990 <= year <= self.current_year + 1:
                            return date_str, year
                except Exception:
                    pass

        for pattern in self.YEAR_PATTERNS:
            match = re.search(pattern, base_name)
            if match:
                year = int(match.group(1))
                if 1990 <= year <= self.current_year + 1:
                    return None, year
        return None, None

    def extract_from_content(self, content: str, max_chars: int = 5000) -> Tuple[Optional[str], Optional[int]]:
        """Extract date using medical specific patterns from document text."""
        search_text = content[:max_chars]

        # Try medical patterns first
        for pattern in self.MEDICAL_DATE_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                date_part = match.group(1)
                parsed_date, parsed_year = self._parse_simple_date(date_part)
                if parsed_date:
                    return parsed_date, parsed_year

        # Fallback to general patterns
        for pattern, fmt in self.DATE_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                try:
                    date_str = self._parse_date_match(match, fmt)
                    if date_str:
                        year = int(date_str[:4])
                        if 1990 <= year <= self.current_year + 1:
                            return date_str, year
                except Exception:
                    pass

        # Check for any isolated year
        year_match = re.search(r'\b(20[0-2][0-9])\b', search_text)
        if year_match:
            year = int(year_match.group(1))
            if 1990 <= year <= self.current_year + 1:
                return None, year

        return None, None

    def _parse_date_match(self, match: re.Match, fmt: str) -> Optional[str]:
        groups = match.groups()
        if fmt == '%Y-%m-%d':
            return f"{groups[0]}-{groups[1].zfill(2)}-{groups[2].zfill(2)}"
        elif fmt == 'mdy' or fmt == 'dmy':
            # Basic validation: if first group > 12, assume dmy
            if int(groups[0]) > 12:
                return f"{groups[2]}-{groups[1].zfill(2)}-{groups[0].zfill(2)}"
            return f"{groups[2]}-{groups[0].zfill(2)}-{groups[1].zfill(2)}"
        elif fmt == 'written' or fmt == 'written_eu':
            # Extract year from the last group regardless of order
            m_idx = 0 if fmt == 'written' else 1
            d_idx = 1 if fmt == 'written' else 0
            month = self.MONTH_MAP.get(groups[m_idx].lower()[:3], 1)
            return f"{groups[2]}-{str(month).zfill(2)}-{groups[d_idx].zfill(2)}"
        return None

    def _parse_simple_date(self, date_str: str) -> Tuple[Optional[str], Optional[int]]:
        formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d']
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if 1990 <= dt.year <= self.current_year + 1:
                    return dt.strftime('%Y-%m-%d'), dt.year
            except ValueError:
                continue
        return None, None

    def extract(self, filename: str, content: Optional[str] = None) -> Tuple[Optional[str], Optional[int]]:
        """Orchestrates date extraction based on priority."""
        date_str, year = self.extract_from_filename(filename)
        if date_str:
            return date_str, year
        if content:
            c_date, c_year = self.extract_from_content(content)
            if c_date:
                return c_date, c_year
            if not year and c_year:
                return None, c_year
        return date_str, year


_extractor = DateExtractor()


def extract_document_date(filename: str, content: Optional[str] = None) -> Tuple[Optional[str], Optional[int]]:
    """Entry point for document date extraction."""
    return _extractor.extract(filename, content)