"""Presidio-style PII recognizers and anonymization."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Iterable

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
PERSON_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]{2,}\b")
PERSON_NAMES = {
    "rivera", "chen", "patel", "silva", "garcia", "morgan", "taylor", "jordan",
    "aisha", "marco", "jane", "li", "wang", "khan", "nguyen", "okafor",
}
NAME_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}\s+){1,2}(?:[A-Z][a-z]{2,})\b")

REPLACEMENTS = {
    "EMAIL": "<EMAIL>",
    "PHONE": "<PHONE>",
    "CREDIT_CARD": "<CREDIT_CARD>",
    "SSN": "<SSN>",
    "IPV4": "<IPV4>",
    "PERSON": "<PERSON>",
}
PRIORITY = {"SSN": 0, "CREDIT_CARD": 1, "EMAIL": 2, "IPV4": 3, "PHONE": 4, "PERSON": 5}


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    entity: str
    score: float = 1.0


def luhn_valid(digits: str) -> bool:
    values = [int(ch) for ch in digits if ch.isdigit()]
    if len(values) < 13:
        return False
    checksum = 0
    parity = len(values) % 2
    for index, value in enumerate(values):
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _valid_ipv4(value: str) -> bool:
    try:
        return ipaddress.IPv4Address(value) is not None
    except ValueError:
        return False


def _candidates(text: str, entities: set[str]) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    if "EMAIL" in entities:
        spans.extend(EntitySpan(m.start(), m.end(), "EMAIL") for m in EMAIL_RE.finditer(text))
    if "CREDIT_CARD" in entities:
        for match in CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if luhn_valid(digits):
                spans.append(EntitySpan(match.start(), match.end(), "CREDIT_CARD"))
    if "SSN" in entities:
        spans.extend(EntitySpan(m.start(), m.end(), "SSN") for m in SSN_RE.finditer(text))
    if "IPV4" in entities:
        for match in IPV4_RE.finditer(text):
            if _valid_ipv4(match.group(0)):
                spans.append(EntitySpan(match.start(), match.end(), "IPV4"))
    if "PHONE" in entities:
        spans.extend(EntitySpan(m.start(), m.end(), "PHONE") for m in PHONE_RE.finditer(text))
    if "PERSON" in entities:
        spans.extend(EntitySpan(m.start(), m.end(), "PERSON") for m in PERSON_RE.finditer(text))
        for match in NAME_RE.finditer(text):
            if any(part.lower() in PERSON_NAMES for part in match.group(0).split()):
                spans.append(EntitySpan(match.start(), match.end(), "PERSON"))
    return spans


def analyze(text: str, entities: Iterable[str] | None = None) -> list[EntitySpan]:
    requested = set(entities or REPLACEMENTS)
    candidates = _candidates(text, requested)
    candidates.sort(key=lambda span: (PRIORITY.get(span.entity, 99), span.start, -(span.end - span.start)))
    chosen: list[EntitySpan] = []
    for span in candidates:
        if any(span.start < other.end and other.start < span.end for other in chosen):
            continue
        chosen.append(span)
    return sorted(chosen, key=lambda span: span.start)


def anonymize(text: str, entities: Iterable[str] | None = None) -> tuple[str, list[EntitySpan]]:
    spans = analyze(text, entities)
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(text[cursor : span.start])
        parts.append(REPLACEMENTS[span.entity])
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts), spans


def residual_entities(text: str, entities: Iterable[str] | None = None) -> list[EntitySpan]:
    return analyze(text, entities)


def summarize(spans: list[EntitySpan]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.entity] = counts.get(span.entity, 0) + 1
    return dict(sorted(counts.items()))
