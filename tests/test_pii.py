from __future__ import annotations

from ftpipe.data.pii import analyze, anonymize, luhn_valid, residual_entities, summarize


def test_recognizers_and_luhn():
    text = (
        "contact jane.doe@example.test or 555-010-0107; card 4111 1111 1111 1111; "
        "ssn 123-45-6789; ip 192.0.2.44; Ms. Rivera reported it."
    )
    spans = analyze(text)
    assert {span.entity for span in spans} >= {"EMAIL", "PHONE", "CREDIT_CARD", "SSN", "IPV4", "PERSON"}
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not analyze("card 4111 1111 1111 1112")


def test_anonymization_is_complete_and_idempotent():
    text = "email a@example.test phone 555-010-0107 card 4111 1111 1111 1111 ip 198.51.100.9"
    redacted, spans = anonymize(text)
    assert "a@example.test" not in redacted
    assert "555-010-0107" not in redacted
    assert "4111 1111 1111 1111" not in redacted
    assert "198.51.100.9" not in redacted
    assert residual_entities(redacted) == []
    assert summarize(spans)["EMAIL"] == 1
    assert anonymize(redacted)[0] == redacted
