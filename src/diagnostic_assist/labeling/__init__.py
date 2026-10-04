"""Turns closed, unlabelled service cases into a per-family list of root causes.

Stages, in order (pipeline.py wires them together):

    triage -> extract (LLM) -> validate -> decide status -> canonicalise -> quality checks
"""
