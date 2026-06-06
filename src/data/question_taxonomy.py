"""
Maps CUAD clause categories to 4 task types corresponding to the pipeline's
expected performance envelope (documented in research/06_proposed_pipeline.md).

framework_reasoning:  Pipeline expected to perform well — asks about reasoning structures
compliance_regulatory: Pipeline expected to perform well — asks what rules/clauses apply
procedural:           Pipeline expected to perform well — asks about process steps
entity_specific:      Pipeline expected to degrade — requires specific entity/value extraction
"""

CATEGORY_TO_TASK_TYPE: dict[str, str] = {
    # framework_reasoning — structural / relational reasoning
    "Non-Compete": "framework_reasoning",
    "Competitive Restriction Exception": "framework_reasoning",
    "Anti-Assignment": "framework_reasoning",
    "Change Of Control": "framework_reasoning",
    "Covenant Not To Sue": "framework_reasoning",
    "Non-Transferable License": "framework_reasoning",
    "Non-Disparagement": "framework_reasoning",
    "No-Solicit Of Customers": "framework_reasoning",
    "No-Solicit Of Employees": "framework_reasoning",
    "Most Favored Nation": "framework_reasoning",
    "Rofr/Rofo/Rofn": "framework_reasoning",
    "Exclusivity": "framework_reasoning",

    # compliance_regulatory — what rules, caps, or obligations apply
    "Governing Law": "compliance_regulatory",
    "Audit Rights": "compliance_regulatory",
    "Cap On Liability": "compliance_regulatory",
    "Uncapped Liability": "compliance_regulatory",
    "Insurance": "compliance_regulatory",
    "Liquidated Damages": "compliance_regulatory",
    "Third Party Beneficiary": "compliance_regulatory",
    "Revenue/Profit Sharing": "compliance_regulatory",
    "Price Restrictions": "compliance_regulatory",
    "Ip Ownership Assignment": "compliance_regulatory",
    "Joint Ip Ownership": "compliance_regulatory",
    "Source Code Escrow": "compliance_regulatory",

    # procedural — process, steps, timelines
    "Notice Period To Terminate Renewal": "procedural",
    "Post-Termination Services": "procedural",
    "Renewal Term": "procedural",
    "Termination For Convenience": "procedural",
    "Affiliate License-Licensor": "procedural",
    "Affiliate License-Licensee": "procedural",

    # entity_specific — requires extracting specific values (degradation expected)
    "Parties": "entity_specific",
    "Document Name": "entity_specific",
    "Agreement Date": "entity_specific",
    "Effective Date": "entity_specific",
    "Expiration Date": "entity_specific",
    "Warranty Duration": "entity_specific",
    "Minimum Commitment": "entity_specific",
    "Volume Restriction": "entity_specific",
    "License Grant": "entity_specific",
    "Unlimited/All-You-Can-Eat-License": "entity_specific",
    "Irrevocable Or Perpetual License": "entity_specific",
}

TASK_TYPES = ["framework_reasoning", "compliance_regulatory", "procedural", "entity_specific"]


def get_task_type(category: str) -> str:
    return CATEGORY_TO_TASK_TYPE.get(category, "entity_specific")
