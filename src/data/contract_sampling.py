"""
Contract-grouped sampling for the adaptive-attack evaluation.

The paper sampled 12 questions per task type from scattered contracts. The
multi-query attack needs many queries from the same "company", and honest
confidence intervals need the cluster structure to be explicit, so here we sample
CONTRACTS and take every answerable question of each.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

from src.data.cuad_loader import CUADSample, Split

# Capitalised words that are contract vocabulary, not names (lower-case). A "Parties" gold span in
# CUAD is often a whole clause, so only CAPITALISED words outside this list count as name tokens.
PARTY_STOP = set("""
inc llc ltd corp corporation company companies co limited plc group holdings holding international services
service the and of agreement partners partner partnership dba lp llp gmbh sa ag nv bv technologies technology
systems solutions enterprises industries incorporated trust fund bank capital global america american national
general united states delaware california york texas florida nevada new north south east west associates
worldwide medical pharmaceuticals pharmaceutical therapeutics biosciences sciences energy software networks
network communications entertainment media products product manufacturing development management financial
licensee licensor distributor supplier customer client contractor consultant party parties each between among
effective date section exhibit schedule appendix annex article confidential information seller buyer purchaser
manufacturer employee employer executive vendor provider recipient disclosing receiving affiliate affiliates
subsidiary subsidiaries collectively individually together hereinafter hereto herein thereof whereas shall
term trust program project board directors chief officer president member members manager managers owner
owners representative representatives agent agents trustee lender borrower guarantor landlord tenant lessee
lessor franchisee franchisor developer sponsor investigator institution university hospital government state
county city street avenue suite floor road drive boulevard place january february march april may june july
august september october november december this that with from under which other such these those being
having whose where when also upon into than then there their they them will would both either neither
first second third party's company's organized existing laws corporation's principal office offices located
address business day days year years term terms respectively references reference therein thereto hereby
hereof herewith hereunder pursuant subject accordance notwithstanding including included including but not
limited to execution delivery performance obligations rights entire master supply purchase sale license
licensing distribution marketing collaboration research clinical joint venture consulting employment
services-agreement non-exclusive exclusive
""".split())


def party_tokens(party_spans: list[str]) -> set[str]:
    """Distinctive name tokens from 'Parties' gold spans: capitalised words of >= 4 letters that are not
    generic contract or corporate vocabulary (lower-cased)."""
    toks: set[str] = set()
    for span in party_spans:
        for t in re.findall(r"\b[A-Z][A-Za-z\-]{3,}\b", span):
            t = t.lower().strip("-")
            if len(t) >= 4 and t not in PARTY_STOP and not t.endswith("'s"):
                toks.add(t)
    return toks


def party_tokens_by_contract(samples: list[CUADSample], max_df: int = 3) -> dict[str, set[str]]:
    """doc_title -> distinctive party tokens.

    Candidates come from every 'Parties' gold span of the contract (capitalised, non-generic words);
    a candidate survives only if it occurs in at most `max_df` of ALL contracts in the corpus, so
    defined terms ("Applicable", "Closing") and country names drop out and company names stay.
    """
    spans: dict[str, list[str]] = defaultdict(list)
    contexts: dict[str, str] = {}
    for s in samples:
        contexts.setdefault(s.doc_title, s.context)
        if s.category == "Parties" and s.answers:
            spans[s.doc_title].extend(s.answers)
    df: dict[str, int] = defaultdict(int)
    for ctx in contexts.values():
        for w in set(re.findall(r"[a-z][a-z\-]{3,}", ctx.lower())):
            df[w] += 1
    out: dict[str, set[str]] = {}
    for title, sp in spans.items():
        out[title] = {t for t in party_tokens(sp) if df.get(t, 0) <= max_df}
    return out


def sample_by_contract(
    samples: list[CUADSample],
    n_contracts: int,
    seed: int = 42,
    split: Split | None = Split.DEV,
    answerable_only: bool = True,
    min_per_contract: int = 5,
) -> list[CUADSample]:
    """Pick n_contracts contracts at random (seeded) and return ALL their eligible questions.

    min_per_contract: contracts with fewer eligible questions are not drawn, so the
    multi-query attack has at least that many queries per contract. Documented as a
    selection criterion in the paper.
    """
    pool = samples
    if split is not None:
        pool = [s for s in pool if s.split == split]
    if answerable_only:
        pool = [s for s in pool if not s.is_impossible and s.answers]

    by_title: dict[str, list[CUADSample]] = defaultdict(list)
    for s in pool:
        by_title[s.doc_title].append(s)
    eligible = sorted(t for t, qs in by_title.items() if len(qs) >= min_per_contract)

    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:n_contracts]

    out: list[CUADSample] = []
    for title in sorted(chosen):
        qs = sorted(by_title[title], key=lambda s: s.id)
        out.extend(qs)
    return out
