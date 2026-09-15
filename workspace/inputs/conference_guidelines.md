# Conference Guidelines: PETS / PoPETs

## Venue

**Proceedings on Privacy Enhancing Technologies (PoPETs)**
Published by De Gruyter Open Access.

PoPETs is the journal-style proceedings of the Privacy Enhancing Technologies Symposium (PETS). All accepted papers are published open access.

## Formatting and Length

- **Page limit:** 24 pages excluding references and appendices.
- **Template:** LaTeX using `popets-paper.cls` (based on the standard `article` class). The class file and sample document are available from the PoPETs submission site.
- Do NOT create a template.tex file -- the template will be downloaded separately.

## Review Process

- **Double-blind review.** Author identities must not be revealed in the submission. Remove author names, affiliations, acknowledgements, and any self-identifying references from the submitted version.
- **Rolling submission** with 4 deadlines per year (typically February, May, August, November). Papers rejected at one deadline may be revised and resubmitted to a later deadline.
- Each submission receives at least 3 reviews. Decisions are: Accept, Major Revision (one round of revision permitted), Reject.

## Required Sections

A PoPETs submission should include the following sections:

1. **Abstract** -- concise summary of the problem, approach, and key findings.
2. **Introduction** -- motivation, problem statement, contributions, and paper organization.
3. **Related Work** -- thorough coverage of prior work with clear differentiation.
4. **Methodology** -- detailed description of the proposed system/technique, including threat model.
5. **Evaluation** -- empirical or formal analysis with appropriate metrics and baselines.
6. **Discussion** -- interpretation of results, limitations, and implications.
7. **Conclusion** -- summary and future work.

## Reviewer Expectations

PoPETs reviewers particularly expect:

- **Clear threat model.** Define the adversary's capabilities, goals, and knowledge. State what is and is not protected.
- **Formal or empirical privacy analysis.** Either formal proofs of privacy guarantees or rigorous empirical evaluation of privacy properties (or both).
- **Honest limitations.** Explicitly state what the system does not protect against. Reviewers value intellectual honesty over overclaiming.
- **Reproducibility.** Artifact availability is strongly encouraged. Submissions that include code, data, and instructions for reproduction are viewed favorably.
- **Comparison with baselines.** Evaluate against relevant prior work, not just the proposed system in isolation.

## Artifact Policy

- Artifact availability is strongly encouraged but not strictly required for acceptance.
- Authors are encouraged to submit artifacts (code, datasets, experimental scripts) alongside the paper.
- Accepted papers with available artifacts receive an artifact badge.

## Additional Notes

- PoPETs uses single-column format.
- Figures, tables, and algorithms count toward the page limit; references and appendices do not.
- Supplementary material may be included as appendices but reviewers are not required to read it.
- Ethical considerations: if the work involves human subjects data, attacks on real systems, or potentially harmful techniques, address ethical implications explicitly.
