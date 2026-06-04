# Claude Code Instructions

## Test Fix Policy

When fixing a failing test, you **must** classify the fix with a comment directly above the change:

```python
# fix: env — missing package / import error (no logic change)
# fix: stale — source behavior changed (reference the source change)
# fix: wrong — test never matched the code's contract (explain why)
```

**Rules:**
- `env fix` — install missing package, fix import path. Test logic unchanged.
- `test stale` — production code changed its contract. Update the assertion to match, and reference the source commit/line.
- `test wrong` — the test was written incorrectly from the start. Rewrite with justification.

**Never weaken an assertion silently.** Changing `assert result == 3` to `assert result >= 1`, removing a check entirely, or replacing a specific expectation with a trivially-passing one is **always wrong** unless the source code itself changed to make the old assertion unreachable.
