# sample-size-mcp

How many test cases do you need to trust that model A is really better than model B?

Usually more than you'd think: telling 82% from 78% takes about **1,500** cases if the models' errors are unrelated. If both models run on the same test set, and they usually make the same mistakes, the number can drop to about **500**. This tool does the math. You can use it as a CLI or as an MCP server, so an agent can check an eval before trusting it.

## CLI

```sh
uv run evalsize n 0.82 0.78                     # cases needed (same test set for both models)
uv run evalsize n 0.82 0.78 --discordance 0.10  # ...when the models disagree on 10% of cases
uv run evalsize n 0.82 0.78 --unpaired          # each model gets its own sample
uv run evalsize gap 500 0.78                    # smallest win 500 cases can detect
uv run evalsize compare 500 --a-only 41 --b-only 22 --both 360   # is this result real?
```

```
$ uv run evalsize n 0.82 0.78 --discordance 0.10
489 test cases (shared by both models)
  to detect 82.0% vs 78.0% (gap 4.0%) at alpha=0.05, power=80%
  discordance 10.0% (given); ~49 cases will carry the signal

$ uv run evalsize compare 500 --a-only 41 --b-only 22 --both 360
A 80.2% vs B 76.4%: gap +3.8%, 95% CI [+0.7%, +6.9%], p=0.0226 -> SIGNIFICANT
```

All commands accept `--alpha` (false-positive rate, default 0.05) and `--power` (the chance of detecting a real gap, default 0.8).

## MCP server

```json
{
  "mcpServers": {
    "eval-sample-size": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/fayehall-ai/sample-size-mcp", "sample-size-mcp"]
    }
  }
}
```

Or with Claude Code: `claude mcp add eval-sample-size -- uvx --from git+https://github.com/fayehall-ai/sample-size-mcp sample-size-mcp`

Tools: `eval_sample_size`, `eval_min_detectable_gap`, `eval_compare_paired`, `eval_compare_unpaired`.

## The statistics

**Paired** (the default) means both models are scored on the same cases. Cases where both models are right, or both wrong, tell you nothing about which model is better. Only the cases where they disagree count, so the required sample size depends on **discordance**: the fraction of cases where exactly one model is right. This tool uses McNemar's test, with sample size from Connor (1987):

    n = (z_{α/2}·√ψ + z_β·√(ψ − δ²))² / δ²       ψ = discordance, δ = accuracy gap

If you don't pass a discordance, the tool assumes the models' errors are independent, ψ = a(1−b) + b(1−a). That's the worst realistic case, and it needs about as many cases as the unpaired design. Real models usually fail on the same hard cases, so the true discordance is lower and you need fewer cases. **Run a small pilot (~100 cases), measure how often the models disagree, and pass that number.** This is usually the biggest lever you have.

**Unpaired** means each model is scored on its own random sample. This uses the two-proportion z-test (Fleiss formula, no continuity correction), and *n* is the number of cases per model.

**Checking results:** `compare` / `eval_compare_paired` uses the exact McNemar (binomial) test, with a Wald confidence interval on the gap.

Tests include a Monte Carlo check that the recommended *n* actually gives about 80% power. Run them with `uv run pytest`.

### Caveats
- Every test case counts as independent. If your cases come in clusters (several prompts per document, or several samples per prompt), you need more cases than this says.
- Each test is a single comparison. If you compare many models or checkpoints, divide `alpha` by the number of comparisons (Bonferroni correction).
- For accuracies close to 0 or 1, or very small samples, the normal approximation becomes rough.
