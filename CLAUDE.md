<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **macro-main** (8910 symbols, 13589 relationships, 139 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/macro-main/context` | Codebase overview, check index freshness |
| `gitnexus://repo/macro-main/clusters` | All functional areas |
| `gitnexus://repo/macro-main/processes` | All execution flows |
| `gitnexus://repo/macro-main/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |
| Work in the Func area (306 symbols) | `.claude/skills/generated/func/SKILL.md` |
| Work in the Readers area (55 symbols) | `.claude/skills/generated/readers/SKILL.md` |
| Work in the Spoof_data area (46 symbols) | `.claude/skills/generated/spoof-data/SKILL.md` |
| Work in the Synthetic_banks area (41 symbols) | `.claude/skills/generated/synthetic-banks/SKILL.md` |
| Work in the Economic_data area (40 symbols) | `.claude/skills/generated/economic-data/SKILL.md` |
| Work in the Rest_of_the_world area (35 symbols) | `.claude/skills/generated/rest-of-the-world/SKILL.md` |
| Work in the Unit area (35 symbols) | `.claude/skills/generated/unit/SKILL.md` |
| Work in the Synthetic_population area (31 symbols) | `.claude/skills/generated/synthetic-population/SKILL.md` |
| Work in the Macromodel area (27 symbols) | `.claude/skills/generated/macromodel/SKILL.md` |
| Work in the Synthetic_firms area (26 symbols) | `.claude/skills/generated/synthetic-firms/SKILL.md` |
| Work in the Util area (25 symbols) | `.claude/skills/generated/util/SKILL.md` |
| Work in the Synthetic_matching area (16 symbols) | `.claude/skills/generated/synthetic-matching/SKILL.md` |
| Work in the Forecaster area (15 symbols) | `.claude/skills/generated/forecaster/SKILL.md` |
| Work in the Population_data area (14 symbols) | `.claude/skills/generated/population-data/SKILL.md` |
| Work in the Io_tables area (14 symbols) | `.claude/skills/generated/io-tables/SKILL.md` |
| Work in the Credit_market area (14 symbols) | `.claude/skills/generated/credit-market/SKILL.md` |
| Work in the Synthetic_credit_market area (12 symbols) | `.claude/skills/generated/synthetic-credit-market/SKILL.md` |
| Work in the Test_goods_market area (11 symbols) | `.claude/skills/generated/test-goods-market/SKILL.md` |
| Work in the Agent area (11 symbols) | `.claude/skills/generated/agent/SKILL.md` |
| Work in the Synthetic_central_government area (11 symbols) | `.claude/skills/generated/synthetic-central-government/SKILL.md` |

<!-- gitnexus:end -->
