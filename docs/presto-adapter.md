# BarunAction-35M PRESTO English adapter

This adapter converts the English portion of PRESTO into contextual Action IR v1 SFT
manifests for BarunAction-35M. It is a compile-only dataset transformation: it never executes a
PRESTO action or touches a real contact, list, message, payment, ride, or device.

## Frozen source and firewall

The source definition is pinned to commit
[`fa47167477453afebe698a287409514df5a7dadf`](https://github.com/google-research-datasets/presto/tree/fa47167477453afebe698a287409514df5a7dadf)
of the official archived PRESTO repository and the `presto_v1.zip` URL recorded in that
[revision's README](https://raw.githubusercontent.com/google-research-datasets/presto/fa47167477453afebe698a287409514df5a7dadf/README.md).
The research paper describes the contextual semantic-parsing task and official split policy
([Goel et al., 2023](https://arxiv.org/abs/2303.08954)). PRESTO is CC BY 4.0.

Frozen provenance:

- Archive: 415,990,813 bytes; SHA-256
  `1fc671692cceb31fbda17e351e47f2cc52ee8779042f92dc26674cc0cca2167f`.
- Train member: 276,259 rows; SHA-256
  `b93ccf00ef8a7a67e0f1380da39e273791c26c87855babade5c045bd10c0ad65`.
- Development member: 82,547 rows; SHA-256
  `dcb8beadcf82802b0aa06d3b9b4ce6b25518b15816b7eb6071116e3bb7ccbd02`.
- Official test member: 194,118 rows; SHA-256
  `9549050809fdd91fd117f42d62bdaa71db4770f4ddb5ecb3f394f97539bf21ef`.
- Redundant combined member: 552,924 rows; SHA-256
  `d0c03d0d8cef8b80edf489db0ac97c6b01c8edf99105a970854cc3ce4c2967f8`.

The official test hash and newline count were obtained as source-level byte evidence only. Test
bytes were not decoded and test JSON, prompts, targets, labels, contexts, and partitions were not
inspected. During normal adapter execution, neither `presto_test.jsonl`, the redundant combined
member, nor any `test_partitions/` member is opened. The audit records runtime open counts and byte
counts for every frozen main member and fails if any sensitive member was opened or read. The
whole-archive SHA-256 and central-directory size/CRC metadata are sufficient to verify their pinned
identity without opening their payloads.

## English coverage and mapping

The adapter preserves the official train/development assignment and selects locale `en-US`:

- Train: 47,806 selected; 228,453 non-English rows excluded.
- Development: 14,288 selected; 68,259 non-English rows excluded.
- Semantic-parse exclusions: zero. All 62,094 selected targets parse under the frozen recursive
  grammar.

The native grammar contains ordered slots whose values are quoted text, one of the bare symbols
`InferFromContext`, `AUDIO`, or `VISUAL`, or a nested semantic node. Action IR represents slots as
an ordered list of `{name, value}` objects. This preserves repeated slot names, which a normal JSON
object would lose. The native target string is not copied into the manifest; only its SHA-256,
root intent, and derived Action IR are retained.

Policy mapping is explicit and frozen:

- `Other` and explicit `Cancel` become `ABSTAIN`.
- State-changing, externally visible, or transactional roots become `CONFIRM`.
- Read-only lookup/open roots become `CALL`.

This is a BarunAction safety transformation, not the native PRESTO output contract. Exact match on
the derived Action IR must not be reported as the paper's native semantic-parse string exact-match
metric. No reference timestamp or timezone is supplied by PRESTO, so relative temporal values stay
as annotated strings or `InferFromContext`; the adapter does not invent a resolved time.

The revision-pinned README documents seeded notes with `name` and `text`, but the pinned train/dev
rows consistently use `name` and `content`. The adapter validates the observed pinned representation
and records this discrepancy in its audit.

## Leakage and length audit

With the pinned BarunLM-35M tokenizer revision
`ef3e483a9fd7d906ecf2a7929babeffaf82d1d16`, every selected row fits the 2,048-token contract.
Maximum prompt/target/total-with-EOS lengths are 1,170/317/1,183 for train and 1,085/263/1,098 for
development. No row is truncated or silently dropped.

The official random split has substantial template overlap even though exact structured-context and
dialogue group overlap is zero:

- Exact normalized input: 711 shared values, affecting 2,638 train and 1,227 development rows.
- Target-informed delexicalized family: 1,782 shared families, affecting 6,305 train and 3,000
  development rows.
- Whitespace 13-grams: 178 shared, affecting 108 train and 84 development rows.
- Exact example IDs, structured-context IDs, and context-plus-dialogue group IDs: zero overlap.

These measurements make the official development set useful for product-style PRESTO adaptation,
but they are not evidence of a leakage-controlled blind breakthrough. A scientific grouped
comparison must use a separately frozen grouped split or a new hidden human set.

Pinned conversion artifact hashes from the full English run:

- Train manifest: `49a553ea37a981f2a8009ce8bc6575fd56182d8f959352c9815fcc463b832e06`.
- Development manifest: `1e34c5554480f78e3c0616ef36207380f0973ab581f37a61ea33cb7122795c42`.
- Audit: `8e1f035809c14baf6f247f1bd8bf0f4b874ff4b117dc6dd217fbc732c1d8b7c4`.

Run the adapter with `python -m barunlm.datasets.presto --archive presto_v1.zip --output-dir
<new-directory> --tokenizer-json <pinned-tokenizer.json>`. Outputs are immutable `train.jsonl`,
`dev.jsonl`, and `audit.json`; there is deliberately no test-export API.
