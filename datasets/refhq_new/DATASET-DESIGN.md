# Dataset design: `pretrain/refhq-new`

**purpose:** Filtered instruct-sourced CE corpus for OLMo-2 370M reference / rho-1 scoring, tuned toward the 20-label OLMES BPB suite (not IFEval/tools/safety).

**family:** `pretrain`  
**profile:** `pretrain-tokens/v1` (tokens only; no raw-text companion is staged)  
**name:** `refhq-instruct` → `pretrain/refhq-instruct`  
*(The FarmShare scratch layout keeps the `refhq-new` directory name; `refhq-new` was not
usable as the dataset id itself, since `new` collides with a version-token convention.)*  
(No size suffix — realized size is whatever one pass yields.)

**tokenizer:** HF `allenai/dolma2-tokenizer` → publish dep `tokenizer/dolma2-bpe` (EOS 100257). Already published; do not republish.

**budget:** One pass of filtered unique data only. No upsampling. Count tokens during build and report realized size in plan summary + the local `dataset_manifest.json` `sources[]`.

**dedup:** None (explicit). Keep SmolTalk `openhermes-100k` even though it overlaps full OpenHermes-2.5.

---

## Irreversible decisions

| Decision | Choice |
|---|---|
| slice path | `tokens/<source>/<domain>/<split>-NNNNN.u32le.bin` (two path labels) |
| held-out | **0.15% of documents per `(source, domain)` before tokenize** (seed 42), then tokenize train/val separately |
| dtype / ext | `uint32` / `.u32le.bin` |
| target shard size | ~1 GiB; no `-of-N` in filenames |

### Path labels

- **source:** `tulu-v2` \| `openhermes-25` \| `tulu-3` \| `hermes-3` \| `smoltalk` \| `dolci`
- **domain:** `general` \| `math` \| `code` \| `science` \| `chat` — from row metadata / SmolTalk config; default `general`

---

## Inclusion / exclusion rules

Metadata drops run **before** Dolma English. Rules live in [`exclusion_rules.yaml`](exclusion_rules.yaml); helpers in [`exclusion.py`](exclusion.py). Domain labels: [`domain_map.py`](domain_map.py).

| Source | Keep | Drop |
|---|---|---|
| **Tulu-v2** | all | — |
| **OpenHermes-2.5** | all, including rows with null/missing `language` | rows with `language` set and not en/eng/english |
| **Tulu-3** | FLAN, WildChat, personas math/GSM/code/algebra, Numina-TIR, Evol CodeAlpaca, SciRIFF, TableGPT, No Robots, OASST, hardcoded | `wildguardmix`, `wildjailbreak`, `coconot`, `tulu-3-sft-personas-instruction-following`, `aya` / Aya |
| **Hermes-3** | all (~959K; no category column) | — |
| **SmolTalk** | all configs except listed | `apigen-80k`, `smol-constraints` |
| **Dolci** | everything else | `domain == Safety`; Precise IF; CoCoNot; Aya; WildGuard/WildJailbreak; Tool Use `source_dataset`; any row with non-null `function_calls`/`functions` |

**As built (`refhq-new-v1`, published internally as `pretrain/refhq-instruct` v3), two rows differ from this design:**

- **SmolTalk.** The download kept `all` together with the 11 configs other than
  `apigen-80k` and `smol-constraints` (`logs/refhqn-download-1674284_4.out`), and
  normalization streamed `config=all` first. The aggregate config contains every
  subset, so `apigen-80k` and `smol-constraints` are included once and the other 11
  configs twice (1,043,917 + 926,349 = 1,970,266 kept rows; 1.607B tokens).
- **Tulu-3.** The `tulu-3-sft-personas-instruction-following` needle matched nothing:
  the Persona IF rows come from `ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980`
  with `personahub_…` ids, so that subset (29,980 rows) is retained. The 211,103 dropped
  rows are CoCoNot, WildJailbreak, WildGuardMix and Aya, plus 120 rows whose random ids
  contain `aya`.

The realized corpus is about 3.9B tokens (paper Table 6). We do not redistribute it; rebuild
it byte for byte with [`../manifests/refhq-new-v1/`](../manifests/refhq-new-v1/).

### Dolma English filter (all kept docs)

Configs under [`configs/`](configs/); tag+mix wrapper in [`process.py`](process.py) (mirrors RefHQ).

- **Taggers:** `ft_lang_id_en_paragraph_with_doc_score_v2` (optional companion tagger `ft_lang_id_en_doc_v2`)
- **Mix include:** document English score (`…__doc_en`) ≥ **0.5**
- No Gopher/C4/toxicity/NSFW — English only
- Conversations → Dolma docs: flatten `messages`/`conversations` to plain text (role labels optional but consistent); one example = one document

Drop Tulu-3/Dolci Aya via metadata **before** Dolma so those rows never hit lang-id.

---

## Layout

```
tokens/<source>/<domain>/<split>-NNNNN.u32le.bin
```

Staged locally under `RUN_DIR/publish-stage/` (see `publish_refhq_new.py`); no raw-text
companion, no remote object store.

---

## Dependencies

- **tokenizer:** `tokenizer/dolma2-bpe` — must already be published
- **parent:** none

---

## Deferred (backfillable, don't block)

`about` / rich mix table / measured `sources[]` token counts after the build.

---

## Local layout split (`publish_refhq_new.py`)

The dataset is not published to a remote object store. `publish_refhq_new.py` builds the
`tokens/<source>/<domain>/` layout under `--stage-dir` and writes a local
`dataset_manifest.json` recording `dataset_id`, `purpose`, `about`, `notes`, `license`,
`limitations`, and measured per-source token counts (`sources[]`):

```python
dataset_id = "pretrain/refhq-instruct"
purpose = (
    "One-pass filtered instruct mix (Tulu-v2/OH-2.5, Tulu-3/Hermes-3, SmolTalk/Dolci) "
    "for OLMo-2 370M CE reference / rho-1; tool/safety/IF/Aya removed; Dolma English; "
    "tuned for 20-label OLMES BPB"
)
tokenizer = "tokenizer/dolma2-bpe"
license = {"id": "ODC-By-1.0", "basis": "declared"}
notes = "No dedup. No upsampling. Realized size is one filtered pass."
limitations = [{"kind": "license", "detail": "Tulu ODC-BY with some NC subsets; research use"}]
```

**Licensing, as stated in the token selection paper (Appendix A).** The single
`ODC-By-1.0` declaration above does not cover every source. Tulu-v2, Tulu-3 and
Dolci-Instruct-SFT are ODC-By v1.0; Hermes-3 is Apache-2.0; SmolTalk has no dataset-level
license (its author-created subsets are Apache-2.0); OpenHermes-2.5 declares no license,
so treat it as research use only. Non-commercial subsets: GPT4-Alpaca and Code-Alpaca
(CC-BY-NC-4.0) and LIMA (CC-BY-NC-SA) within Tulu-v2, and No Robots (CC-BY-NC-4.0) within
Tulu-3.
