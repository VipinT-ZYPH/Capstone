Github Repo - https://github.com/VipinT-ZYPH/Capstone.git 

# VoiceOps 2026 — ZentroMart

This project builds the VoiceOps capstone customer-support agent for the
synthetic ZentroMart dataset. The implementation is being delivered in phases
so every foundation is validated before the agent, retrieval, tools, and voice
layers are added.

## Phase 1: data foundation

Phase 1 adds a read-only dataset repository that:

- discovers the supplied extracted dataset without changing it;
- loads CSV values as text, preserving order IDs, SKUs, pincodes, and references;
- parses all KB frontmatter and `[SECTION-ID]` headings;
- exposes the fixed 20 October 2026 reference-date calendar helper; and
- runs the supplied structural and relational validator.

Run from the project directory:

```powershell
$env:PYTHONPATH = "src"
python scripts\validate_phase1.py
python -m unittest discover -s tests
```

The expected validator output is a PASS for 9 CSVs, 21 KB documents, and the
verification checks. The supplied dataset under
`Capstone_Project_Dataset_FTTT_2026\` is synthetic and must not be edited.

## Step 1: prompt-only text chatbot

Install dependencies and configure the API key:

```powershell
cd C:\Users\Admin\OneDrive\Desktop\Projects\Capstone
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY
```

Run the 52-question batch:

```powershell
$env:PYTHONPATH = "src"
python scripts\run_step1.py --output outputs\step1_results.json
```

The JSON output contains every question, generated answer, model decision,
expected decision, required-fact checks, forbidden-claim checks, and summary.
The script uses Google's Gemini API through the official `google-genai` SDK; it does not send expected
answers or evaluation labels to the model. A missing API key fails explicitly.

Run tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Step 3: conversational support and tools

`voiceops.conversation.ConversationalRAGAgent` wraps the existing RAG chatbot
with isolated state per `conversation_id`. It processes turns in order,
retains verified order context and prior actions, calls the supplied
`lookup_order` and `create_ticket` functions, and records retrieved evidence
and tool results for every turn. Order details are disclosed only after a
successful phone-last-four or delivery-pincode verification. Verification is
limited to one retry, ticket creation is idempotent per conversation, and tool
or model failures are returned as unscored structured turn records.

The supplied conversation dataset can be processed offline with a mocked RAG
client:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The current tests cover context isolation, follow-ups, identity verification,
policy-driven refusal and escalation, ticket failures and deduplication,
handover summaries, API failures, and evidence provenance. No live Gemini
conversation evaluation is performed by the tests.

Run the dedicated deterministic Step 3 scorer:

```powershell
$env:PYTHONPATH = "src"
python scripts\run_step3_evaluation.py
```

It writes `outputs\step3\step3_conversation_per_turn.json` and
`outputs\step3\step3_conversation_report.json`. The report keeps decision,
required-fact, forbidden-claim, tool, identity, ticket, handover, evidence,
context, and failure metrics separate, each with an explicit numerator and
denominator. Evidence recall is scored only when an active section ID is
required by `gold_evidence` and the turn actually invokes the RAG flow;
tool-only identity-verification turns are non-applicable. Non-applicable
identity, ticket, handover, and context checks are excluded from their
denominators. Tool arguments are
normalized for whitespace and argument ordering while tool-call order remains
significant. The runner uses a deterministic mocked model and synthetic tool
adapters; its metrics are labeled `deterministic_mock`, validate the evaluation
pipeline, and are not live Gemini accuracy.

## Planned phases

1. Data foundation and dataset validation (complete).
2. Prompt-only text chatbot and 52-question evaluation (complete).
3. KB chunking and policy-aware RAG retrieval (complete).
4. LLM conversation agent, identity-gated order lookup, and ticket escalation (offline implementation complete).
5. Gnani `gnani-vachana` STT/TTS integration foundation (offline complete;
   live audio validation pending).
6. Prompt-only versus RAG evaluation and multi-turn evaluation.

## Step 2: policy-aware RAG

Step 2 indexes every `[SECTION-ID]` section from `kb\*.md`, retaining its
document path and frontmatter metadata. Retrieval is dependency-free lexical
retrieval over section titles and text. Current policy sections receive
precedence over agent guides, help, FAQs, and marketing pages; sections marked
`status: superseded` are excluded from customer answers.

Run the prompt-only baseline and RAG comparison over all 52 practice questions:

```powershell
cd C:\Users\Admin\OneDrive\Desktop\Projects\Capstone
$env:PYTHONPATH = "src"
python scripts\run_step2.py --output outputs\step2_results.json
```

The output contains separate baseline and RAG results, generated answers,
decisions, required-fact checks, forbidden-claim checks, retrieved section
IDs, metadata, scores, evidence recall against `gold_evidence`, and summaries.
Both modes use the same questions and model configuration. Expected answers and
evaluation labels are kept out of the LLM prompt.

Run all tests and the supplied dataset validation:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python Capstone_Project_Dataset_FTTT_2026\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
```

## Gemini configuration and result verification

Copy `.env.example` to `.env` only if you do not already have one, then set:

```dotenv
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_REQUEST_DELAY_SECONDS=1.0
```

The migration does not overwrite `.env`. API keys are read from environment
variables and are never stored in source code or output logs. Both runners use
the configured Gemini model for baseline and RAG generation. The request delay
is applied between practice questions; retries use bounded exponential
backoff with jitter, while exhausted daily quotas are not retried.

## Gemini diagnostics and small tests

Do not start the 52-question run while debugging. Use one request first:

```powershell
cd C:\Users\Admin\OneDrive\Desktop\Projects\Capstone
$env:PYTHONPATH = "src"
@'
from pathlib import Path
from dotenv import load_dotenv
from voiceops.chatbot import GeminiChatClient, ZentroMartChatbot
from voiceops.config import Settings
load_dotenv(Path(".env"))
s = Settings.from_environment(Path.cwd())
bot = ZentroMartChatbot(GeminiChatClient(s.llm_api_key), s.llm_model)
print(bot.answer("SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?"))
'@ | python -
```

Then run a three-question evaluation without changing the dataset:

```powershell
$env:PYTHONPATH = "src"
@'
from pathlib import Path
from dotenv import load_dotenv
from voiceops.chatbot import GeminiChatClient, ZentroMartChatbot
from voiceops.config import Settings
from voiceops.dataset import DatasetPaths, DatasetRepository
from voiceops.evaluation import evaluate_questions, records_as_dicts
import json
load_dotenv(Path(".env"))
s = Settings.from_environment(Path.cwd())
r = DatasetRepository(DatasetPaths.discover(Path.cwd()))
bot = ZentroMartChatbot(GeminiChatClient(s.llm_api_key), s.llm_model)
rows = evaluate_questions(r.practice_questions()[:3], bot,
                           request_delay_seconds=s.request_delay_seconds)
print(json.dumps(records_as_dicts(rows), ensure_ascii=False, indent=2))
'@ | python -
```

Each failed request now records `error_category`, `error_type`,
`error_status_code`, `error_api_code`, and a credential-redacted
`error_message`, while
`scored` is `false`. This distinguishes quota exhaustion, requests-per-minute
rate limits, temporary service failures, authentication errors, invalid model
names, and timeouts.

Offline Step 2 retrieval validation currently covers **52/52 active evidence
requirements** on the supplied practice-question dataset. This is an evidence
coverage result only; it does not represent 100% answer accuracy. No
question-specific answer or evidence ID is hardcoded.

Run Step 1:

```powershell
cd C:\Users\Admin\OneDrive\Desktop\Projects\Capstone
$env:PYTHONPATH = "src"
python scripts\run_step1.py --output outputs\step1_gemini_results.json
```

Run Step 2:

```powershell
cd C:\Users\Admin\OneDrive\Desktop\Projects\Capstone
$env:PYTHONPATH = "src"
python scripts\run_step2.py --output outputs\step2_gemini_results.json
```

Verify a real Gemini run by checking the output JSON:

```powershell
$step1 = Get-Content outputs\step1_gemini_results.json -Raw | ConvertFrom-Json
$step2 = Get-Content outputs\step2_gemini_results.json -Raw | ConvertFrom-Json
$step1.model
$step1.summary
$step2.model
$step2.baseline.summary
$step2.rag.summary
($step1.results | Where-Object { $_.scored }).Count
($step1.results | Where-Object { $_.decision -eq "error" }).Count
```

`model` should be the configured `gemini-*` model, scored counts should be
greater than zero, and error counts should be zero for a clean run. API
failures are retained as records with `decision: "error"`, a non-empty
`error`, and `scored: false`; they are not counted as incorrect model answers.
Quota, rate-limit, timeout, and transient service failures are retried with
bounded exponential backoff. Authentication and non-transient failures are
reported immediately without exposing the API key.

Voice will use only Gnani's official
`gnani-vachana` SDK; no alternate speech provider is supported.

## Step 4: voice integration foundation

The offline voice layer is in `voiceops.voice`. `VoiceConversationService`
accepts audio bytes, sends them through the official
`gnani.stt.GnaniSTTClient.transcribe_bytes` contract, passes the resulting
transcript once to `ConversationalRAGAgent`, and sends the agent's response
through the official `gnani.tts.GnaniTTSClient.synthesize` contract. Session
IDs are passed unchanged to the existing isolated conversation state.

The service retries only STT and TTS stages with a bounded attempt count; it
never retries the conversational agent or its tools. Retries are limited to
timeouts, connection failures, and transient HTTP service responses; auth,
quota, invalid-request, and invalid-audio failures are not retried. This
prevents duplicate order lookups and tickets. Tests use protocol-compatible
mocks and make no network requests.

For the capstone streaming path, `GnaniSTTStreamingAdapter` uses the SDK's
`GnaniSTTStreamClient` WebSocket lifecycle and joins completed transcript
events into one agent turn. `GnaniTTSStreamingAdapter` uses the documented
`GnaniTTSRealtimeClient.synthesize_events` WebSocket stream and exposes audio
chunks through `VoiceConversationService.iter_audio_stream`. Playback clients
can consume each `VoiceAudioEvent.audio` value as it arrives and stop when an
event has `complete=True` (the completion event has empty audio). The existing
`process_audio_stream` method remains available and collects those events into
the original `StreamingVoiceTurnResult` shape. The agent runs exactly once
after STT completion. A TTS retry is allowed only before the first chunk is
delivered; after partial delivery, failures are propagated rather than
duplicating audio. A stream retry can still repeat provider-side work if a
network failure occurs before delivery after the provider accepted the
request; the SDK does not provide exactly-once guarantees, so the voice layer
does not claim them.

Required environment variables for a later live run:

```dotenv
GNANI_API_KEY=your-key
GNANI_BASE_URL=https://api.vachana.ai
```

The current implementation does not perform a live request. Before a live
test, record a supported audio file or raw PCM stream as described by the
official Gnani documentation, choose an STT language code, and verify the
configured TTS voice/model and output audio format against the installed SDK
