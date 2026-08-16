# Project Rules — Read Before Doing Anything

This file governs how Claude Code operates in this repository. These rules exist
because of specific, documented incidents in past sessions. They are not
generic caution — they are direct responses to real mistakes that cost real time.

## Non-negotiable rule: ask first, show the diff, then write locally

**You may write directly to files in this project folder — but only after you
have shown me exactly what will change and I have explicitly said go ahead.
No exceptions, no "small fix" carve-outs, no writing first and explaining
after.**

For every change, before touching any file:

1. Tell me which file(s) you intend to change.
2. Show me the actual diff or exact new content — not a summary of it, the
   real text/code — so I can see precisely what will land in the file.
3. Wait for explicit approval ("go ahead", "yes", "do it" — not silence, not
   an unrelated reply).
4. Only then write the change to disk, directly in this folder.

This applies to everything, including:

- Config values, hyperparameters, constants (e.g. `NUM_WORKERS`, learning rate,
  batch size, resolution, patience, seeds)
- "Helpful" additions I did not ask for (auto-adjustment logic, extra error
  handling, refactors, new abstractions)
- Docstrings, comments, README/CHANGELOG content
- File/variable renames
- Anything framed as a "small fix" or "obvious correction"

If something looks broken or wrong while you're reading code, **stop and tell
me what you found**. Do not fix it silently, even if the fix seems trivial or
obviously correct. I decide what gets changed — you write it only after I say
so, and only after I've seen the actual diff.

Exception: pure read-only investigation (viewing files, running non-mutating
commands like `nvidia-smi`, `git status`, `git diff`, `pytest --collect-only`,
and fetching data from the internet — documentation, papers, package/repo
pages, anything read-only) does not need pre-approval, ever. You may search
and fetch web content freely and at any time, without asking first — this
directly supports the Verification standard below and should not be gated.
This exception does not cover actually *running* tests — see Testing section
below, that's mine to run, always.

## Why this rule exists

Specific past incidents in this project:

- `NUM_WORKERS` was silently changed from my intended 8 down to 4 in a past
  session, with no notice given. It took hours to track down why performance
  had degraded.
- A batch-size auto-adjustment feature was added that I never asked for. I had
  to discard it and reimplement the behavior myself, my way.
- Bugs were identified during review but *not* fixed when I explicitly asked
  for fixes to be made — and separately, changes were made to things I hadn't
  asked to be touched.
- A technical rationale was fabricated and presented as settled fact (why
  quantization functions existed only in one optimizer's files) rather than
  saying "I don't know, let me check" — the real answer was simple script
  drift, not a design decision.
- Corrections were verbally agreed to and then not actually followed in
  subsequent actions within the same session.

Do not repeat any of these patterns.

## Verification standard

**If there is any doubt at all, look it up online before acting or answering.
Don't rely on training data alone for anything that could be outdated, version-
specific, or simply misremembered. You never need to ask permission to search
or fetch web content — that's covered by the read-only exception above and
should happen freely, as often as needed, without checking in first.**

Documented failure, recurring: flagged once already in session memory
(2026-08-02, zero proactive searches performed an entire session) and again
on 2026-08-07 in this session — no search happened until directly told "you
did not pull any internet data." The instruction is proactive ("look it
up... before acting"), not reactive. A memory note from the first occurrence
did not prevent the second; this paragraph is the second attempt at making
it stick.

- Always verify — search for and check current documentation, or pull
  relevant internet data, before writing code that uses a library/API/
  function, no matter how certain you feel. Do not gate this behind your
  own judgment of whether you're "certain enough" — that self-assessment is
  exactly what fails. "I've seen this before" is not the same as "I
  verified this is still correct for the version in use here." Do not guess
  and present the guess as fact, and do not skip verification because it
  feels unnecessary this time.

  Documented failure, 2026-08-08: diagnosed a plain Python `TypeError`
  (missing positional argument) from source alone and skipped a web search
  on the reasoning that this was "basic language semantics, not a
  library/API/version question" — narrowing what counts as a checkable
  claim, instead of narrowing self-judged certainty. Corrected directly:
  "the use of internet data is not dependent on syntactics of code, its
  use is explicit any and all verification." Deciding a claim is too
  basic/self-evident/language-level to need a source is the same
  self-gatekeeping the "do not gate this behind your own judgment" line
  above already forbids — it just moved up one level, from "how sure am I"
  to "does this even count."
- This applies especially to PyTorch, optimizer implementations (SOAP,
  AdaHessian, Schedule-Free AdamW, etc.), CUDA/driver behavior, and anything
  else that changes between versions — check the actual current docs/source
  rather than trusting memory, even if you're confident.
- If you don't know why something in the code is the way it is, say so
  explicitly. Do not construct a plausible-sounding explanation and state it
  as fact. "I don't know, here's what I can confirm and here's what's
  speculation" is always the correct answer over a confident guess.

  Documented failure, 2026-08-07: asserted that a router model's architecture
  was "carried forward from older code" — a plausible, tidy-sounding
  narrative — before checking the file. It turned out to be true (confirmed
  after the fact via the script's own docstring), but being right by
  coincidence doesn't make constructing the claim first and verifying second
  acceptable.

- When asked to verify something (e.g. "check all 5 files are consistent"),
  actually check all of them — don't sample and generalize. "Check" here means
  read and review the actual file contents, not run anything; execution is
  always mine per the Testing section below.

  Documented failure, 2026-08-07: asked to check whether an architecture
  issue was consistent across all optimizers used in this project, ran a
  handful of targeted greps (3 lines per file), saw a consistent pattern, and
  presented it as if every file had been checked — when only a sample had.
  Caught and corrected by actually reading all 44 files' architecture
  definitions in full. The rule above already covered this; it wasn't
  applied until called out.

  Documented failure, 2026-08-08: verified 44 scripts' call-site
  consistency via targeted greps before editing 5 of them — the matched
  lines were accurate and covered every file, but grep only confirms a
  pattern exists on a line, not that the surrounding scope/control-flow is
  what it looks like. Grep is fine for locating candidates; before editing
  a file or asserting a conclusion from it, open it with Read. William had
  to say directly "do not grep at all, read the code every time" before
  this actually happened.

- When you do look something up, say what you checked (e.g. "confirmed via
  PyTorch docs" or "confirmed via the optimizer's official repo/paper") so the
  source is visible, not just the conclusion. This feeds directly into the
  changelog citation requirement below — verify first, then cite what you
  verified.
- Before presenting anything as a source's own words — in quotation marks,
  or described as "the paper says X" / "the docs state X" — confirm it's
  actually verbatim from that primary source, not a tool's synthesis of it.
  This applies to every retrieval tool equally: WebSearch returns an
  aggregated summary across multiple results, not any single source's own
  text; WebFetch returns a fetching model's summary of a page, which can
  include its own inferences sitting right next to actual quoted material.
  Neither is the primary source itself. If exact wording matters (e.g. for a
  changelog citation), fetch or quote the primary text directly and confirm
  the quoted span is really there, not paraphrased.

  Documented failure, 2026-08-07 (twice, same session): first, early
  research relied on WebSearch's synthesized summaries and presented their
  conclusions as if read directly from the papers, before being pushed to
  actually fetch primary sources. Later, a WebFetch summary included one
  genuinely quoted sentence from a paper plus one unquoted sentence of the
  tool's own inference about why the paper did what it did — both were then
  presented back as the paper's own words, one in quotation marks with
  "(their words)" attached, when only the first sentence was actually the
  paper's text.
- Literature/documentation search resolves version-specific or checkable
  facts — it does not, and cannot, "prove" that a fix will work in this
  specific codebase. If pushed to "verify more" on a claim that external
  sources can only ever corroborate rather than settle (e.g. "is this a good
  general design pattern" vs. "does this specific PyTorch flag exist in this
  version"), say that distinction out loud once, instead of running more
  searches of the same kind and re-declaring the result "thorough."

  Documented failure, 2026-08-07: repeatedly told "you didn't fully verify,"
  ran three more rounds of paper searches on a general architecture-design
  claim before naming that no citation count converts general corroboration
  into proof for this repo's specific case — a distinction that should have
  been stated the first time, not the fourth.

## Scope discipline

- Do exactly what is asked. Not more, not less.
- If a request is ambiguous, ask what I mean rather than picking an
  interpretation and running with it.
- If completing a task would require touching a file or system I didn't
  mention, stop and ask first — don't assume it's implied.

**Don't assert conclusions about whether something is "fine," "valid,"
"acceptable to keep," or otherwise adequate without first confirming what it
actually needs to serve.** This is the same failure as picking an
interpretation of an ambiguous request and running with it — except here the
ambiguity is about *purpose*, not the literal wording of a request. If I
don't know why a project exists, what a set of trained models is actually
for, or what "done" means for this specific task, say so and ask — don't
default to "it still technically works, so it's probably fine."

Documented twice now, five days apart: 2026-08-02 (`git checkout` reverting
a bad edit without asking, surfaced only when William asked directly whether
something unauthorized had happened) and 2026-08-07 (asserting that
architecturally-inefficient trained AdamW/Router models were "not invalid,
just reflect old architecture, fine to keep for now" without knowing what
the project's models are actually for). Both times the rule was already
written down here and simply wasn't consulted at the decision point — only
cited reactively, after being caught. A memory note from the first incident
didn't prevent the second, so this now lives in the file itself: check
before asserting, not after.

## System-specific paths

- Dataset paths and output paths in this project are hardcoded to my local
  machine on purpose. Do not "fix" them, make them relative, or add
  auto-detection logic unless I ask for that specifically.
- If you add new files that reference paths, use the existing path constants
  already defined in the codebase — don't invent new ones.
- If a path needs a note for portability (e.g. for eventual repo sharing), ask
  before adding one, and keep it to a comment, not a behavior change.

## Changelog requirement

**Every approved change gets a CHANGELOG entry. No exceptions.**

- After I approve a change and you make it, immediately add an entry to
  `CHANGELOG.md` in the same turn — don't wait to be asked, don't batch it up
  for "later."
- One entry per change, even for small ones. A one-line config tweak still
  gets a line.
- Each entry should include: date, file(s) touched, what changed, and why
  (one sentence is fine — "why" can just be "per instruction" if that's all
  it is).
- **Cite the source that justified the change.** If the reasoning came from
  official documentation, a paper, or another external reference, name it
  (e.g. "per PyTorch docs on `torch.cuda.amp`" or "per Schedule-Free AdamW
  paper, warm-up requirement"), ideally with a link. If it came from reading
  the actual code/output in this repo (a log, an error message, another file),
  say which file(s) or output you looked at. If it was my direct instruction
  with no other source, say that plainly ("per William's instruction — no
  external source"). Never cite a source you didn't actually check — if you
  can't point to where the reasoning came from, say so instead of inventing
  a plausible-sounding citation.
- If a single approved request results in multiple distinct changes across
  files, log each one separately rather than one vague combined line.
- Never edit or delete a past changelog entry to "clean it up" — the log is a
  record, not a draft. If something in an old entry was wrong, add a new
  entry correcting it; don't rewrite history.
- If `CHANGELOG.md` doesn't exist yet in a given directory where you're
  making a change, ask whether to create one there rather than assuming.

## README maintenance

**Keep README.md accurate to what's actually in the folder.**

- Whenever an approved change adds, removes, or renames a file, script, model
  output, or directory that the README describes or should describe, update
  the README as part of that same change — don't leave it stale.
- Before updating it, actually look at the current folder structure on disk
  rather than assuming from memory what files exist — list the directory,
  confirm what's really there, and reflect that.
- This is still subject to the ask-first rule above: propose the README
  change alongside the code change you're asking me to approve, don't push it
  through silently afterward.
- If you notice the README is already out of sync with the actual file
  structure (independent of anything you're currently changing), tell me —
  don't fix it without asking, same as any other file.
- Same changelog rule applies: a README update gets its own changelog entry
  when it happens.

## Testing

**I run all tests. You do not run tests, ever — including short ones.**

- Do not execute test suites, training runs, or verification scripts yourself,
  regardless of expected duration. Propose what should be tested and how, and
  I will run it.
- If a task involves anything requiring more than ~5 minutes of GPU time
  (training, extended inference, benchmarking), that decision — whether to run
  it at all, and when — is mine. Tell me what you'd want run and why; do not
  kick it off yourself.
- You may still read existing logs, past run outputs, or checkpoint metadata
  to inform your work — that's not the same as running something new.

## Git / GitHub

**Do not run `commit`, `push`, `pull`, `fetch`, `merge`, `rebase`, `reset`, or
any other git operation that changes repo state or touches GitHub, unless I
explicitly ask for that specific operation in that moment.**

- Read-only git commands (`git status`, `git diff`, `git log`) are fine
  anytime, per the exception above.
- "I finished the change, want me to commit it?" is fine to ask — actually
  running the commit without me saying yes is not.
- This applies regardless of how obviously safe or routine the operation
  seems.

## Commit message style

**Commit messages must stand on their own — someone reading `git log`
cold, without ever opening `v3_CHANGELOG.md`, must be able to tell what
happened and why from the commit message alone.**

- Write a real subject line plus a body, not a single line of
  semicolon-joined clauses. A dense one-liner reads fine to whoever was
  in the conversation when it was written, but is opaque to anyone
  reading it later without that context.
- The body explains what actually happened in plain language first —
  what broke, what the real-world symptom was — then what was done
  about it and why. Not just an inventory of files/fields touched.
- Don't assume the reader will cross-reference the changelog to
  understand the commit. The changelog is the detailed record; the
  commit message is the standalone summary, and has to work on its own.
- Since I run all git commands myself (see Git/GitHub above), this
  applies to any commit message you draft for me to use — get it right
  before handing it over, not as a first pass to be iterated on.
- Drafted commit messages contain only the subject + body content
  above — nothing else. No trailers, signatures, or other additions
  (e.g. a `Co-Authored-By: Claude...` line) unless I explicitly
  instruct you to include one. I run every commit myself under my own
  git identity; there is no co-authorship to record unless I say so.

Documented, 2026-08-11: a drafted commit message written as dense,
jargon-heavy semicolon-joined clauses ("raise Windows pagefile to 128GB
max after a DataLoader shared-memory crash (error 1455)") was flagged
directly — anyone reading it without having read the changelog first
would be confused about why any of it happened. A second attempt at
"improving" it turned out to be cosmetic rewording of the same terse
structure, not an actual fix — caught and called out separately. The
real fix was a proper subject line plus a prose body explaining the
crash and the fix in plain terms, not a jargon-compressed summary.

Documented, 2026-08-16: a `Co-Authored-By: Claude Sonnet 5
<noreply@anthropic.com>` trailer was included in drafted commit
messages three separate times in this project before actually being
fixed — twice in a single 2026-08-14 session (once caught and dropped
by Claude itself mid-conversation, with an offer to save it to memory
that was never followed through on; a second time caught directly by
William on a different commit later that same session), and a third
time on 2026-08-16 in a new session, because the rule had never
actually been written down despite being discussed at length twice
already. William: "you are only to give em the comments to use in my
git commits not add unnecessary things to them like co authored by
unless i specifically instruct you to."

## Delivering finished work

**Writes happen directly in this project folder, after approval — not as
downloads. See the opening rule for the show-diff-then-write sequence.**

- Before presenting anything to me as done, fully verify the fix or feature
  actually works — end to end, not just "the code looks right."
- Don't tell me something is complete based on code review alone. If it needs
  to run to be verified, tell me it needs to run and let me run it (per
  Testing above) before calling it done.
- If you can verify a piece without running the full test suite (e.g. a
  syntax check, a dry-run with `--collect-only`, confirming imports resolve),
  do that — but be explicit about what was and wasn't actually verified.
- This applies to every file type without exception: code, config,
  CHANGELOG.md, README.md, everything. The diff-then-approve-then-write
  sequence in the opening rule is never skipped, regardless of file type.

## Worktrees

- If working in a worktree, do not merge, rebase, or otherwise push changes
  into `main` (or any other branch outside the worktree) without asking me
  first and getting explicit confirmation, per the Git/GitHub rule above.
- Once a worktree change is approved, write it directly following the same
  show-diff-then-write sequence as any other change. You never perform a
  merge into my main checkout, under any circumstance — merging is a git
  operation and falls under the Git/GitHub rule, not this one.

**I do not run, test, or interact with worktree copies at all — a worktree
is purely your own isolated workspace, invisible to my actual workflow.** I
work from my real checkout only (for this repo, `E:/mnist_v3`). If a task
happens to run in a worktree, don't treat writing there as equivalent to
delivering the work to me — either write the approved changes directly to
my real checkout too (still no merge/commit needed, just plain file edits
at that path), or ask which location I want before finishing up. Leaving
changes only in a worktree branch means, from my side, nothing happened.

Documented, 2026-08-07: wrote 5 files' worth of an approved fix into a
worktree copy before it came up that I never use worktree copies for
anything — the changes sat somewhere I'd never see them until I pointed it
out.

## Hardware / execution

- This environment should have access to local GPU (verify with `nvidia-smi`
  or a `torch.cuda.is_available()` check at the start of a session — flag it
  to me immediately if GPU is not visible, don't silently fall back to CPU and
  proceed). Checking availability is read-only and fine to do; actually
  running anything on the GPU falls under the Testing rule above.
- Don't kill, restart, or modify an in-progress training run without asking,
  even if it looks stalled or wrong — check with me first.

## When in doubt

Stop and ask. A clarifying question costs me ten seconds. An unauthorized
change costs hours to find and undo.

Six-plus documented incidents across two sessions five days apart (see the
incident notes under Verification standard and Scope discipline above) all
share one root cause: checking this file's rules *after* being caught
violating them, not before acting. Every one of those rules already existed
in writing at the time. The fix is not writing more rules — it's actually
running the action against them at the moment of deciding, every time,
including when the action seems small, obviously safe, or not worth
interrupting flow to double-check.
