# Frontend — Bilingual CRAG Assistant (Next.js 16, React 19, Tailwind 4)

Two-view chat client over the FastAPI backend in `../app/`:

- **Chat view** — SSE-streamed answers, source-cards split by origin
  (DB / Web / Cache), thumbs feedback per answer, a semantic-cache toggle in
  the header, and suggestion chips on the welcome screen.
- **Pipeline Inspector view** — a grid of agent cards (Query Rewrite /
  Retriever / CRAG Grader / Web Search / Query Router / Generator) with live
  status from the same SSE stream, a security-verdict strip
  (Input / Document / Output guards), and a per-card drawer showing static
  backend configuration alongside live request data.

State persistence is 100% client-side (`localStorage`); backend Redis is only
the 4-turn sliding-window conversation memory used by the follow-up rewriter.

## Tech stack

- Next.js 16 (App Router, Turbopack, `output: "standalone"`)
- React 19, TypeScript 5
- Tailwind 4 with `@theme inline` color tokens
- Hand-rolled minimal components (no component library)
- `clsx` + `tailwind-merge` for class composition, `lucide-react` for icons

## How to run

The frontend is part of the Docker Compose stack — see the top-level
[`../README.md`](../README.md); one command brings backend + frontend up
together.

Standalone (frontend only, against an already-running backend):

```bash
cd frontend
pnpm install --frozen-lockfile   # pnpm 11.8+, requires Node 22.13+
pnpm dev                         # http://localhost:3000
```

`NEXT_PUBLIC_API_BASE` defaults to `http://localhost:8000` and is inlined at
build time — override it if the backend lives elsewhere.

## Engineering notes

- **React-19 / Next-16 strict-mode side-effect race** in the typewriter and
  thumbs-feedback hooks — updater functions must stay pure; side effects are
  deferred rather than run inside a setState updater.
- **PDF Cyrillic-shift fix at the render layer** (no re-indexing) —
  `lib/utils.ts:unshiftCyrillic`, applied to source text before display.
- **Light-mode soft-hue text tokens** — sage / dusty-rose / rust get a
  darker `-text` variant in light mode for readability, without regressing
  dark mode.
- **Intentional chat scroll** — no auto-follow; a one-shot scroll on each new
  user message keeps the reading position stable during long streamed answers.
