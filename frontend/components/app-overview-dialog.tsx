"use client";

import { useEffect } from "react";
import { X } from "lucide-react";
import { Matryoshka } from "@/components/matryoshka";
import content from "@/lib/content.json";

// App overview modal, opened from the brand icon in the header (all screen
// sizes). Gives the 30-second summary: what the app is, what it is built on,
// and how a question flows through the CRAG pipeline.
// Landscape 16:9-ish two-column layout so the whole story fits one screen
// for the demo video; collapses to a single scrollable column on mobile.
//
// The left column (intro text + the four figures) describes whichever corpus
// the system is pointed at, so it is read from lib/content.json rather than
// written here. The right column describes the pipeline itself and is the
// same for every deployment, so it stays in the component.

interface OverviewContent {
  intro: string;
  stats: { value: string; label: string }[];
  statsNote: string;
}

interface AppOverviewDialogProps {
  onClose: () => void;
}

export function AppOverviewDialog({ onClose }: AppOverviewDialogProps) {
  const overview = content.overview as OverviewContent;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div className="fixed inset-0 bg-foreground/40 backdrop-blur-[2px]" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="App overview"
        className="relative max-h-[92vh] w-full max-w-5xl overflow-y-auto rounded-xl border border-border bg-background p-7 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-16 w-12 shrink-0 items-center justify-center rounded-lg bg-surface">
              <Matryoshka size={54} />
            </div>
            <div>
              <div className="text-[13px] uppercase tracking-wider text-accent">
                What is this app?
              </div>
              <h3 className="text-lg font-semibold leading-snug text-foreground">
                Bilingual German-Russian Language Learning Assistant
              </h3>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted hover:bg-surface hover:text-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-5 grid gap-6 md:grid-cols-2">
          {/* left column: what + for whom */}
          <div className="flex flex-col gap-4">
            {/* body copy sits on lighter surface panels (same pattern as
                the chat bubbles) + 90% foreground -- full #F2EEE3 directly
                on the darkest background glared in dark mode */}
            <div className="rounded-md border border-border bg-surface px-4 py-3">
              <p className="text-base leading-relaxed text-foreground/90">
                {overview.intro}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              {overview.stats.map((s) => (
                <Stat key={s.label} value={s.value} label={s.label} />
              ))}
            </div>

            <p className="text-xs leading-relaxed text-muted">
              {overview.statsNote}
            </p>

            <p className="rounded-md border border-border bg-surface px-3 py-2.5 text-sm leading-relaxed text-muted">
              The <span className="font-medium text-accent">Pipeline</span>{" "}
              tab shows every answer live as an agent grid -- the Overview
              button there opens the animated architecture diagram.
            </p>
          </div>

          {/* right column: how it works */}
          <div className="rounded-md border border-border bg-surface px-4 py-3.5">
            <div className="text-[13px] uppercase tracking-wider text-muted">
              How a question gets answered
            </div>
            {/* Bullets follow the REAL pipeline order: guard + rewrite +
                cache lookup first, then retrieval, then CRAG, then the
                query router, then the answer (which is written back into
                the cache). Mirrors the node order in pipeline-diagram.tsx. */}
            {/* strong lead-ins in the SAME accent orange as the stats
                numbers on the left -- consistent in both modes (bold
                white on white-ish body text had no pop in dark) */}
            <ul className="mt-2.5 flex flex-col gap-3 text-[15px] leading-relaxed text-foreground/90 [&_strong]:text-accent">
              <li className="flex gap-2.5">
                <Bullet />
                <span>
                  <strong>Guard, rewrite, cache:</strong> a security guard
                  screens the input, follow-ups become standalone questions
                  (conversation memory), and repeated questions are answered
                  from the semantic cache instantly (~35x faster).
                </span>
              </li>
              <li className="flex gap-2.5">
                <Bullet />
                <span>
                  <strong>Hybrid search:</strong> Voyage embeddings + BM25
                  search all books at once; a reranker picks the 10 best
                  passages.
                </span>
              </li>
              <li className="flex gap-2.5">
                <Bullet />
                <span>
                  <strong>CRAG grader:</strong> checks whether the books
                  really contain the answer -- if not, Tavily searches the
                  web instead.
                </span>
              </li>
              <li className="flex gap-2.5">
                <Bullet />
                <span>
                  <strong>Query router:</strong> picks the best answer
                  format -- factual, explanation or comparison.
                </span>
              </li>
              <li className="flex gap-2.5">
                <Bullet />
                <span>
                  <strong>Answer:</strong> written in the language of the
                  question, with [book, page] or web citations --
                  PII-checked and stored back into the cache.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2.5">
      <div className="text-2xl font-semibold text-accent">{value}</div>
      <div className="text-[13px] leading-snug text-muted">{label}</div>
    </div>
  );
}

function Bullet() {
  return (
    <span className="mt-[8px] inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
  );
}
