"use client";

// All corpus-specific copy lives in lib/content.json -- see its _comment.
import suggestionsData from "@/lib/content.json";
import { cn } from "@/lib/utils";
import { Matryoshka } from "@/components/matryoshka";

interface Chip {
  label: string;
  query: string;
  tag: string;
}

interface Welcome {
  title: string;
  subtitle: string;
}

interface SuggestionChipsProps {
  onPick: (query: string) => void;
}

export function SuggestionChips({ onPick }: SuggestionChipsProps) {
  const chips = (suggestionsData.chips ?? []) as Chip[];
  const welcome = (suggestionsData.welcome ?? {
    title: "Hallo!",
    subtitle: "Frag etwas.",
  }) as Welcome;
  return (
    <div className="flex flex-col items-center gap-6 text-center">
      <div className="flex h-20 w-16 items-center justify-center rounded-full bg-accent/10 text-accent">
        <Matryoshka size={64} ariaLabel="Matrjoschka -- Bilingual Assistant" />
      </div>
      <div className="max-w-md">
        <h1 className="text-2xl font-semibold text-foreground">
          {welcome.title}
        </h1>
        <p className="mt-2 text-sm text-muted">{welcome.subtitle}</p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        {chips.map((chip) => (
          <button
            key={chip.label + chip.tag}
            onClick={() => onPick(chip.query)}
            className={cn(
              "rounded-full border border-border bg-surface px-4 py-2 text-sm text-foreground transition-colors",
              "hover:border-accent hover:bg-accent/10 hover:text-accent",
            )}
            title={chip.query}
          >
            {chip.label}
          </button>
        ))}
      </div>
    </div>
  );
}
