"use client";

import { useState } from "react";
import { BookOpenText, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { CacheToggleIcon } from "@/components/cache-toggle-icon";
import { CacheClearIcon } from "@/components/cache-clear-icon";
import { AppOverviewDialog } from "@/components/app-overview-dialog";
import { cn } from "@/lib/utils";

interface HeaderProps {
  activeView: "chat" | "pipeline";
  onChangeView: (v: "chat" | "pipeline") => void;
  useCache: boolean;
  onToggleCache: (next: boolean) => void;
  onOpenSidebar: () => void;
}

export function Header({
  activeView,
  onChangeView,
  useCache,
  onToggleCache,
  onOpenSidebar,
}: HeaderProps) {
  // The brand icon opens the app-overview modal on every screen size --
  // the elevator summary of what the app is (corpus, pipeline, audience).
  const [overviewOpen, setOverviewOpen] = useState(false);
  return (
    <header className="flex items-center justify-between gap-2 border-b border-border bg-surface px-3 py-3 sm:gap-4 sm:px-4">
      <div className="flex items-center gap-2 min-w-0 sm:gap-3">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Chat-Verlauf oeffnen"
          className="md:hidden"
          onClick={onOpenSidebar}
        >
          <Menu className="h-4 w-4" />
        </Button>
        <button
          type="button"
          onClick={() => setOverviewOpen(true)}
          aria-haspopup="dialog"
          aria-expanded={overviewOpen}
          aria-label="App-Ueberblick oeffnen"
          title="Was ist diese App?"
          className={cn(
            "flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent text-on-accent",
            "cursor-pointer transition-shadow hover:shadow-[0_0_0_2px_var(--accent-soft)]",
            "focus:outline-none focus:ring-2 focus:ring-accent/40",
            overviewOpen && "shadow-[0_0_0_2px_var(--accent-soft)]",
          )}
        >
          <BookOpenText className="h-4 w-4" />
        </button>
        <div className="hidden flex-col leading-tight min-w-0 sm:flex">
          <span className="text-sm font-semibold text-foreground truncate">
            Bilingual German-Russian Language Learning Assistant
          </span>
          <span className="text-xs text-muted truncate">
            CRAG: DE/RU Lehrwerke + Web-Fallback
          </span>
        </div>
      </div>

      <nav className="flex shrink-0 items-center gap-1 rounded-lg border border-border bg-background p-1">
        <TabButton
          active={activeView === "chat"}
          onClick={() => onChangeView("chat")}
        >
          Chat
        </TabButton>
        <TabButton
          active={activeView === "pipeline"}
          onClick={() => onChangeView("pipeline")}
          title="Live-Sicht auf Retriever, Router, Generator und Security-Guards"
        >
          Pipeline
        </TabButton>
      </nav>

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-0.5 rounded-lg border border-border bg-background p-1">
          <span className="hidden px-2 text-[10px] font-medium uppercase tracking-wider text-muted sm:inline">
            Cache
          </span>
          <CacheToggleIcon value={useCache} onChange={onToggleCache} />
          <CacheClearIcon />
        </div>
        <div className="flex items-center gap-0.5 rounded-lg border border-border bg-background p-1">
          <span className="hidden px-2 text-[10px] font-medium uppercase tracking-wider text-muted sm:inline">
            Theme
          </span>
          <ThemeToggle />
        </div>
      </div>
      {overviewOpen && (
        <AppOverviewDialog onClose={() => setOverviewOpen(false)} />
      )}
    </header>
  );
}

function TabButton({
  active,
  children,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  active: boolean;
}) {
  return (
    <button
      {...rest}
      className={cn(
        "px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
        active
          ? "bg-accent text-on-accent"
          : "text-muted hover:text-foreground hover:bg-surface",
        rest.disabled && "opacity-50 cursor-not-allowed hover:bg-transparent hover:text-muted",
        className,
      )}
    >
      {children}
    </button>
  );
}
