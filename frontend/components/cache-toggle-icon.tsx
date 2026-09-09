"use client";

import { Zap, ZapOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface CacheToggleIconProps {
  value: boolean;
  onChange: (next: boolean) => void;
}

// Compact header-bar version of the cache toggle. Visual hierarchy matches
// ThemeToggle (ghost icon button) so the right-hand side of the header
// stays calm. Tooltip + aria-label carry the status text so we do not
// duplicate the verbose sidebar-style copy here.
export function CacheToggleIcon({ value, onChange }: CacheToggleIconProps) {
  const label = value
    ? "Semantischer Cache aktiv -- Treffer werden genutzt"
    : "Semantischer Cache umgangen -- volle Pipeline";
  return (
    <Button
      variant="ghost"
      size="icon"
      role="switch"
      aria-checked={value}
      aria-label={label}
      title={label}
      onClick={() => onChange(!value)}
      className={cn(value ? "text-accent" : "text-muted")}
    >
      {value ? (
        <Zap className="h-4 w-4" fill="currentColor" />
      ) : (
        <ZapOff className="h-4 w-4" />
      )}
    </Button>
  );
}
