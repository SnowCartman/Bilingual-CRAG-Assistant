"use client";

import { cn } from "@/lib/utils";

interface MatryoshkaProps {
  className?: string;
  size?: number; // controls height in px; width follows the 2:3 silhouette
  ariaLabel?: string;
}

// 1:1 port of archive/web_app_gradio_v1/app.py MATRYOSHKA_HTML.
// Colors are intentionally hard-coded (not theme variables) so the doll
// keeps the same warm look in light AND dark mode -- using --surface for
// the face turned it grey in dark mode, which felt gloomy. The fixed
// cream face + peach cheeks + claude body keep the doll friendly.
export function Matryoshka({
  className,
  size = 24,
  ariaLabel = "Matrjoschka",
}: MatryoshkaProps) {
  const width = Math.round(size * (2 / 3));
  return (
    <svg
      width={width}
      height={size}
      viewBox="0 0 100 150"
      role="img"
      aria-label={ariaLabel}
      className={cn(className)}
    >
      <ellipse cx="50" cy="105" rx="38" ry="42" fill="#D97757" />
      <ellipse cx="50" cy="42" rx="28" ry="32" fill="#D97757" />
      <path
        d="M 22,38 Q 50,6 78,38 Q 50,30 22,38 Z"
        fill="#A04A2C"
      />
      <ellipse cx="50" cy="50" rx="18" ry="22" fill="#F5F4ED" />
      <circle cx="36" cy="56" r="3.2" fill="#FFB89A" opacity="0.75" />
      <circle cx="64" cy="56" r="3.2" fill="#FFB89A" opacity="0.75" />
      <circle cx="43" cy="49" r="2" fill="#1A1A1A" />
      <circle cx="57" cy="49" r="2" fill="#1A1A1A" />
      <path
        d="M 45,61 Q 50,64 55,61"
        stroke="#1A1A1A"
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
      />
      <ellipse cx="50" cy="115" rx="22" ry="22" fill="#F5F4ED" />
      <circle cx="44" cy="111" r="3" fill="#FFB89A" />
      <circle cx="56" cy="111" r="3" fill="#FFB89A" />
      <circle cx="44" cy="119" r="3" fill="#FFB89A" />
      <circle cx="56" cy="119" r="3" fill="#FFB89A" />
      <circle cx="50" cy="115" r="4" fill="#D97757" />
      <circle cx="50" cy="115" r="1.8" fill="#A04A2C" />
    </svg>
  );
}
