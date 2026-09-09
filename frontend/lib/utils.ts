import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Heuristic mojibake fix: undo the visible artefacts that appear when a UTF-8
 * byte sequence has been read with a Latin-1 / cp1252 codec somewhere upstream
 * (root cause here: the marker-pdf extraction that fed Qdrant). Done at render
 * time only -- no re-indexing.
 *
 * The source file must stay pure ASCII per the ASCII-only convention (the
 * Write tool eats raw \\uXXXX escapes and emits the decoded char). We build
 * the patterns at module init from numeric codepoints instead -- same trick
 * used in app/services/security.py for the Cyrillic injection patterns.
 */
const C = (...codes: number[]) => String.fromCharCode(...codes);

const _bytes = (...codes: number[]) =>
  new RegExp(C(...codes), "g");

const MOJIBAKE_MAP: Array<[RegExp, string]> = [
  // German umlauts: UTF-8 (C3 XX) misread as Latin-1 -> (U+00C3 + U+00XX)
  [_bytes(0x00C3, 0x00A4), "ae"], // a-umlaut
  [_bytes(0x00C3, 0x00B6), "oe"], // o-umlaut
  [_bytes(0x00C3, 0x00BC), "ue"], // u-umlaut
  [_bytes(0x00C3, 0x0084), "Ae"], // A-umlaut
  [_bytes(0x00C3, 0x0096), "Oe"], // O-umlaut
  [_bytes(0x00C3, 0x009C), "Ue"], // U-umlaut
  [_bytes(0x00C3, 0x009F), "ss"], // sharp-s
  // Smart punctuation: UTF-8 (E2 80 XX) misread as Latin-1 -> (U+00E2 + U+0080 + U+00XX)
  [_bytes(0x00E2, 0x0080, 0x0094), "-"], // em-dash
  [_bytes(0x00E2, 0x0080, 0x0093), "-"], // en-dash
  [_bytes(0x00E2, 0x0080, 0x0099), "'"], // right single quote
  [_bytes(0x00E2, 0x0080, 0x009C), '"'], // left double quote
  [_bytes(0x00E2, 0x0080, 0x009D), '"'], // right double quote
  // Non-breaking space: UTF-8 (C2 A0) misread as Latin-1 -> (U+00C2 + U+00A0)
  [_bytes(0x00C2, 0x00A0), " "],
];

/**
 * Some PDF chunks in the Russian dictionary have a font-encoding artefact:
 * Cyrillic codepoints were shifted DOWN by exactly 0x01D6 (=470), landing
 * inside the IPA Extensions block (U+025A..U+0279) and Latin Extended-B
 * (U+023A..U+0259). We reverse that shift at render time. Verified by
 * Smoke test 2026-06-21: a chunk read "blond: [9 IPA-Extensions
 * glyphs]" -- adding +0x01D6 per glyph yields the expected 9-letter
 * Cyrillic word for "blond" (the literal cannot appear here per
 * ASCII-only convention).
 *
 * Shift range covers Cyrillic 0x0410-0x044F (uppercase + lowercase) plus
 * common extras (capital Yo=0x0401, small yo=0x0451) mapped down by
 * 0x01D6 to land in 0x022A-0x027B.
 */
const SHIFT_LO = 0x022a;
const SHIFT_HI = 0x027b;
const SHIFT_BACK = 0x01d6;

function unshiftCyrillic(text: string): string {
  let out = "";
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= SHIFT_LO && code <= SHIFT_HI) {
      out += String.fromCharCode(code + SHIFT_BACK);
    } else {
      out += text[i];
    }
  }
  return out;
}

export function fixMojibake(text: string): string {
  if (!text) return text;
  let out = text;
  for (const [re, sub] of MOJIBAKE_MAP) {
    out = out.replace(re, sub);
  }
  out = unshiftCyrillic(out);
  return out;
}

export function formatScore(n: number | null | undefined, digits = 3): string {
  if (n == null) return "-";
  return n.toFixed(digits);
}
