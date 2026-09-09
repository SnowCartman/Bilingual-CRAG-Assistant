"use client";

import type { ChatMessage } from "./types";

const INDEX_KEY = "rag.chats.index";
const MSG_PREFIX = "rag.chats.msg.";
const MAX_CHATS = 30;

export interface ChatSummary {
  id: string;
  title: string;
  number: number;
  createdAt: number;
  updatedAt: number;
}

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function listChats(): ChatSummary[] {
  if (typeof window === "undefined") return [];
  const idx = safeParse<ChatSummary[]>(
    window.localStorage.getItem(INDEX_KEY),
    [],
  );
  return [...idx].sort((a, b) => b.updatedAt - a.updatedAt);
}

export function loadChatMessages(id: string): ChatMessage[] {
  if (typeof window === "undefined") return [];
  return safeParse<ChatMessage[]>(
    window.localStorage.getItem(MSG_PREFIX + id),
    [],
  );
}

export function saveChat(summary: ChatSummary, messages: ChatMessage[]): void {
  if (typeof window === "undefined") return;
  // Drop transient streaming flag; persisted messages are always "done".
  const persisted: ChatMessage[] = messages.map((m) => ({
    ...m,
    isStreaming: false,
  }));
  try {
    window.localStorage.setItem(
      MSG_PREFIX + summary.id,
      JSON.stringify(persisted),
    );
  } catch (err) {
    // Quota exceeded -- evict oldest then retry once.
    evictOldest();
    try {
      window.localStorage.setItem(
        MSG_PREFIX + summary.id,
        JSON.stringify(persisted),
      );
    } catch {
      return;
    }
  }
  const idx = safeParse<ChatSummary[]>(
    window.localStorage.getItem(INDEX_KEY),
    [],
  );
  const filtered = idx.filter((c) => c.id !== summary.id);
  filtered.unshift(summary);
  while (filtered.length > MAX_CHATS) {
    const dropped = filtered.pop();
    if (dropped) window.localStorage.removeItem(MSG_PREFIX + dropped.id);
  }
  window.localStorage.setItem(INDEX_KEY, JSON.stringify(filtered));
}

function evictOldest(): void {
  if (typeof window === "undefined") return;
  const idx = safeParse<ChatSummary[]>(
    window.localStorage.getItem(INDEX_KEY),
    [],
  );
  if (idx.length === 0) return;
  const sorted = [...idx].sort((a, b) => a.updatedAt - b.updatedAt);
  const dropped = sorted[0];
  if (!dropped) return;
  window.localStorage.removeItem(MSG_PREFIX + dropped.id);
  const remaining = idx.filter((c) => c.id !== dropped.id);
  window.localStorage.setItem(INDEX_KEY, JSON.stringify(remaining));
}

export function deleteChat(id: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(MSG_PREFIX + id);
  const idx = safeParse<ChatSummary[]>(
    window.localStorage.getItem(INDEX_KEY),
    [],
  );
  const filtered = idx.filter((c) => c.id !== id);
  window.localStorage.setItem(INDEX_KEY, JSON.stringify(filtered));
}

export function deriveTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser) return "Neuer Chat";
  const t = firstUser.content.trim().replace(/\s+/g, " ");
  if (!t) return "Neuer Chat";
  return t.length > 60 ? t.slice(0, 57) + "..." : t;
}
