"use client";

import { useCallback, useEffect, useState } from "react";
import { Header } from "@/components/header";
import { Sidebar } from "@/components/sidebar";
import { ChatShell } from "@/components/chat-shell";
import {
  ChatSummary,
  deleteChat as deleteChatFromStorage,
  deriveTitle,
  listChats,
  loadChatMessages,
  saveChat,
} from "@/lib/chat-storage";
import type { ChatMessage } from "@/lib/types";

const SESSION_KEY = "rag.sessionId";
const COUNTER_KEY = "rag.chatCounter";
const USE_CACHE_KEY = "rag.useCache";

function freshUuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

interface Session {
  id: string;
  number: number;
  createdAt: number;
}

function readCounter(): number {
  if (typeof window === "undefined") return 1;
  const raw = window.localStorage.getItem(COUNTER_KEY);
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

export default function Home() {
  const [view, setView] = useState<"chat" | "pipeline">("chat");
  const [session, setSession] = useState<Session>({
    id: "",
    number: 1,
    createdAt: 0,
  });
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [initialMessages, setInitialMessages] = useState<ChatMessage[]>([]);
  const [useCache, setUseCache] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const indexed = listChats();
    setChats(indexed);

    const storedId = window.localStorage.getItem(SESSION_KEY);
    const counter = readCounter();
    const found = storedId ? indexed.find((c) => c.id === storedId) : null;

    if (found) {
      setSession({
        id: found.id,
        number: found.number,
        createdAt: found.createdAt,
      });
      setInitialMessages(loadChatMessages(found.id));
    } else {
      const id = storedId || freshUuid();
      setSession({ id, number: counter, createdAt: Date.now() });
      setInitialMessages([]);
      window.localStorage.setItem(SESSION_KEY, id);
      if (!window.localStorage.getItem(COUNTER_KEY)) {
        window.localStorage.setItem(COUNTER_KEY, String(counter));
      }
    }

    const cacheStored = window.localStorage.getItem(USE_CACHE_KEY);
    if (cacheStored !== null) setUseCache(cacheStored === "1");
  }, []);

  const handleToggleCache = useCallback((next: boolean) => {
    setUseCache(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(USE_CACHE_KEY, next ? "1" : "0");
    }
  }, []);

  const handleNewChat = useCallback(() => {
    const counter = readCounter();
    const next: Session = {
      id: freshUuid(),
      number: counter + 1,
      createdAt: Date.now(),
    };
    if (typeof window !== "undefined") {
      window.localStorage.setItem(SESSION_KEY, next.id);
      window.localStorage.setItem(COUNTER_KEY, String(next.number));
    }
    setSession(next);
    setInitialMessages([]);
  }, []);

  const handlePickChat = useCallback((id: string) => {
    const indexed = listChats();
    const summary = indexed.find((c) => c.id === id);
    if (!summary) return;
    const msgs = loadChatMessages(id);
    setSession({
      id: summary.id,
      number: summary.number,
      createdAt: summary.createdAt,
    });
    setInitialMessages(msgs);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(SESSION_KEY, summary.id);
      window.localStorage.setItem(COUNTER_KEY, String(summary.number));
    }
  }, []);

  const handleDeleteChat = useCallback(
    (id: string) => {
      deleteChatFromStorage(id);
      const remaining = listChats();
      setChats(remaining);
      if (id === session.id) {
        handleNewChat();
      }
    },
    [session.id, handleNewChat],
  );

  const handlePersist = useCallback(
    (messages: ChatMessage[]) => {
      if (!session.id || messages.length === 0) return;
      const title = deriveTitle(messages);
      saveChat(
        {
          id: session.id,
          title,
          number: session.number,
          createdAt: session.createdAt || Date.now(),
          updatedAt: Date.now(),
        },
        messages,
      );
      setChats(listChats());
    },
    [session.id, session.number, session.createdAt],
  );

  return (
    <div className="flex flex-col h-full min-h-screen">
      <Header
        activeView={view}
        onChangeView={setView}
        useCache={useCache}
        onToggleCache={handleToggleCache}
        onOpenSidebar={() => setSidebarOpen(true)}
      />
      <div className="flex flex-1 min-h-0">
        <Sidebar
          chats={chats}
          activeId={session.id}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onPickChat={handlePickChat}
          onNewChat={handleNewChat}
          onDeleteChat={handleDeleteChat}
        />
        <main className="flex-1 min-w-0 flex flex-col">
          {session.id ? (
            <ChatShell
              key={session.id}
              sessionId={session.id}
              useCache={useCache}
              view={view}
              initialMessages={initialMessages}
              onPersist={handlePersist}
            />
          ) : (
            <div className="flex flex-1 items-center justify-center p-8 text-muted text-sm">
              Lade Sitzung...
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
