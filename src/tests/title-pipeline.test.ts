import { describe, it, expect, vi, beforeEach } from "vitest";
import { ClaudeAcpAgent } from "../acp-agent.js";

type AgentInternals = {
  extractRealUserText: (params: { prompt: Array<{ type: string; text?: string }> }) => string | null;
  maybeGenerateSessionTitleForTurn: (
    session: SessionLike,
    params: { sessionId: string; prompt: Array<{ type: string; text?: string }> },
  ) => Promise<void>;
};

type SessionLike = {
  query: { generateSessionTitle: ReturnType<typeof vi.fn> };
  realUserMessageCount: number;
  recentUserPromptTexts: string[];
  titleGenerationSequence: number;
  lastEmittedTitle: string | undefined;
};

function makeSession(query: SessionLike["query"]): SessionLike {
  return {
    query,
    realUserMessageCount: 0,
    recentUserPromptTexts: [],
    titleGenerationSequence: 0,
    lastEmittedTitle: undefined,
  };
}

function makeFakeQuery(...titles: Array<string | undefined>): SessionLike["query"] {
  const fn = vi.fn();
  for (const t of titles) fn.mockResolvedValueOnce(t);
  if (titles.length === 0) fn.mockResolvedValue("fake title");
  return { generateSessionTitle: fn };
}

const flush = () => new Promise((r) => setImmediate(r));

describe("title pipeline — extractRealUserText", () => {
  let agent: AgentInternals;
  let sessionUpdate: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    sessionUpdate = vi.fn().mockResolvedValue(undefined);
    const mockClient = {
      sessionUpdate,
    } as never;
    agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} }) as unknown as AgentInternals;
  });

  it("returns the trimmed first text block for a normal prompt", () => {
    expect(agent.extractRealUserText({ prompt: [{ type: "text", text: "  Compare TCP.  " }] }))
      .toBe("Compare TCP.");
  });

  it("returns null when no text block is present (tool-only)", () => {
    expect(agent.extractRealUserText({ prompt: [{ type: "image" } as never] })).toBeNull();
  });

  it("returns null when the text is empty / whitespace only", () => {
    expect(agent.extractRealUserText({ prompt: [{ type: "text", text: "" }] })).toBeNull();
    expect(agent.extractRealUserText({ prompt: [{ type: "text", text: "   \n " }] })).toBeNull();
  });

  it("returns null for known local-only slash commands", () => {
    // The set is named LOCAL_ONLY_COMMANDS in acp-agent.ts. Update this list
    // along with the source if the set ever changes.
    expect(agent.extractRealUserText({ prompt: [{ type: "text", text: "/context" }] })).toBeNull();
    expect(agent.extractRealUserText({ prompt: [{ type: "text", text: "/heapdump now" }] })).toBeNull();
  });

  it("does NOT skip slash-prefixed text that isn't a local-only command", () => {
    expect(agent.extractRealUserText({ prompt: [{ type: "text", text: "/explain how X works" }] }))
      .toBe("/explain how X works");
  });
});

describe("title pipeline — maybeGenerateSessionTitleForTurn", () => {
  let agent: AgentInternals;
  let sessionUpdate: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    sessionUpdate = vi.fn().mockResolvedValue(undefined);
    const mockClient = {
      sessionUpdate,
    } as never;
    agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} }) as unknown as AgentInternals;
  });

  it("fires generateSessionTitle on the first real user message and emits update", async () => {
    const query = makeFakeQuery("Compare TCP and UDP networking protocols");
    const session = makeSession(query);

    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s1",
      prompt: [{ type: "text", text: "Compare TCP and UDP." }],
    });

    expect(query.generateSessionTitle).toHaveBeenCalledTimes(1);
    expect(query.generateSessionTitle).toHaveBeenCalledWith("Compare TCP and UDP.", { persist: true });
    expect(sessionUpdate).toHaveBeenCalledWith({
      sessionId: "s1",
      update: { sessionUpdate: "session_info_update", title: "Compare TCP and UDP networking protocols" },
    });
    expect(session.realUserMessageCount).toBe(1);
    expect(session.recentUserPromptTexts).toEqual(["Compare TCP and UDP."]);
    expect(session.lastEmittedTitle).toBe("Compare TCP and UDP networking protocols");
  });

  it("skips generation but still leaves the session in a sane state for local-only commands", async () => {
    const query = makeFakeQuery("never");
    const session = makeSession(query);

    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s2",
      prompt: [{ type: "text", text: "/context" }],
    });

    expect(query.generateSessionTitle).not.toHaveBeenCalled();
    expect(session.realUserMessageCount).toBe(0);
    expect(sessionUpdate).not.toHaveBeenCalled();
  });

  it("fires again on the third real user message using accumulated context", async () => {
    const query = makeFakeQuery("First-pass title", "Refined third-pass title");
    const session = makeSession(query);

    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s3", prompt: [{ type: "text", text: "Hello." }],
    });
    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s3", prompt: [{ type: "text", text: "What about Rust?" }],
    });
    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s3", prompt: [{ type: "text", text: "Specifically, ownership semantics." }],
    });

    expect(query.generateSessionTitle).toHaveBeenCalledTimes(2);
    expect(query.generateSessionTitle.mock.calls[1][0]).toBe(
      "Hello.\n\nWhat about Rust?\n\nSpecifically, ownership semantics."
    );
    expect(session.realUserMessageCount).toBe(3);
  });

  it("does NOT fire on counts other than 1 and 3", async () => {
    const query = makeFakeQuery("title-1", "title-3");
    const session = makeSession(query);

    for (let i = 0; i < 6; i++) {
      await agent.maybeGenerateSessionTitleForTurn(session, {
        sessionId: "s4", prompt: [{ type: "text", text: `msg ${i + 1}` }],
      });
    }

    expect(query.generateSessionTitle).toHaveBeenCalledTimes(2);
    expect(session.realUserMessageCount).toBe(6);
    // Cap on accumulated text — should still be the first three.
    expect(session.recentUserPromptTexts).toEqual(["msg 1", "msg 2", "msg 3"]);
  });

  it("drops a stale first-pass title when a newer third-pass already won", async () => {
    // Construct two pending generations: the first deferred, the second
    // resolves immediately.
    let resolveFirst!: (v: string) => void;
    const firstPromise = new Promise<string>((r) => { resolveFirst = r; });
    const query: SessionLike["query"] = {
      generateSessionTitle: vi.fn()
        .mockReturnValueOnce(firstPromise)
        .mockResolvedValueOnce("Third-pass winner"),
    };
    const session = makeSession(query);

    // Kick off message 1 — its generation will hang on `firstPromise`.
    const firstCall = agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s5", prompt: [{ type: "text", text: "msg 1" }],
    });
    // Don't await yet — its `await generateSessionTitle` is parked.

    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s5", prompt: [{ type: "text", text: "msg 2" }],
    });
    await agent.maybeGenerateSessionTitleForTurn(session, {
      sessionId: "s5", prompt: [{ type: "text", text: "msg 3" }],
    });

    // Third-pass already emitted.
    expect(sessionUpdate).toHaveBeenCalledWith({
      sessionId: "s5",
      update: { sessionUpdate: "session_info_update", title: "Third-pass winner" },
    });
    expect(sessionUpdate).toHaveBeenCalledTimes(1);

    // Now release the slow first-pass — it should drop on the sequence check.
    resolveFirst("Slow first-pass loser");
    await firstCall;
    await flush();

    expect(sessionUpdate).toHaveBeenCalledTimes(1);
    expect(session.lastEmittedTitle).toBe("Third-pass winner");
  });
});
