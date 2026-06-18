import { useCallback, useEffect, useRef, useState } from "react";

import { checkConnectivity, streamChat } from "../api/chatApi";
import type {
  ChatMessage,
  ChatResponsePayload,
  ConfirmationUI,
  InvocationTrace,
  StreamingState,
} from "../types/chat";
import { formatAssistantContent } from "../utils/planReply";

const DEFAULT_BASE_URL = "http://localhost:8000";

function extractNewTraces(
  traces: InvocationTrace[],
  seenSteps: Set<number>,
): { newTraces: InvocationTrace[]; seenSteps: Set<number> } {
  const nextSeen = new Set(seenSteps);
  const newTraces: InvocationTrace[] = [];

  const sorted = [...traces].sort(
    (left, right) => (left.step ?? 0) - (right.step ?? 0),
  );

  for (const trace of sorted) {
    const step = trace.step;
    if (typeof step === "number" && step > 0) {
      if (nextSeen.has(step)) {
        continue;
      }
      nextSeen.add(step);
    }
    newTraces.push(trace);
  }

  return { newTraces, seenSteps: nextSeen };
}

function isAwaitingConfirmation(
  awaitingInput: boolean,
  taskId: string | null,
  pendingConfirmation: ConfirmationUI | null,
  messages: ChatMessage[],
): boolean {
  if (pendingConfirmation) {
    return true;
  }
  if (!awaitingInput || !taskId) {
    return false;
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant") {
      return Boolean(
        message.confirmation && message.task_id === taskId,
      );
    }
    if (message.role === "user") {
      return false;
    }
  }
  return false;
}

export function useChatSession() {
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [contextId, setContextId] = useState<string | null>(null);
  const [awaitingInput, setAwaitingInput] = useState(false);
  const [pendingConfirmation, setPendingConfirmation] =
    useState<ConfirmationUI | null>(null);
  const [seenTraceSteps, setSeenTraceSteps] = useState<Set<number>>(new Set());
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [streaming, setStreaming] = useState<StreamingState>({
    traces: [],
    summary: "",
    isStreaming: false,
  });

  const abortRef = useRef<AbortController | null>(null);

  const refreshConnectivity = useCallback(async () => {
    const online = await checkConnectivity(baseUrl);
    setIsOnline(online);
    return online;
  }, [baseUrl]);

  useEffect(() => {
    void refreshConnectivity();
  }, [refreshConnectivity]);

  const resetConversation = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setTaskId(null);
    setContextId(null);
    setAwaitingInput(false);
    setPendingConfirmation(null);
    setSeenTraceSteps(new Set());
    setStreaming({ traces: [], summary: "", isStreaming: false });
    setIsSending(false);
  }, []);

  const resolveSendTaskId = useCallback(
    (forNewPrompt: boolean): string | null => {
      if (!forNewPrompt) {
        return taskId;
      }
      const awaitingConfirm = isAwaitingConfirmation(
        awaitingInput,
        taskId,
        pendingConfirmation,
        messages,
      );
      if (awaitingConfirm) {
        setTaskId(null);
        setContextId(null);
        setAwaitingInput(false);
        setPendingConfirmation(null);
        return null;
      }
      if (awaitingInput) {
        return taskId;
      }
      return null;
    },
    [awaitingInput, messages, pendingConfirmation, taskId],
  );

  const resolveSendContextId = useCallback(
    (sendTaskId: string | null): string | null => {
      if (awaitingInput && sendTaskId) {
        return contextId;
      }
      return null;
    },
    [awaitingInput, contextId],
  );

  const applyChatResponse = useCallback(
    (
      chatResp: ChatResponsePayload,
      turnTraces: InvocationTrace[],
    ): ChatMessage => {
      const isInputRequired = chatResp.state === "input-required";
      setAwaitingInput(isInputRequired);
      if (isInputRequired) {
        setTaskId(chatResp.task_id || null);
        setContextId(chatResp.context_id || null);
        setPendingConfirmation(chatResp.confirmation ?? null);
      } else {
        setTaskId(null);
        setContextId(null);
        setPendingConfirmation(null);
      }

      return {
        role: "assistant",
        content: chatResp.text,
        invocation_traces: turnTraces.length
          ? turnTraces
          : chatResp.invocation_traces,
        confirmation: chatResp.confirmation,
        task_id: chatResp.task_id,
        is_error: chatResp.is_error,
      };
    },
    [],
  );

  const sendMessage = useCallback(
    async (userPrompt: string, options?: { forceTaskId?: string | null }) => {
      if (isSending) {
        return;
      }

      const online = await refreshConnectivity();
      const userMessage: ChatMessage = {
        role: "user",
        content: userPrompt,
      };
      setMessages((prev) => [...prev, userMessage]);

      if (!online) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "主控 Agent 未连接，请先启动服务后再试。",
            is_error: true,
          },
        ]);
        return;
      }

      const isNewTask =
        options?.forceTaskId === undefined
          ? resolveSendTaskId(true) === null && !awaitingInput
          : options.forceTaskId === null;
      const sendTaskId =
        options?.forceTaskId !== undefined
          ? options.forceTaskId
          : resolveSendTaskId(true);
      const sendContextId = resolveSendContextId(sendTaskId);

      if (isNewTask || sendTaskId === null) {
        setSeenTraceSteps(new Set());
      }

      const turnTraces: InvocationTrace[] = [];
      let currentSeen = isNewTask || sendTaskId === null ? new Set<number>() : seenTraceSteps;

      setIsSending(true);
      setStreaming({ traces: [], summary: "", isStreaming: true });
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      await streamChat(
        {
          message: userPrompt,
          task_id: sendTaskId,
          context_id: sendContextId,
          base_url: baseUrl,
        },
        {
          onTrace: (trace) => {
            const extracted = extractNewTraces([trace], currentSeen);
            currentSeen = extracted.seenSteps;
            if (extracted.newTraces.length > 0) {
              turnTraces.push(...extracted.newTraces);
              setSeenTraceSteps(extracted.seenSteps);
              setStreaming((prev) => ({
                ...prev,
                traces: [...prev.traces, ...extracted.newTraces],
              }));
            }
          },
          onSummary: (chunk) => {
            setStreaming((prev) => ({
              ...prev,
              summary: prev.summary + chunk,
            }));
          },
          onDone: (chatResp) => {
            let finalTraces = turnTraces;
            if (finalTraces.length === 0 && chatResp.invocation_traces.length > 0) {
              const extracted = extractNewTraces(
                chatResp.invocation_traces,
                currentSeen,
              );
              finalTraces = extracted.newTraces;
              setSeenTraceSteps(extracted.seenSteps);
            }
            const assistantMessage = applyChatResponse(chatResp, finalTraces);
            setMessages((prev) => [...prev, assistantMessage]);
            setStreaming({ traces: [], summary: "", isStreaming: false });
            setIsSending(false);
          },
          onError: (message) => {
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: `请求失败：${message}`,
                is_error: true,
              },
            ]);
            setStreaming({ traces: [], summary: "", isStreaming: false });
            setIsSending(false);
          },
        },
        abortRef.current.signal,
      );
    },
    [
      applyChatResponse,
      awaitingInput,
      baseUrl,
      isSending,
      refreshConnectivity,
      resolveSendContextId,
      resolveSendTaskId,
      seenTraceSteps,
    ],
  );

  const sendConfirmationReply = useCallback(
    async (replyText: string) => {
      setPendingConfirmation(null);
      await sendMessage(replyText, { forceTaskId: taskId });
    },
    [sendMessage, taskId],
  );

  const handleUserInput = useCallback(
    async (prompt: string) => {
      await sendMessage(prompt);
    },
    [sendMessage],
  );

  const getDisplayContent = useCallback(
    (message: ChatMessage) =>
      formatAssistantContent(message.content, message.confirmation),
    [],
  );

  const shouldShowConfirmation = useCallback(
    (message: ChatMessage) =>
      message.role === "assistant" &&
      awaitingInput &&
      Boolean(message.confirmation) &&
      message.task_id === taskId,
    [awaitingInput, taskId],
  );

  return {
    baseUrl,
    setBaseUrl,
    messages,
    taskId,
    isOnline,
    isSending,
    streaming,
    resetConversation,
    refreshConnectivity,
    handleUserInput,
    sendConfirmationReply,
    getDisplayContent,
    shouldShowConfirmation,
  };
}
