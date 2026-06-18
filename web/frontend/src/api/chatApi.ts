import type { ChatResponsePayload, InvocationTrace } from "../types/chat";

export interface ChatStreamCallbacks {
  onTrace?: (trace: InvocationTrace) => void;
  onSummary?: (chunk: string) => void;
  onDone?: (response: ChatResponsePayload) => void;
  onError?: (message: string) => void;
}

export interface ChatRequestBody {
  message: string;
  task_id?: string | null;
  context_id?: string | null;
  base_url?: string;
}

export async function checkConnectivity(baseUrl: string): Promise<boolean> {
  const params = new URLSearchParams({ base_url: baseUrl });
  const response = await fetch(`/api/connectivity?${params.toString()}`);
  if (!response.ok) {
    return false;
  }
  const data = (await response.json()) as { online: boolean };
  return data.online;
}

/** 解析 SSE 文本块。 */
function parseSseBlock(block: string): { event: string; data: string } | null {
  const lines = block.split("\n");
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }
  return { event, data: dataLines.join("\n") };
}

/** 通过 POST + ReadableStream 消费 SSE 聊天流。 */
export async function streamChat(
  body: ChatRequestBody,
  callbacks: ChatStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    callbacks.onError?.(text || `HTTP ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.("响应体不可读");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const parsed = parseSseBlock(block.trim());
      if (!parsed) {
        continue;
      }

      try {
        const payload = JSON.parse(parsed.data) as Record<string, unknown>;
        if (parsed.event === "trace") {
          callbacks.onTrace?.(payload as InvocationTrace);
        } else if (parsed.event === "summary") {
          callbacks.onSummary?.(String(payload.chunk ?? ""));
        } else if (parsed.event === "done") {
          callbacks.onDone?.(payload as unknown as ChatResponsePayload);
        } else if (parsed.event === "error") {
          callbacks.onError?.(String(payload.message ?? "未知错误"));
        }
      } catch {
        callbacks.onError?.("SSE 数据解析失败");
      }
    }
  }
}
