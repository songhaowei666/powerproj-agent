export interface ConfirmationOption {
  id: string;
  label: string;
  replyText: string;
}

export interface ConfirmationUI {
  action: string;
  title: string;
  options: ConfirmationOption[];
  confirm_type: "confirmation" | "plan_confirm";
  body?: Record<string, unknown> | null;
}

export interface InvocationTrace {
  step?: number;
  agent_type?: string;
  agent_name?: string;
  capability?: string;
  task_id?: string;
  phase?: number;
  status?: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  invocation_traces?: InvocationTrace[];
  confirmation?: ConfirmationUI | null;
  task_id?: string;
  is_error?: boolean;
}

export interface ChatResponsePayload {
  task_id: string;
  context_id: string;
  state: string;
  text: string;
  artifacts: Record<string, unknown>[];
  invocation_traces: InvocationTrace[];
  is_error: boolean;
  confirmation?: ConfirmationUI | null;
}

export interface StreamingState {
  traces: InvocationTrace[];
  summary: string;
  isStreaming: boolean;
}
