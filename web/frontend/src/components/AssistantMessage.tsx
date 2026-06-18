import ReactMarkdown from "react-markdown";

import type { ChatMessage } from "../types/chat";
import { ConfirmationButtons } from "./ConfirmationButtons";
import { InvocationTraces } from "./InvocationTraces";
import { PlanConfirmPanel } from "./PlanConfirmPanel";

interface AssistantMessageProps {
  message: ChatMessage;
  displayContent: string;
  showConfirmation: boolean;
  onConfirmationSelect: (replyText: string) => void;
  disabled?: boolean;
}

export function AssistantMessage({
  message,
  displayContent,
  showConfirmation,
  onConfirmationSelect,
  disabled = false,
}: AssistantMessageProps) {
  return (
    <div className={`message-row assistant`}>
      <div className="message-avatar">AI</div>
      <div className="message-body">
        {message.invocation_traces && message.invocation_traces.length > 0 ? (
          <InvocationTraces traces={message.invocation_traces} />
        ) : null}
        <div
          className={`message-content${message.is_error ? " error" : ""}`}
        >
          <ReactMarkdown>{displayContent}</ReactMarkdown>
        </div>
        {showConfirmation && message.confirmation ? (
          message.confirmation.confirm_type === "plan_confirm" ? (
            <PlanConfirmPanel
              confirmation={message.confirmation}
              onSelect={onConfirmationSelect}
              disabled={disabled}
            />
          ) : (
            <ConfirmationButtons
              confirmation={message.confirmation}
              onSelect={onConfirmationSelect}
              disabled={disabled}
            />
          )
        ) : null}
      </div>
    </div>
  );
}
