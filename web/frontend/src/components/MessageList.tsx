import type { ChatMessage, StreamingState } from "../types/chat";
import { AssistantMessage } from "./AssistantMessage";
import { InvocationTraces } from "./InvocationTraces";

interface MessageListProps {
  messages: ChatMessage[];
  streaming: StreamingState;
  getDisplayContent: (message: ChatMessage) => string;
  shouldShowConfirmation: (message: ChatMessage) => boolean;
  onConfirmationSelect: (replyText: string) => void;
  isSending: boolean;
}

export function MessageList({
  messages,
  streaming,
  getDisplayContent,
  shouldShowConfirmation,
  onConfirmationSelect,
  isSending,
}: MessageListProps) {
  return (
    <div className="message-list">
      {messages.map((message, index) => {
        if (message.role === "user") {
          return (
            <div key={`user-${index}`} className="message-row user">
              <div className="message-avatar">我</div>
              <div className="message-body">
                <div className="message-content">{message.content}</div>
              </div>
            </div>
          );
        }

        return (
          <AssistantMessage
            key={`assistant-${index}`}
            message={message}
            displayContent={getDisplayContent(message)}
            showConfirmation={shouldShowConfirmation(message)}
            onConfirmationSelect={onConfirmationSelect}
            disabled={isSending}
          />
        );
      })}

      {streaming.isStreaming ? (
        <div className="message-row assistant">
          <div className="message-avatar">AI</div>
          <div className="message-body">
            {streaming.traces.length > 0 ? (
              <InvocationTraces traces={streaming.traces} />
            ) : null}
            <div className="message-content">
              {streaming.summary ? (
                <p>{streaming.summary}</p>
              ) : (
                <p className="streaming-placeholder">正在处理，请稍候...</p>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
