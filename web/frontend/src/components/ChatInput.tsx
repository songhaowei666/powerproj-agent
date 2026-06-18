import { FormEvent, useState } from "react";

interface ChatInputProps {
  onSubmit: (text: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSubmit, disabled = false }: ChatInputProps) {
  const [value, setValue] = useState("");

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSubmit(trimmed);
    setValue("");
  };

  return (
    <div className="chat-input-bar">
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={value}
          disabled={disabled}
          placeholder="请输入您的问题，例如：帮我统计今年的投资收益并做明年规划"
          onChange={(event) => setValue(event.target.value)}
        />
        <button type="submit" disabled={disabled || !value.trim()}>
          发送
        </button>
      </form>
    </div>
  );
}
