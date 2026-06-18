import type { ConfirmationUI } from "../types/chat";

interface ConfirmationButtonsProps {
  confirmation: ConfirmationUI;
  onSelect: (replyText: string) => void;
  disabled?: boolean;
}

export function ConfirmationButtons({
  confirmation,
  onSelect,
  disabled = false,
}: ConfirmationButtonsProps) {
  return (
    <div className="confirmation-panel">
      {confirmation.title ? (
        <p className="confirmation-title">{confirmation.title}</p>
      ) : null}
      <div className="confirmation-buttons">
        {confirmation.options.map((option) => (
          <button
            key={option.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(option.replyText)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
