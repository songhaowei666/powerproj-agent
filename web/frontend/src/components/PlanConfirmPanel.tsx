import { useMemo, useState } from "react";

import type { ConfirmationUI } from "../types/chat";
import { formatPlanApproveReply } from "../utils/planReply";

interface PlanConfirmPanelProps {
  confirmation: ConfirmationUI;
  onSelect: (replyText: string) => void;
  disabled?: boolean;
}

interface PlanTask {
  id: string;
  name: string;
  required_agent: string;
}

export function PlanConfirmPanel({
  confirmation,
  onSelect,
  disabled = false,
}: PlanConfirmPanelProps) {
  const body = confirmation.body ?? {};
  const tasks = (body.tasks as PlanTask[] | undefined) ?? [];
  const revision = Number(body.revision ?? 1);

  const taskItems = useMemo(
    () => tasks.filter((task) => typeof task.id === "string" && task.id),
    [tasks],
  );

  const [selectedIds, setSelectedIds] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    for (const task of taskItems) {
      initial[task.id] = true;
    }
    return initial;
  });

  const allTaskIds = taskItems.map((task) => task.id);
  const checkedIds = allTaskIds.filter((id) => selectedIds[id]);

  const approveOption = confirmation.options.find(
    (option) => option.id === "approve",
  );
  const otherOptions = confirmation.options.filter(
    (option) => option.id !== "approve",
  );

  const toggleTask = (taskId: string) => {
    setSelectedIds((prev) => ({
      ...prev,
      [taskId]: !prev[taskId],
    }));
  };

  return (
    <div className="confirmation-panel">
      {confirmation.title ? (
        <p className="confirmation-title">{confirmation.title}</p>
      ) : null}
      <div className="plan-task-list">
        {taskItems.map((task, index) => {
          const label = task.required_agent
            ? `[${task.required_agent}] ${task.name}`
            : task.name;
          return (
            <label
              key={`${task.id}-${revision}-${index}`}
              className="plan-task-item"
            >
              <input
                type="checkbox"
                checked={selectedIds[task.id] ?? false}
                disabled={disabled}
                onChange={() => toggleTask(task.id)}
              />
              <span>{label}</span>
            </label>
          );
        })}
      </div>
      <div className="confirmation-buttons">
        {approveOption ? (
          <button
            type="button"
            disabled={disabled || checkedIds.length === 0}
            onClick={() =>
              onSelect(formatPlanApproveReply(checkedIds, allTaskIds))
            }
          >
            {approveOption.label}
          </button>
        ) : null}
        {otherOptions.map((option) => (
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
