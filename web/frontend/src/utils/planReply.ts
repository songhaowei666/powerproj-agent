import type { ConfirmationUI } from "../types/chat";

/** 构建带勾选结果的确认执行回复文本。 */
export function formatPlanApproveReply(
  selectedTaskIds: string[],
  allTaskIds: string[],
): string {
  const selectedSet = new Set(selectedTaskIds);
  const allSet = new Set(allTaskIds);
  if (
    selectedSet.size === allSet.size &&
    allTaskIds.every((id) => selectedSet.has(id))
  ) {
    return "确认执行";
  }
  return `确认执行:${selectedTaskIds.join(",")}`;
}

/** 计划确认时仅展示引导语。 */
export function formatAssistantContent(
  content: string,
  confirmation?: ConfirmationUI | null,
): string {
  if (confirmation?.confirm_type === "plan_confirm" && content) {
    return content.split("\n\n")[0] ?? content;
  }
  return content;
}
