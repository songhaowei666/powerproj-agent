import type { InvocationTrace } from "../types/chat";

function formatTraceStatusLabel(status: string): string {
  const mapping: Record<string, string> = {
    success: "成功",
    failed: "失败",
    input_required: "待确认",
  };
  return mapping[status] ?? (status || "未知");
}

function buildTraceTitle(
  trace: InvocationTrace,
  turnIndex: number,
  intentIndex: number,
): string {
  const agentType = trace.agent_type ?? "";
  const agentName = trace.agent_name ?? "未知 Agent";
  const globalStep = trace.step ?? turnIndex;
  const statusLabel = formatTraceStatusLabel(trace.status ?? "success");

  if (agentType === "intent") {
    const roleText =
      intentIndex > 1 ? "意图识别（补充后）" : "意图识别";
    return `本轮 ${turnIndex} | ${roleText} | ${statusLabel}（总步骤 ${globalStep}）`;
  }

  const capability = trace.capability ?? "";
  const taskId = trace.task_id ?? "";
  const phase = trace.phase;
  const phaseText = phase !== undefined ? `Phase ${phase}` : "";
  return (
    `本轮 ${turnIndex} | 业务 Agent: ${agentName}` +
    ` | 能力: ${capability} | 任务: ${taskId}` +
    ` | ${phaseText} | ${statusLabel}（总步骤 ${globalStep}）`
  );
}

interface InvocationTracesProps {
  traces: InvocationTrace[];
}

export function InvocationTraces({ traces }: InvocationTracesProps) {
  if (traces.length === 0) {
    return null;
  }

  const sortedTraces = [...traces].sort(
    (left, right) => (left.step ?? 0) - (right.step ?? 0),
  );
  let intentSeen = 0;

  return (
    <div className="invocation-traces">
      <h4>本轮调用轨迹</h4>
      {sortedTraces.map((trace, index) => {
        if (trace.agent_type === "intent") {
          intentSeen += 1;
        }
        const title = buildTraceTitle(trace, index + 1, intentSeen);
        return (
          <details key={`${trace.step ?? index}-${title}`} className="trace-item">
            <summary className="trace-summary">{title}</summary>
            <div className="trace-body">
              <div className="trace-column">
                <h5>调用参数</h5>
                <pre>{JSON.stringify(trace.input ?? {}, null, 2)}</pre>
              </div>
              <div className="trace-column">
                <h5>返回结果</h5>
                <pre>{JSON.stringify(trace.output ?? {}, null, 2)}</pre>
              </div>
            </div>
          </details>
        );
      })}
    </div>
  );
}
