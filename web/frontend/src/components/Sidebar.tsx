interface SidebarProps {
  baseUrl: string;
  taskId: string | null;
  onBaseUrlChange: (value: string) => void;
  onNewConversation: () => void;
}

export function Sidebar({
  baseUrl,
  taskId,
  onBaseUrlChange,
  onNewConversation,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <h2>设置</h2>
      <div>
        <label htmlFor="base-url">主控 Agent 地址</label>
        <input
          id="base-url"
          type="text"
          value={baseUrl}
          placeholder="http://localhost:8000"
          onChange={(event) => onBaseUrlChange(event.target.value.trim())}
        />
      </div>
      <button type="button" onClick={onNewConversation}>
        新对话
      </button>
      <div className="sidebar-divider" />
      <div className="sidebar-instructions">
        <strong>使用说明</strong>
        <ol>
          <li>先启动主控 Agent（端口 8000）及业务 Agent</li>
          <li>输入自然语言问题，例如统计、规划、投资相关需求</li>
          <li>计划确认时可勾选子任务并点「开始执行」，业务确认可点「是/否」</li>
          <li>调用轨迹与总结会在处理过程中实时更新</li>
        </ol>
      </div>
      {taskId ? (
        <>
          <div className="sidebar-divider" />
          <div className="task-id-block">
            <strong>当前任务 ID</strong>
            <code>{taskId}</code>
          </div>
        </>
      ) : null}
    </aside>
  );
}
