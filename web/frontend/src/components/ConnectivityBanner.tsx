interface ConnectivityBannerProps {
  isOnline: boolean | null;
  baseUrl: string;
}

export function ConnectivityBanner({
  isOnline,
  baseUrl,
}: ConnectivityBannerProps) {
  if (isOnline === null) {
    return (
      <div className="connectivity-banner checking">
        正在检查主控 Agent 连接状态...
      </div>
    );
  }

  if (isOnline) {
    return (
      <div className="connectivity-banner online">
        已连接主控 Agent：{baseUrl}
      </div>
    );
  }

  return (
    <div className="connectivity-banner offline">
      无法连接主控 Agent（{baseUrl}），请确认服务已启动：
      <code>python main_agent/server.py</code>
    </div>
  );
}
