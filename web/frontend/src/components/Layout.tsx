import { ChatInput } from "./ChatInput";
import { ConnectivityBanner } from "./ConnectivityBanner";
import { MessageList } from "./MessageList";
import { Sidebar } from "./Sidebar";
import { useChatSession } from "../hooks/useChatSession";
import "../styles/banner.css";
import "../styles/chat.css";
import "../styles/layout.css";
import "../styles/sidebar.css";

export function Layout() {
  const {
    baseUrl,
    setBaseUrl,
    messages,
    taskId,
    isOnline,
    isSending,
    streaming,
    resetConversation,
    handleUserInput,
    sendConfirmationReply,
    getDisplayContent,
    shouldShowConfirmation,
  } = useChatSession();

  return (
    <div className="app-layout">
      <Sidebar
        baseUrl={baseUrl}
        taskId={taskId}
        onBaseUrlChange={setBaseUrl}
        onNewConversation={resetConversation}
      />
      <div className="main-panel">
        <header className="page-header">
          <h1>电网智能助手</h1>
          <p>通过主控 Agent 统一调度统计、规划、投资等业务能力</p>
        </header>
        <div className="chat-area">
          <ConnectivityBanner isOnline={isOnline} baseUrl={baseUrl} />
          <MessageList
            messages={messages}
            streaming={streaming}
            getDisplayContent={getDisplayContent}
            shouldShowConfirmation={shouldShowConfirmation}
            onConfirmationSelect={sendConfirmationReply}
            isSending={isSending}
          />
          <ChatInput onSubmit={handleUserInput} disabled={isSending} />
        </div>
      </div>
    </div>
  );
}
