// 职责: 为每段对话推导稳定的 Command Code threadId（同时用作 x-session-id），让上游把同一会话的请求归到同一 thread。
// 关联文件:
//   llm/transformer/commandcodes/outbound.go
//   llm/transformer/commandcodes/request.go
//   llm/transformer/shared/session.go
// 注意事项:
//   官方 CLI 在主对话每一轮都发送同一个 threadId（会话 UUID），上游按 thread 管理提示缓存；
//   缺失时同一会话的请求无法稳定命中前缀缓存。
//   上游只接受 UUID，任意会话键都要经 UUIDv5 映射；同一输入必须得到同一 UUID，不能引入随机数或时间。
//   会话键优先级：网关会话 ID → prompt_cache_key → system 与首条消息的哈希；作用域（用户/API Key）参与映射，
//   避免不同租户的同名会话共用 thread。

package commandcodes

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/google/uuid"
	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

// threadNamespace is the UUIDv5 namespace for Command Code thread IDs.
var threadNamespace = uuid.NewSHA1(uuid.NameSpaceURL, []byte("axonhub:commandcodes:thread"))

// resolveThreadID maps the conversation identity to a deterministic UUID.
func resolveThreadID(ctx context.Context, llmReq *llm.Request, params *generateParams) string {
	scope, _ := shared.GetSessionScope(ctx)

	return uuid.NewSHA1(threadNamespace, []byte(strings.TrimSpace(scope)+"\x00"+conversationKey(ctx, llmReq, params))).String()
}

func conversationKey(ctx context.Context, llmReq *llm.Request, params *generateParams) string {
	if sessionID, ok := shared.GetSessionID(ctx); ok && strings.TrimSpace(sessionID) != "" {
		return "session:" + strings.TrimSpace(sessionID)
	}

	if cacheKey := strings.TrimSpace(lo.FromPtr(llmReq.PromptCacheKey)); cacheKey != "" {
		return "prompt_cache_key:" + cacheKey
	}

	// Without a client session, the system prompt and the first message stay
	// fixed for the whole conversation while later turns only append.
	anchor, _ := json.Marshal([]any{params.System, lo.FirstOrEmpty(params.Messages)})

	return "conversation:" + string(anchor)
}
