// 职责: 为每段对话推导稳定的 Command Code threadId（同时用作 x-session-id），让上游把同一前缀的请求归到同一 thread。
// 关联文件:
//   llm/transformer/commandcodes/outbound.go
//   llm/transformer/commandcodes/request.go
//   llm/transformer/shared/session.go
// 注意事项:
//   官方 CLI 在主对话每一轮都发送同一个 threadId（会话 UUID），上游按 thread 建立前缀缓存亲和；
//   threadId 逐轮变化时，只有所有会话共用的 system+tools 前缀会命中（缓存读取停在一个固定的块边界上）。
//   不能使用 shared.GetSessionID：Responses 入站在没有 trace 来源时会为每个请求合成随机会话 ID，
//   用它会让 threadId 每轮都变。会话键只取 prompt_cache_key 和对话锚点（system + 首条消息）。
//   上游只接受 UUID，会话键经 UUIDv5 映射；同一输入必须得到同一 UUID，不能引入随机数或时间。
//   作用域（用户/API Key）参与映射，避免不同租户的同名会话共用 thread。

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

	return uuid.NewSHA1(threadNamespace, []byte(strings.TrimSpace(scope)+"\x00"+conversationKey(llmReq, params))).String()
}

func conversationKey(llmReq *llm.Request, params *generateParams) string {
	if cacheKey := strings.TrimSpace(lo.FromPtr(llmReq.PromptCacheKey)); cacheKey != "" {
		return "prompt_cache_key:" + cacheKey
	}

	// The system prompt and the first message stay fixed for a whole
	// conversation while later turns only append, so they identify the prefix
	// the upstream cache can reuse. Only the system text counts: a client that
	// toggles cache markers switches the wire form between string and blocks.
	anchor, _ := json.Marshal([]any{systemText(params.System), lo.FirstOrEmpty(params.Messages)})

	return "conversation:" + string(anchor)
}

func systemText(system any) string {
	switch value := system.(type) {
	case string:
		return value
	case []wireSystemBlock:
		return strings.Join(lo.Map(value, func(block wireSystemBlock, _ int) string { return block.Text }), "\n\n")
	default:
		return ""
	}
}
