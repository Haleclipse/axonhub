// 职责: 锁定 threadId 的推导契约：稳定、为合法 UUID、按会话键优先级取值，并按作用域隔离。
// 关联文件:
//   llm/transformer/commandcodes/thread.go
//   llm/transformer/commandcodes/outbound.go

package commandcodes

import (
	"context"
	"net/http"
	"testing"

	"github.com/google/uuid"
	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func threadOf(t *testing.T, ctx context.Context, req *llm.Request) string {
	t.Helper()

	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})
	httpReq, err := outbound.TransformRequest(ctx, req)
	require.NoError(t, err)

	threadID := gjson.GetBytes(httpReq.Body, "threadId").String()
	require.Equal(t, threadID, httpReq.Headers.Get("x-session-id"))

	parsed, err := uuid.Parse(threadID)
	require.NoError(t, err)
	require.Equal(t, uuid.Version(5), parsed.Version())

	return threadID
}

func conversation(turns ...string) *llm.Request {
	req := &llm.Request{
		Model:    "deepseek/deepseek-v4.1-flash",
		Messages: []llm.Message{{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("You are concise.")}}},
	}
	for i, turn := range turns {
		if i > 0 {
			req.Messages = append(req.Messages, llm.Message{Role: "assistant", Content: llm.MessageContent{Content: lo.ToPtr("reply")}})
		}
		req.Messages = append(req.Messages, llm.Message{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr(turn)}})
	}

	return req
}

func TestThreadID_SessionTakesPrecedenceAndIsStable(t *testing.T) {
	ctx := shared.WithSessionID(context.Background(), "trace-123")

	first := threadOf(t, ctx, conversation("hello"))
	later := threadOf(t, ctx, conversation("hello", "again", "more"))
	require.Equal(t, first, later)

	withCacheKey := conversation("hello")
	withCacheKey.PromptCacheKey = lo.ToPtr("cache-key")
	require.Equal(t, first, threadOf(t, ctx, withCacheKey), "the gateway session wins over prompt_cache_key")

	require.NotEqual(t, first, threadOf(t, shared.WithSessionID(context.Background(), "trace-456"), conversation("hello")))
}

func TestThreadID_ScopeIsolatesTenants(t *testing.T) {
	ctx := shared.WithSessionID(context.Background(), "trace-123")

	userA := threadOf(t, shared.WithSessionScope(ctx, "user:1"), conversation("hello"))
	userB := threadOf(t, shared.WithSessionScope(ctx, "user:2"), conversation("hello"))
	require.NotEqual(t, userA, userB)
}

func TestThreadID_PromptCacheKeyWithoutSession(t *testing.T) {
	first := conversation("hello")
	first.PromptCacheKey = lo.ToPtr("conv-1")
	later := conversation("hello", "again")
	later.PromptCacheKey = lo.ToPtr("conv-1")
	other := conversation("hello")
	other.PromptCacheKey = lo.ToPtr("conv-2")

	require.Equal(t, threadOf(t, context.Background(), first), threadOf(t, context.Background(), later))
	require.NotEqual(t, threadOf(t, context.Background(), first), threadOf(t, context.Background(), other))
}

func TestThreadID_ConversationFallback(t *testing.T) {
	ctx := context.Background()

	// Later turns only append, so the anchor (system + first message) is unchanged.
	require.Equal(t, threadOf(t, ctx, conversation("hello")), threadOf(t, ctx, conversation("hello", "again", "more")))
	require.NotEqual(t, threadOf(t, ctx, conversation("hello")), threadOf(t, ctx, conversation("goodbye")))
}

func TestFinalizeTransportRequest_RestoresSessionHeader(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})
	httpReq, err := outbound.TransformRequest(context.Background(), conversation("hello"))
	require.NoError(t, err)
	threadID := httpReq.Headers.Get("x-session-id")

	// A client X-Session-Id merged from the inbound request must not survive.
	merged := httpclient.MergeInboundRequest(httpReq, &httpclient.Request{Headers: http.Header{"X-Session-Id": []string{"client-session"}}})
	require.Equal(t, "client-session", merged.Headers.Get("x-session-id"))

	finalized := outbound.FinalizeTransportRequest(merged)
	require.Equal(t, threadID, finalized.Headers.Get("x-session-id"))
}
