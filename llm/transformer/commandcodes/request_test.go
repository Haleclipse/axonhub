// 职责: 锁定 /alpha/generate 请求体、URL 与请求头契约，覆盖消息、工具与 system 的映射规则。
// 关联文件:
//   llm/transformer/commandcodes/request.go
//   llm/transformer/commandcodes/outbound.go

package commandcodes

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer"
)

func newTestTransformer(t *testing.T, config *Config) *OutboundTransformer {
	t.Helper()

	if config.APIKeyProvider == nil {
		config.APIKeyProvider = auth.NewStaticKeyProvider("sk-test")
	}

	outbound, err := NewOutboundTransformerWithConfig(config)
	require.NoError(t, err)

	transformerImpl, ok := outbound.(*OutboundTransformer)
	require.True(t, ok)
	transformerImpl.now = func() time.Time { return time.Date(2026, 11, 3, 10, 0, 0, 0, time.UTC) }

	return transformerImpl
}

func TestTransformRequest_MatchesCLIWireShape(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	req, err := outbound.TransformRequest(context.Background(), &llm.Request{
		Model:           "z-ai/glm-5.3-flash",
		Stream:          lo.ToPtr(false),
		MaxTokens:       lo.ToPtr(int64(4096)),
		ReasoningEffort: "low",
		Messages: []llm.Message{
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("You are a precise assistant. Follow the user request.")}},
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("请用 Python 实现二分查找，并给出三个测试用例。")}},
		},
	})
	require.NoError(t, err)

	require.Equal(t, http.MethodPost, req.Method)
	require.Equal(t, "https://api.commandcode.ai/alpha/generate", req.URL)
	require.Equal(t, "*/*", req.Headers.Get("Accept"))
	require.Equal(t, "application/json", req.Headers.Get("Content-Type"))
	require.Equal(t, CLIVersion, req.Headers.Get("x-command-code-version"))
	require.Equal(t, httpclient.AuthTypeBearer, req.Auth.Type)
	require.Equal(t, "sk-test", req.Auth.APIKey)
	require.Equal(t, llm.APIFormatCommandCodeGenerate.String(), req.APIFormat)
	require.Equal(t, llm.RequestTypeChat.String(), req.RequestType)
	require.NotNil(t, req.StreamDecoder)

	require.JSONEq(t, `{
		"config": {
			"workingDir": "/",
			"date": "2026-11-03",
			"environment": "linux",
			"structure": [],
			"isGitRepo": false,
			"currentBranch": "",
			"mainBranch": "",
			"gitStatus": "",
			"recentCommits": []
		},
		"memory": null,
		"taste": null,
		"skills": null,
		"permissionMode": "standard",
		"params": {
			"model": "z-ai/glm-5.3-flash",
			"system": "You are a precise assistant. Follow the user request.",
			"messages": [
				{"role": "user", "content": [{"type": "text", "text": "请用 Python 实现二分查找，并给出三个测试用例。"}]}
			],
			"tools": [],
			"max_tokens": 4096,
			"stream": true,
			"reasoning_effort": "low"
		}
	}`, string(req.Body))
}

func TestTransformRequest_ConvertsConversation(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	req, err := outbound.TransformRequest(context.Background(), &llm.Request{
		Model:               "claude-sonnet-4-5",
		MaxCompletionTokens: lo.ToPtr(int64(2048)),
		Temperature:         lo.ToPtr(0.2),
		ReasoningEffort:     "minimal",
		Messages: []llm.Message{
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("sys-1")}},
			{Role: "developer", Content: llm.MessageContent{Content: lo.ToPtr("sys-2")}},
			{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{
				{Type: "text", Text: lo.ToPtr("look")},
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/png;base64,AAAA"}},
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.com/a.jpg", MIMEType: "image/jpeg"}},
			}}},
			{
				Role:             "assistant",
				ReasoningContent: lo.ToPtr("thinking"),
				Content:          llm.MessageContent{Content: lo.ToPtr("calling")},
				ToolCalls: []llm.ToolCall{
					{ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"path":"a.go"}`}},
					{ID: "call_2", Type: "function", Function: llm.FunctionCall{Name: "list", Arguments: `not json`}},
				},
			},
			{Role: "tool", ToolCallID: lo.ToPtr("call_1"), Content: llm.MessageContent{Content: lo.ToPtr("package main")}},
			{Role: "tool", ToolCallID: lo.ToPtr("call_2"), ToolCallIsError: lo.ToPtr(true), Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{
				{Type: "text", Text: lo.ToPtr("line 1")},
				{Type: "text", Text: lo.ToPtr("line 2")},
			}}},
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("continue")}},
		},
		Tools: []llm.Tool{
			{Type: llm.ToolTypeFunction, Function: llm.Function{Name: "read", Description: "Read a file", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)}},
			{Type: llm.ToolTypeFunction, Function: llm.Function{Name: "list"}},
			{Type: "web_search"},
		},
	})
	require.NoError(t, err)

	params := gjson.GetBytes(req.Body, "params")
	require.Equal(t, "sys-1\n\nsys-2", params.Get("system").String())
	require.Equal(t, int64(2048), params.Get("max_tokens").Int())
	require.InDelta(t, 0.2, params.Get("temperature").Float(), 1e-9)
	require.Equal(t, "low", params.Get("reasoning_effort").String())
	require.True(t, params.Get("stream").Bool())

	require.JSONEq(t, `[
		{"role": "user", "content": [
			{"type": "text", "text": "look"},
			{"type": "image", "image": "data:image/png;base64,AAAA", "mimeType": "image/png"},
			{"type": "image", "image": "https://example.com/a.jpg", "mimeType": "image/jpeg"}
		]},
		{"role": "assistant", "content": [
			{"type": "reasoning", "text": "thinking"},
			{"type": "text", "text": "calling"},
			{"type": "tool-call", "toolCallId": "call_1", "toolName": "read", "input": {"path": "a.go"}},
			{"type": "tool-call", "toolCallId": "call_2", "toolName": "list", "input": {}}
		]},
		{"role": "tool", "content": [
			{"type": "tool-result", "toolCallId": "call_1", "toolName": "read", "output": {"type": "text", "value": "package main"}},
			{"type": "tool-result", "toolCallId": "call_2", "toolName": "list", "output": {"type": "error-text", "value": "line 1\nline 2"}}
		]},
		{"role": "user", "content": [{"type": "text", "text": "continue"}]}
	]`, params.Get("messages").Raw)

	require.JSONEq(t, `[
		{"name": "read", "description": "Read a file", "input_schema": {"type":"object","properties":{"path":{"type":"string"}}}},
		{"name": "list", "input_schema": {"type":"object","properties":{}}}
	]`, params.Get("tools").Raw)
}

func TestTransformRequest_SystemCacheControlUsesBlocks(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	req, err := outbound.TransformRequest(context.Background(), &llm.Request{
		Model: "claude-sonnet-4-5",
		Messages: []llm.Message{
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("static")}, CacheControl: &llm.CacheControl{Type: "ephemeral", TTL: "1h"}},
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("dynamic")}},
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}},
		},
	})
	require.NoError(t, err)

	require.JSONEq(t, `[
		{"type": "text", "text": "static", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
		{"type": "text", "text": "dynamic"}
	]`, gjson.GetBytes(req.Body, "params.system").Raw)
	require.Equal(t, defaultMaxTokens, gjson.GetBytes(req.Body, "params.max_tokens").Int())
	require.False(t, gjson.GetBytes(req.Body, "params.reasoning_effort").Exists())
}

func TestTransformRequest_DefaultSystemPrompt(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	// A missing or blank system makes the upstream inject its CLI agent prompt,
	// so the request always carries a non-blank system.
	for _, messages := range [][]llm.Message{
		{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}}},
		{
			{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("  ")}},
			{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}},
		},
	} {
		req, err := outbound.TransformRequest(context.Background(), &llm.Request{Model: "m", Messages: messages})
		require.NoError(t, err)
		require.Equal(t, defaultSystemPrompt, gjson.GetBytes(req.Body, "params.system").String())
	}
}

func TestTransformRequest_URLVariants(t *testing.T) {
	tests := []struct {
		name   string
		config *Config
		want   string
	}{
		{name: "trailing slash", config: &Config{BaseURL: "https://api.commandcode.ai/"}, want: "https://api.commandcode.ai/alpha/generate"},
		{name: "endpoint path override", config: &Config{BaseURL: "https://proxy.example.com/cc", EndpointPath: "/v2/generate"}, want: "https://proxy.example.com/cc/v2/generate"},
		{name: "raw URL", config: &Config{BaseURL: "https://proxy.example.com/generate##"}, want: "https://proxy.example.com/generate"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outbound := newTestTransformer(t, tt.config)
			req, err := outbound.TransformRequest(context.Background(), &llm.Request{
				Model:    "m",
				Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}}},
			})
			require.NoError(t, err)
			require.Equal(t, tt.want, req.URL)
		})
	}
}

func TestTransformRequest_RejectsUnsupportedInput(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	tests := []struct {
		name string
		req  *llm.Request
	}{
		{name: "document content", req: &llm.Request{Model: "m", Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{
			{Type: "document", Document: &llm.DocumentURL{URL: "data:application/pdf;base64,AAAA"}},
		}}}}}},
		{name: "embedding request", req: &llm.Request{Model: "m", RequestType: llm.RequestTypeEmbedding, Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}}}}},
		{name: "system only", req: &llm.Request{Model: "m", Messages: []llm.Message{{Role: "system", Content: llm.MessageContent{Content: lo.ToPtr("sys")}}}}},
		{name: "missing model", req: &llm.Request{Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}}}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := outbound.TransformRequest(context.Background(), tt.req)
			require.ErrorIs(t, err, transformer.ErrInvalidRequest)
		})
	}
}

func TestNewOutboundTransformer_ValidatesConfig(t *testing.T) {
	_, err := NewOutboundTransformerWithConfig(&Config{APIKeyProvider: auth.NewStaticKeyProvider("k")})
	require.Error(t, err)

	_, err = NewOutboundTransformerWithConfig(&Config{BaseURL: DefaultBaseURL})
	require.Error(t, err)
}
