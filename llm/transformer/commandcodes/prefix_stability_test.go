// 职责: 验证 Claude Code 风格的多轮 Anthropic 请求经网关转换后，发往 /alpha/generate 的前缀逐轮稳定。
// 关联文件:
//   llm/transformer/commandcodes/request.go
//   llm/transformer/commandcodes/thread.go
//   llm/transformer/anthropic/inbound.go
// 注意事项:
//   上游前缀缓存要求第 N+1 轮的 system、tools 与历史 messages 与第 N 轮逐字节一致；
//   Claude Code 会逐轮移动 cache_control 标记，并在对话中途追加 system 消息，转换结果不能因此改变已有前缀。

package commandcodes

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
)

const claudeCodeTurnTemplate = `{
  "model": "deepseek/deepseek-v4.1-flash",
  "max_tokens": 32000,
  "stream": true,
  "metadata": {"user_id": "{\"device_id\":\"dev\",\"account_uuid\":\"acc\",\"session_id\":\"6b3c2a1e-0000-4000-8000-000000000001\"}"},
  "thinking": {"type": "enabled", "budget_tokens": 8000},
  "system": [
    {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
    {"type": "text", "text": "Long static instructions.", "cache_control": {"type": "ephemeral"}}
  ],
  "tools": [
    {"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}
  ],
  "messages": %s
}`

var claudeCodeTurns = []string{
	`[
	  {"role": "user", "content": [
	    {"type": "text", "text": "<system-reminder>project context</system-reminder>"},
	    {"type": "text", "text": "Read main.go", "cache_control": {"type": "ephemeral"}}]}
	]`,
	`[
	  {"role": "user", "content": [
	    {"type": "text", "text": "<system-reminder>project context</system-reminder>"},
	    {"type": "text", "text": "Read main.go"}]},
	  {"role": "assistant", "content": [
	    {"type": "thinking", "thinking": "I should read the file.", "signature": ""},
	    {"type": "text", "text": "Reading."},
	    {"type": "tool_use", "id": "call_00_abc", "name": "Read", "input": {"file_path": "main.go"}}]},
	  {"role": "user", "content": [
	    {"type": "tool_result", "tool_use_id": "call_00_abc", "content": "package main", "cache_control": {"type": "ephemeral"}}]},
	  {"role": "system", "content": "<system-reminder>The todo list changed.</system-reminder>"}
	]`,
	`[
	  {"role": "user", "content": [
	    {"type": "text", "text": "<system-reminder>project context</system-reminder>"},
	    {"type": "text", "text": "Read main.go"}]},
	  {"role": "assistant", "content": [
	    {"type": "thinking", "thinking": "I should read the file.", "signature": ""},
	    {"type": "text", "text": "Reading."},
	    {"type": "tool_use", "id": "call_00_abc", "name": "Read", "input": {"file_path": "main.go"}}]},
	  {"role": "user", "content": [
	    {"type": "tool_result", "tool_use_id": "call_00_abc", "content": "package main"}]},
	  {"role": "system", "content": "<system-reminder>The todo list changed.</system-reminder>"},
	  {"role": "assistant", "content": [
	    {"type": "thinking", "thinking": "Summarize.", "signature": ""},
	    {"type": "text", "text": "It is an empty main package."}]},
	  {"role": "user", "content": [{"type": "text", "text": "thanks", "cache_control": {"type": "ephemeral"}}]}
	]`,
}

func TestClaudeCodeTurns_UpstreamPrefixIsStable(t *testing.T) {
	inbound := anthropic.NewInboundTransformer()
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	bodies := make([]gjson.Result, 0, len(claudeCodeTurns))
	for _, turn := range claudeCodeTurns {
		raw := []byte(fmtTurn(turn))
		require.True(t, json.Valid(raw))

		llmReq, err := inbound.TransformRequest(context.Background(), &httpclient.Request{
			Method:  http.MethodPost,
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    raw,
		})
		require.NoError(t, err)

		httpReq, err := outbound.TransformRequest(context.Background(), llmReq)
		require.NoError(t, err)
		bodies = append(bodies, gjson.ParseBytes(httpReq.Body))
	}

	for i := 1; i < len(bodies); i++ {
		prev, cur := bodies[i-1], bodies[i]
		require.Equal(t, prev.Get("threadId").String(), cur.Get("threadId").String(), "turn %d threadId", i+1)
		require.Equal(t, prev.Get("params.system").Raw, cur.Get("params.system").Raw, "turn %d system", i+1)
		require.Equal(t, prev.Get("params.tools").Raw, cur.Get("params.tools").Raw, "turn %d tools", i+1)

		prevMessages, curMessages := prev.Get("params.messages").Array(), cur.Get("params.messages").Array()
		require.Greater(t, len(curMessages), len(prevMessages))
		for j := range prevMessages {
			require.JSONEq(t, prevMessages[j].Raw, curMessages[j].Raw, "turn %d message %d diverges from the previous turn", i+1, j)
			require.Equal(t, prevMessages[j].Raw, curMessages[j].Raw, "turn %d message %d bytes diverge", i+1, j)
		}
	}
}

func fmtTurn(messages string) string {
	return fmt.Sprintf(claudeCodeTurnTemplate, messages)
}
