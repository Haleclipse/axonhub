// 职责: 锁定 NDJSON 事件到统一响应的转换契约：流式增量、终止判定、错误、用量与非流式聚合。
// 关联文件:
//   llm/transformer/commandcodes/stream.go
//   llm/transformer/commandcodes/aggregate.go
//   llm/transformer/commandcodes/outbound.go
// 注意事项:
//   testdata 下的 .ndjson 是 /alpha/generate 的真实抓包（已剔除 finish-step 中的上游响应头）。

package commandcodes

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

const (
	deepSeekFixture = "deepseek_reasoning_tool_calls.ndjson"
	glmFixture      = "glm_text.ndjson"

	deepSeekModel = "deepseek/deepseek-v4.1-flash"
	glmModel      = "z-ai/glm-5.3-flash"
)

func ndjsonEvents(lines ...string) []*httpclient.StreamEvent {
	return lo.Map(lines, func(line string, _ int) *httpclient.StreamEvent {
		return &httpclient.StreamEvent{Data: []byte(line)}
	})
}

func fixtureBytes(t *testing.T, name string) []byte {
	t.Helper()

	data, err := os.ReadFile(filepath.Join("testdata", name))
	require.NoError(t, err)

	return data
}

func fixtureEvents(t *testing.T, name string) []*httpclient.StreamEvent {
	t.Helper()

	return ndjsonEvents(strings.Split(strings.TrimSpace(string(fixtureBytes(t, name))), "\n")...)
}

func generateHTTPRequest(model string) *httpclient.Request {
	return &httpclient.Request{Body: []byte(`{"params":{"model":"` + model + `"}}`)}
}

func TestTransformStream_DeepSeekReasoningAndParallelToolCalls(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	stream, err := outbound.TransformStream(context.Background(), generateHTTPRequest(deepSeekModel), streams.SliceStream(fixtureEvents(t, deepSeekFixture)))
	require.NoError(t, err)

	chunks, err := streams.All(stream)
	require.NoError(t, err)
	// 6 reasoning deltas, 2 x (tool header + 9 argument deltas), finish, done.
	require.Len(t, chunks, 28)

	require.Equal(t, "assistant", chunks[0].Choices[0].Delta.Role)
	require.Empty(t, chunks[1].Choices[0].Delta.Role)

	var reasoning strings.Builder
	for _, chunk := range chunks[:6] {
		reasoning.WriteString(lo.FromPtr(chunk.Choices[0].Delta.ReasoningContent))
	}
	require.Equal(t, "We need answer. Need call get_weather for Paris and Tokyo in parallel.", reasoning.String())

	header := chunks[6].Choices[0].Delta.ToolCalls[0]
	require.Equal(t, llm.ToolCall{
		Index:    0,
		ID:       "call_00_17vijiekzqudsl4ftgrvw533",
		Type:     "function",
		Function: llm.FunctionCall{Name: "get_weather"},
	}, header)

	argumentDelta := chunks[7].Choices[0].Delta.ToolCalls[0]
	require.Equal(t, 0, argumentDelta.Index)
	require.Equal(t, "call_00_17vijiekzqudsl4ftgrvw533", argumentDelta.ID)
	require.Empty(t, argumentDelta.Function.Name)
	require.Equal(t, "{", argumentDelta.Function.Arguments)

	second := chunks[16].Choices[0].Delta.ToolCalls[0]
	require.Equal(t, 1, second.Index)
	require.Equal(t, "call_01_xflyw9u6zup4qmzbfpntugsm", second.ID)

	finish := chunks[26]
	require.Equal(t, "tool_calls", lo.FromPtr(finish.Choices[0].FinishReason))
	require.Equal(t, &llm.Usage{
		PromptTokens:            317,
		CompletionTokens:        84,
		TotalTokens:             401,
		CompletionTokensDetails: &llm.CompletionTokensDetails{ReasoningTokens: 16},
	}, finish.Usage)

	require.Same(t, llm.DoneResponse, chunks[27])

	for _, chunk := range chunks[:27] {
		require.Equal(t, chunks[0].ID, chunk.ID)
		require.Equal(t, deepSeekModel, chunk.Model)
		require.Equal(t, "chat.completion.chunk", chunk.Object)
	}
}

func TestTransformStream_StopsAtFinish(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	// The upstream sends provider-metadata after finish; anything after the
	// terminal event, including an error, must not be read.
	events := append(fixtureEvents(t, glmFixture), ndjsonEvents(`{"type":"error","error":"must not be read"}`)...)

	stream, err := outbound.TransformStream(context.Background(), generateHTTPRequest(glmModel), streams.SliceStream(events))
	require.NoError(t, err)

	chunks, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, chunks, 3)
	require.Equal(t, "pong", lo.FromPtr(chunks[0].Choices[0].Delta.Content.Content))
	require.Equal(t, "stop", lo.FromPtr(chunks[1].Choices[0].FinishReason))
	require.Equal(t, &llm.Usage{PromptTokens: 29, CompletionTokens: 3, TotalTokens: 32}, chunks[1].Usage)
	require.Same(t, llm.DoneResponse, chunks[2])
}

func TestTransformStream_EndWithoutFinishIsIncomplete(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	stream, err := outbound.TransformStream(context.Background(), generateHTTPRequest("m"), streams.SliceStream(ndjsonEvents(
		`{"type":"text-delta","text":"partial"}`,
	)))
	require.NoError(t, err)

	chunks, err := streams.All(stream)
	require.ErrorIs(t, err, llm.ErrStreamIncomplete)
	require.Len(t, chunks, 1)
}

func TestTransformStream_ErrorEvents(t *testing.T) {
	tests := []struct {
		name       string
		line       string
		wantStatus int
		wantType   string
		wantMsg    string
	}{
		{
			// Captured from the upstream: HTTP 200 with an error part.
			name:       "upstream validation error",
			line:       `{"type":"error","error":{"type":"server_error","message":"system: text content blocks must contain non-whitespace text","statusCode":400,"isRetryable":false}}`,
			wantStatus: http.StatusBadRequest,
			wantType:   "server_error",
			wantMsg:    "system: text content blocks must contain non-whitespace text",
		},
		{
			name:       "embedded upstream error",
			line:       `{"type":"error","error":{"message":"429 {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"slow down\"}}"}}`,
			wantStatus: http.StatusTooManyRequests,
			wantType:   "rate_limit_error",
			wantMsg:    "slow down",
		},
		{
			name:       "string error",
			line:       `{"type":"error","error":"boom"}`,
			wantStatus: http.StatusBadGateway,
			wantType:   "stream_error",
			wantMsg:    "boom",
		},
		{
			name:       "abort",
			line:       `{"type":"abort"}`,
			wantStatus: http.StatusBadGateway,
			wantType:   "stream_aborted",
			wantMsg:    "Command Code aborted the generation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

			stream, err := outbound.TransformStream(context.Background(), generateHTTPRequest("m"), streams.SliceStream(ndjsonEvents(
				`{"type":"start"}`,
				tt.line,
			)))
			require.NoError(t, err)

			_, err = streams.All(stream)

			var respErr *llm.ResponseError
			require.ErrorAs(t, err, &respErr)
			require.Equal(t, tt.wantStatus, respErr.StatusCode)
			require.Equal(t, tt.wantType, respErr.Detail.Type)
			require.Equal(t, tt.wantMsg, respErr.Detail.Message)
		})
	}
}

func TestEventProcessor_ToolCallsAndFinishReasons(t *testing.T) {
	tests := []struct {
		name       string
		events     []string
		wantFinish string
		wantCalls  []llm.ToolCall
	}{
		{name: "stop", events: []string{`{"type":"finish","finishReason":"stop"}`}, wantFinish: "stop"},
		{name: "length", events: []string{`{"type":"finish","finishReason":"length"}`}, wantFinish: "length"},
		{name: "refusal", events: []string{`{"type":"finish","finishReason":"other","rawFinishReason":"refusal"}`}, wantFinish: "content_filter"},
		{name: "raw max tokens", events: []string{`{"type":"finish","finishReason":"other","rawFinishReason":"max_tokens"}`}, wantFinish: "length"},
		{name: "tool-calls without forwarded calls", events: []string{`{"type":"finish","finishReason":"tool-calls"}`}, wantFinish: "stop"},
		{
			name: "started call without argument deltas takes the final input",
			events: []string{
				`{"type":"tool-input-start","id":"c1","toolName":"read"}`,
				`{"type":"tool-input-end","id":"c1"}`,
				`{"type":"tool-call","toolCallId":"c1","toolName":"read","input":{"path":"a.go"}}`,
				`{"type":"finish","finishReason":"tool-calls"}`,
			},
			wantFinish: "tool_calls",
			wantCalls:  []llm.ToolCall{{Index: 0, ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"path":"a.go"}`}}},
		},
		{
			name: "streamed arguments are not repeated by tool-call",
			events: []string{
				`{"type":"tool-input-start","id":"c1","toolName":"read"}`,
				`{"type":"tool-input-delta","id":"c1","delta":"{\"path\": "}`,
				`{"type":"tool-input-delta","id":"c1","delta":"\"a.go\"}"}`,
				`{"type":"tool-call","toolCallId":"c1","toolName":"read","input":{"path":"a.go"}}`,
				`{"type":"finish","finishReason":"tool-calls"}`,
			},
			wantFinish: "tool_calls",
			wantCalls:  []llm.ToolCall{{Index: 0, ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "read", Arguments: `{"path": "a.go"}`}}},
		},
		{
			name: "tool-call without start, string and legacy array inputs",
			events: []string{
				`{"type":"tool-call","toolCallId":"c1","toolName":"t","input":"{\"a\":1}"}`,
				`{"type":"tool-call","toolCallId":"c2","toolName":"t","args":[{"a":2}]}`,
				`{"type":"tool-call","toolCallId":"c3","toolName":"t","input":null}`,
				`{"type":"finish","finishReason":"tool-calls"}`,
			},
			wantFinish: "tool_calls",
			wantCalls: []llm.ToolCall{
				{Index: 0, ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "t", Arguments: `{"a":1}`}},
				{Index: 1, ID: "c2", Type: "function", Function: llm.FunctionCall{Name: "t", Arguments: `{"a":2}`}},
				{Index: 2, ID: "c3", Type: "function", Function: llm.FunctionCall{Name: "t", Arguments: `{}`}},
			},
		},
		{
			name: "provider-executed calls are dropped",
			events: []string{
				`{"type":"tool-input-start","id":"s1","toolName":"web_search","providerExecuted":true}`,
				`{"type":"tool-input-delta","id":"s1","delta":"{}"}`,
				`{"type":"tool-call","toolCallId":"s1","toolName":"web_search","input":{},"providerExecuted":true}`,
				`{"type":"tool-result","toolCallId":"s1","toolName":"web_search","output":"ok","providerExecuted":true}`,
				`{"type":"finish","finishReason":"tool-calls"}`,
			},
			wantFinish: "stop",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, meta, err := aggregateEvents("m", ndjsonEvents(tt.events...))
			require.NoError(t, err)
			require.True(t, meta.Completed)
			require.Equal(t, tt.wantFinish, lo.FromPtr(resp.Choices[0].FinishReason))
			require.Equal(t, tt.wantCalls, resp.Choices[0].Message.ToolCalls)
		})
	}
}

func TestConvertUsage_CacheDetails(t *testing.T) {
	resp, meta, err := aggregateEvents("m", ndjsonEvents(
		`{"type":"text-delta","text":"ok"}`,
		`{"type":"finish","finishReason":"stop","totalUsage":{"inputTokens":7381,"inputTokenDetails":{"noCacheTokens":213,"cacheReadTokens":7168},"outputTokens":3,"cachedInputTokens":7168}}`,
	))
	require.NoError(t, err)
	require.Equal(t, llm.AccountingEvidenceExplicitFinal, meta.AccountingEvidence)
	require.Equal(t, &llm.Usage{
		PromptTokens:        7381,
		CompletionTokens:    3,
		TotalTokens:         7384,
		PromptTokensDetails: &llm.PromptTokensDetails{CachedTokens: 7168},
	}, resp.Usage)

	usage := convertUsage(gjson.Parse(`{"inputTokens":13519,"inputTokenDetails":{"noCacheTokens":4,"cacheReadTokens":0,"cacheWriteTokens":13515},"outputTokens":4}`))
	require.Equal(t, &llm.PromptTokensDetails{WriteCachedTokens: 13515}, usage.PromptTokensDetails)
	require.Equal(t, int64(13519), usage.PromptTokens)
}

func TestAggregateStreamChunks(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	body, meta, err := outbound.AggregateStreamChunks(context.Background(), generateHTTPRequest(deepSeekModel), fixtureEvents(t, deepSeekFixture))
	require.NoError(t, err)
	require.True(t, meta.Completed)
	require.Equal(t, llm.AccountingEvidenceExplicitFinal, meta.AccountingEvidence)
	require.Equal(t, int64(401), meta.Usage.TotalTokens)

	resp, err := outbound.TransformResponse(context.Background(), &httpclient.Response{Body: body})
	require.NoError(t, err)
	require.Equal(t, "chat.completion", resp.Object)
	require.Equal(t, deepSeekModel, resp.Model)
	require.Empty(t, lo.FromPtr(resp.Choices[0].Message.Content.Content))
	require.Equal(t, "We need answer. Need call get_weather for Paris and Tokyo in parallel.", lo.FromPtr(resp.Choices[0].Message.ReasoningContent))
	require.Equal(t, []llm.ToolCall{
		{Index: 0, ID: "call_00_17vijiekzqudsl4ftgrvw533", Type: "function", Function: llm.FunctionCall{Name: "get_weather", Arguments: `{"city": "Paris"}`}},
		{Index: 1, ID: "call_01_xflyw9u6zup4qmzbfpntugsm", Type: "function", Function: llm.FunctionCall{Name: "get_weather", Arguments: `{"city": "Tokyo"}`}},
	}, resp.Choices[0].Message.ToolCalls)
	require.Equal(t, "tool_calls", lo.FromPtr(resp.Choices[0].FinishReason))

	// Partial streams aggregate without claiming completion.
	_, meta, err = outbound.AggregateStreamChunks(context.Background(), generateHTTPRequest("m"), ndjsonEvents(`{"type":"text-delta","text":"a"}`))
	require.NoError(t, err)
	require.False(t, meta.Completed)
	require.Equal(t, llm.AccountingEvidenceUnavailable, meta.AccountingEvidence)
}

type sliceExecutor struct {
	events []*httpclient.StreamEvent
	err    error
}

func (e *sliceExecutor) Do(context.Context, *httpclient.Request) (*httpclient.Response, error) {
	return nil, errors.New("Do must not be called on the inner executor")
}

func (e *sliceExecutor) DoStream(context.Context, *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
	if e.err != nil {
		return nil, e.err
	}

	return streams.SliceStream(e.events), nil
}

func TestGenerateExecutor_DoAggregatesStream(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})
	executor := outbound.CustomizeExecutor(&sliceExecutor{events: fixtureEvents(t, glmFixture)})

	httpResp, err := executor.Do(context.Background(), generateHTTPRequest(glmModel))
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, httpResp.StatusCode)
	require.Equal(t, "application/json", httpResp.Headers.Get("Content-Type"))

	resp, err := outbound.TransformResponse(context.Background(), httpResp)
	require.NoError(t, err)
	require.Equal(t, "pong", lo.FromPtr(resp.Choices[0].Message.Content.Content))
	require.Equal(t, int64(29), resp.Usage.PromptTokens)
}

func TestGenerateExecutor_DoFailures(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	t.Run("incomplete stream", func(t *testing.T) {
		executor := outbound.CustomizeExecutor(&sliceExecutor{events: ndjsonEvents(`{"type":"text-delta","text":"a"}`)})
		_, err := executor.Do(context.Background(), generateHTTPRequest("m"))
		require.ErrorIs(t, err, llm.ErrStreamIncomplete)
	})

	t.Run("error event", func(t *testing.T) {
		executor := outbound.CustomizeExecutor(&sliceExecutor{events: ndjsonEvents(`{"type":"error","error":{"message":"denied","statusCode":403}}`)})
		_, err := executor.Do(context.Background(), generateHTTPRequest("m"))

		var respErr *llm.ResponseError
		require.ErrorAs(t, err, &respErr)
		require.Equal(t, http.StatusForbidden, respErr.StatusCode)
	})

	t.Run("upstream HTTP error", func(t *testing.T) {
		httpErr := &httpclient.Error{StatusCode: http.StatusForbidden, Body: []byte(`{"error":{"code":"upgrade_required","message":"Your Command Code CLI is out of date.","minVersion":"0.18.10"}}`)}
		executor := outbound.CustomizeExecutor(&sliceExecutor{err: httpErr})
		_, err := executor.Do(context.Background(), generateHTTPRequest("m"))
		require.ErrorIs(t, err, httpErr)
	})
}

func TestTransformError_UpstreamShapes(t *testing.T) {
	outbound := newTestTransformer(t, &Config{BaseURL: DefaultBaseURL})

	tests := []struct {
		name string
		err  *httpclient.Error
		want llm.ErrorDetail
	}{
		{
			name: "missing version header",
			err:  &httpclient.Error{StatusCode: http.StatusForbidden, Body: []byte(`{"error":{"code":"upgrade_required","message":"Your Command Code CLI is out of date.","minVersion":"0.18.10"}}`)},
			want: llm.ErrorDetail{Code: "upgrade_required", Message: "Your Command Code CLI is out of date.", Type: "api_error"},
		},
		{
			name: "unknown model",
			err:  &httpclient.Error{StatusCode: http.StatusForbidden, Body: []byte(`{"success":false,"error":{"code":"FORBIDDEN","status":403,"message":"Model/provider not recognized: anthropic:x"}}`)},
			want: llm.ErrorDetail{Code: "FORBIDDEN", Message: "Model/provider not recognized: anthropic:x", Type: "api_error"},
		},
		{
			name: "string error",
			err:  &httpclient.Error{StatusCode: http.StatusBadRequest, Body: []byte(`{"error":"invalid model"}`)},
			want: llm.ErrorDetail{Message: "invalid model", Type: "api_error"},
		},
		{
			name: "plain text",
			err:  &httpclient.Error{StatusCode: http.StatusBadGateway, Body: []byte("upstream down")},
			want: llm.ErrorDetail{Message: "upstream down", Type: "api_error"},
		},
		{
			name: "empty body",
			err:  &httpclient.Error{StatusCode: http.StatusServiceUnavailable},
			want: llm.ErrorDetail{Message: http.StatusText(http.StatusServiceUnavailable), Type: "api_error"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			respErr := outbound.TransformError(context.Background(), tt.err)
			require.Equal(t, tt.err.StatusCode, respErr.StatusCode)
			require.Equal(t, tt.want, respErr.Detail)
		})
	}
}

// TestEndToEnd_EventStreamContentType drives the real HTTP client against a
// server that, like the upstream, labels the NDJSON body text/event-stream.
func TestEndToEnd_EventStreamContentType(t *testing.T) {
	var gotAuth, gotVersion, gotAccept string

	body := fixtureBytes(t, deepSeekFixture)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		gotAuth = request.Header.Get("Authorization")
		gotVersion = request.Header.Get("x-command-code-version")
		gotAccept = request.Header.Get("Accept")

		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = writer.Write(bytes.Clone(body))
	}))
	t.Cleanup(server.Close)

	outbound := newTestTransformer(t, &Config{BaseURL: server.URL})
	executor := outbound.CustomizeExecutor(httpclient.NewHttpClientWithClient(server.Client()))

	newRequest := func() *httpclient.Request {
		req, err := outbound.TransformRequest(context.Background(), &llm.Request{
			Model:    deepSeekModel,
			Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hi")}}},
		})
		require.NoError(t, err)

		req, err = httpclient.FinalizeAuthHeaders(req)
		require.NoError(t, err)

		return req
	}

	t.Run("stream", func(t *testing.T) {
		req := newRequest()
		raw, err := executor.DoStream(context.Background(), req)
		require.NoError(t, err)

		stream, err := outbound.TransformStream(context.Background(), req, raw)
		require.NoError(t, err)

		chunks, err := streams.All(stream)
		require.NoError(t, err)
		require.Len(t, chunks, 28)
		require.Same(t, llm.DoneResponse, chunks[27])
		require.Equal(t, "Bearer sk-test", gotAuth)
		require.Equal(t, CLIVersion, gotVersion)
		require.Equal(t, "*/*", gotAccept)
	})

	t.Run("non-stream", func(t *testing.T) {
		httpResp, err := executor.Do(context.Background(), newRequest())
		require.NoError(t, err)

		resp, err := outbound.TransformResponse(context.Background(), httpResp)
		require.NoError(t, err)
		require.Len(t, resp.Choices[0].Message.ToolCalls, 2)
		require.Equal(t, "tool_calls", lo.FromPtr(resp.Choices[0].FinishReason))
	})
}
