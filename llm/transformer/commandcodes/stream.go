// 职责: 把 /alpha/generate 的 NDJSON 事件（AI SDK fullStream parts）转换为统一 llm.Response 增量块。
// 关联文件:
//   llm/transformer/commandcodes/outbound.go
//   llm/transformer/commandcodes/aggregate.go
//   llm/transformer/commandcodes/decoder.go
// 注意事项:
//   只有 finish 事件代表正常结束；源流在 finish 之前结束必须报 llm.ErrStreamIncomplete，
//   否则截断的回答会被当成成功。
//   工具参数随 tool-input-start/tool-input-delta 流式转发；tool-call 只在参数没有流出时补发完整 input，
//   否则会重复拼接参数。
//   providerExecuted 的工具调用已由上游执行，不能交给客户端再执行。
//   上游错误可能以 HTTP 200 + error 事件出现，状态码取 error.statusCode 或消息中内嵌的状态码。

package commandcodes

import (
	"bytes"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

// eventProcessor converts NDJSON events into unified chunks. It keeps the
// per-response state shared by streaming and aggregation.
type eventProcessor struct {
	id      string
	model   string
	created int64

	roleSent    bool
	toolIndex   int
	sawToolCall bool
	// toolCalls tracks calls opened by tool-input-start, keyed by tool call ID.
	toolCalls map[string]*openToolCall

	finished bool
}

type openToolCall struct {
	index        int
	hasArguments bool
}

func newEventProcessor(model string) *eventProcessor {
	return &eventProcessor{
		id:        "cmdc-" + uuid.NewString(),
		model:     model,
		created:   time.Now().Unix(),
		toolCalls: make(map[string]*openToolCall),
	}
}

// process converts one NDJSON line into at most one unified chunk. Lines that
// are not JSON are skipped, matching the official CLI.
//
//nolint:nilnil // A nil chunk without error means the event produces no output.
func (p *eventProcessor) process(data []byte) (*llm.Response, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || !gjson.ValidBytes(data) {
		return nil, nil
	}

	event := gjson.ParseBytes(data)

	switch event.Get("type").String() {
	case "text-delta":
		text := event.Get("text").String()
		if text == "" {
			return nil, nil
		}

		return p.chunk(&llm.Message{Content: llm.MessageContent{Content: &text}}), nil
	case "reasoning-delta":
		text := event.Get("text").String()
		if text == "" {
			return nil, nil
		}

		return p.chunk(&llm.Message{ReasoningContent: &text}), nil
	case "tool-input-start":
		if event.Get("providerExecuted").Bool() {
			return nil, nil
		}

		return p.openToolCall(event), nil
	case "tool-input-delta":
		return p.toolArgumentsDelta(event), nil
	case "tool-call":
		if event.Get("providerExecuted").Bool() {
			return nil, nil
		}

		return p.completeToolCall(event), nil
	case "error":
		return nil, parseStreamError(event)
	case "abort":
		return nil, &llm.ResponseError{
			StatusCode: http.StatusBadGateway,
			Detail: llm.ErrorDetail{
				Message: "Command Code aborted the generation",
				Type:    "stream_aborted",
			},
		}
	case "finish":
		p.finished = true

		finishReason := p.finishReason(event.Get("finishReason").String(), event.Get("rawFinishReason").String())
		resp := p.chunk(&llm.Message{})
		resp.Choices[0].FinishReason = &finishReason
		resp.Usage = convertUsage(event.Get("totalUsage"))

		return resp, nil
	default:
		// start, start-step, finish-step, text-start/end, reasoning-start/end,
		// tool-input-end, tool-result and provider-metadata carry nothing the
		// unified response needs.
		return nil, nil
	}
}

func (p *eventProcessor) chunk(delta *llm.Message) *llm.Response {
	if !p.roleSent {
		delta.Role = "assistant"
		p.roleSent = true
	}

	return &llm.Response{
		ID:      p.id,
		Object:  "chat.completion.chunk",
		Created: p.created,
		Model:   p.model,
		Choices: []llm.Choice{{Index: 0, Delta: delta}},
	}
}

func (p *eventProcessor) newToolIndex() int {
	index := p.toolIndex
	p.toolIndex++
	p.sawToolCall = true

	return index
}

// openToolCall emits the tool call header; arguments follow as deltas.
func (p *eventProcessor) openToolCall(event gjson.Result) *llm.Response {
	id := toolCallID(event.Get("id").String())
	call := &openToolCall{index: p.newToolIndex()}
	p.toolCalls[id] = call

	return p.chunk(&llm.Message{ToolCalls: []llm.ToolCall{{
		Index:    call.index,
		ID:       id,
		Type:     "function",
		Function: llm.FunctionCall{Name: event.Get("toolName").String()},
	}}})
}

func (p *eventProcessor) toolArgumentsDelta(event gjson.Result) *llm.Response {
	id := event.Get("id").String()

	call, ok := p.toolCalls[id]
	delta := event.Get("delta").String()
	if !ok || delta == "" {
		return nil
	}

	call.hasArguments = true

	return p.chunk(&llm.Message{ToolCalls: []llm.ToolCall{{
		Index:    call.index,
		ID:       id,
		Type:     "function",
		Function: llm.FunctionCall{Arguments: delta},
	}}})
}

// completeToolCall handles the authoritative tool-call part. Calls whose
// arguments were already streamed are complete; otherwise the full input is
// sent, together with the header when no tool-input-start preceded it.
func (p *eventProcessor) completeToolCall(event gjson.Result) *llm.Response {
	id := toolCallID(event.Get("toolCallId").String())

	call, ok := p.toolCalls[id]
	if ok && call.hasArguments {
		return nil
	}

	toolCall := llm.ToolCall{
		ID:       id,
		Type:     "function",
		Function: llm.FunctionCall{Arguments: toolCallArguments(event)},
	}

	if ok {
		toolCall.Index = call.index
		call.hasArguments = true
	} else {
		toolCall.Index = p.newToolIndex()
		toolCall.Function.Name = event.Get("toolName").String()
		p.toolCalls[id] = &openToolCall{index: toolCall.Index, hasArguments: true}
	}

	return p.chunk(&llm.Message{ToolCalls: []llm.ToolCall{toolCall}})
}

func toolCallID(id string) string {
	if id != "" {
		return id
	}

	return "call_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}

// toolCallArguments normalizes the tool input to a JSON object, following the
// CLI's coercion: objects pass through, a single-object array is unwrapped, and
// a string is parsed as JSON. Anything else becomes an empty object.
func toolCallArguments(event gjson.Result) string {
	input := event.Get("input")
	if !input.Exists() {
		input = event.Get("args")
	}

	switch {
	case input.IsObject():
		return input.Raw
	case input.IsArray():
		if items := input.Array(); len(items) == 1 && items[0].IsObject() {
			return items[0].Raw
		}
	case input.Type == gjson.String:
		if text := input.String(); gjson.Valid(text) && gjson.Parse(text).IsObject() {
			return text
		}
	}

	return "{}"
}

func (p *eventProcessor) finishReason(finishReason, rawFinishReason string) string {
	if strings.EqualFold(rawFinishReason, "refusal") {
		return "content_filter"
	}

	reason := "stop"

	switch finishReason {
	case "tool-calls":
		reason = "tool_calls"
	case "length":
		reason = "length"
	case "content-filter":
		reason = "content_filter"
	case "stop":
	default:
		switch strings.ToLower(rawFinishReason) {
		case "tool_use", "tool_calls", "tool-calls":
			reason = "tool_calls"
		case "max_tokens", "length":
			reason = "length"
		}
	}

	// Provider-executed tool calls are not forwarded, so a tool-calls finish
	// without any forwarded call is a normal stop for the client.
	if reason == "tool_calls" && !p.sawToolCall {
		return "stop"
	}

	return reason
}

// convertUsage maps AI SDK totalUsage. inputTokens already includes cache reads
// and writes; the details only break it down.
func convertUsage(totalUsage gjson.Result) *llm.Usage {
	if !totalUsage.Exists() {
		return nil
	}

	inputTokens := totalUsage.Get("inputTokens").Int()
	outputTokens := totalUsage.Get("outputTokens").Int()

	cacheRead := totalUsage.Get("inputTokenDetails.cacheReadTokens").Int()
	if cacheRead == 0 {
		cacheRead = totalUsage.Get("cachedInputTokens").Int()
	}

	cacheWrite := totalUsage.Get("inputTokenDetails.cacheWriteTokens").Int()

	usage := &llm.Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
		TotalTokens:      inputTokens + outputTokens,
	}

	if cacheRead > 0 || cacheWrite > 0 {
		usage.PromptTokensDetails = &llm.PromptTokensDetails{
			CachedTokens:      cacheRead,
			WriteCachedTokens: cacheWrite,
		}
	}

	reasoningTokens := totalUsage.Get("outputTokenDetails.reasoningTokens").Int()
	if reasoningTokens == 0 {
		reasoningTokens = totalUsage.Get("reasoningTokens").Int()
	}

	if reasoningTokens > 0 {
		usage.CompletionTokensDetails = &llm.CompletionTokensDetails{ReasoningTokens: reasoningTokens}
	}

	return usage
}

// parseStreamError reads an error part. The error is either a string or an
// object such as {"type":"server_error","message":"...","statusCode":400};
// the message may also embed an upstream document like `429 {"error":{...}}`.
func parseStreamError(event gjson.Result) *llm.ResponseError {
	errValue := event.Get("error")

	var statusCode int

	detail := llm.ErrorDetail{Type: "stream_error"}

	switch {
	case errValue.Type == gjson.String:
		detail.Message = errValue.String()
	case errValue.IsObject():
		detail.Message = errValue.Get("message").String()
		statusCode = int(errValue.Get("statusCode").Int())
		if errType := errValue.Get("type").String(); errType != "" {
			detail.Type = errType
		}
	}

	if embeddedStatus, embedded, ok := parseEmbeddedError(detail.Message); ok {
		detail = embedded
		if embeddedStatus > 0 {
			statusCode = embeddedStatus
		}
	}

	if detail.Message == "" {
		detail.Message = "Command Code stream error"
	}

	if statusCode < http.StatusBadRequest {
		statusCode = http.StatusBadGateway
	}

	return &llm.ResponseError{StatusCode: statusCode, Detail: detail}
}

func parseEmbeddedError(message string) (int, llm.ErrorDetail, bool) {
	start := strings.Index(message, "{")
	if start < 0 || !gjson.Valid(message[start:]) {
		return 0, llm.ErrorDetail{}, false
	}

	document := gjson.Parse(message[start:])
	if document.Get("error.message").Type != gjson.String {
		return 0, llm.ErrorDetail{}, false
	}

	// Only a bare numeric prefix is an HTTP status.
	statusCode, err := strconv.Atoi(strings.TrimSpace(message[:start]))
	if err != nil {
		statusCode = 0
	}

	detail := llm.ErrorDetail{
		Message: document.Get("error.message").String(),
		Type:    document.Get("error.type").String(),
	}
	if detail.Type == "" {
		detail.Type = "stream_error"
	}

	return statusCode, detail, true
}

// outboundStream adapts NDJSON events to unified chunks and terminates with
// llm.DoneResponse once the finish event has been delivered.
type outboundStream struct {
	source    streams.Stream[*httpclient.StreamEvent]
	processor *eventProcessor
	current   *llm.Response
	err       error
	done      bool
}

var _ streams.Stream[*llm.Response] = (*outboundStream)(nil)

func newOutboundStream(source streams.Stream[*httpclient.StreamEvent], model string) *outboundStream {
	return &outboundStream{
		source:    source,
		processor: newEventProcessor(model),
	}
}

func (s *outboundStream) Next() bool {
	if s.err != nil || s.done {
		return false
	}

	// The finish chunk has been delivered; do not read past the terminal event.
	if s.processor.finished {
		s.current = llm.DoneResponse
		s.done = true

		return true
	}

	for s.source.Next() {
		event := s.source.Current()
		if event == nil {
			continue
		}

		resp, err := s.processor.process(event.Data)
		if err != nil {
			s.err = err
			return false
		}

		if resp != nil {
			s.current = resp
			return true
		}
	}

	if err := s.source.Err(); err != nil {
		s.err = err
	} else {
		s.err = llm.ErrStreamIncomplete
	}

	return false
}

func (s *outboundStream) Current() *llm.Response {
	return s.current
}

func (s *outboundStream) Err() error {
	return s.err
}

func (s *outboundStream) Close() error {
	return s.source.Close()
}
