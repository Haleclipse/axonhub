// 职责: 把 /alpha/generate 的 NDJSON 事件折叠成一个完整的统一 llm.Response。
// 关联文件:
//   llm/transformer/commandcodes/stream.go
//   llm/transformer/commandcodes/outbound.go
// 注意事项:
//   聚合结果是统一 llm.Response 的 JSON，非流式执行器把它当作上游响应体，
//   TransformResponse 也按同一格式解析；两者必须保持一致。
//   遇到 error/abort 事件直接返回错误，不返回部分内容，避免把失败的生成记成成功。

package commandcodes

import (
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func aggregateEvents(model string, events []*httpclient.StreamEvent) (*llm.Response, llm.ResponseMeta, error) {
	processor := newEventProcessor(model)

	var (
		content      strings.Builder
		reasoning    strings.Builder
		toolCalls    []*llm.ToolCall
		finishReason *string
		usage        *llm.Usage
	)

	for _, event := range events {
		if event == nil {
			continue
		}

		chunk, err := processor.process(event.Data)
		if err != nil {
			return nil, llm.ResponseMeta{}, err
		}

		if chunk == nil {
			continue
		}

		if chunk.Usage != nil {
			usage = chunk.Usage
		}

		for _, choice := range chunk.Choices {
			if choice.Delta != nil {
				content.WriteString(lo.FromPtr(choice.Delta.Content.Content))
				reasoning.WriteString(lo.FromPtr(choice.Delta.ReasoningContent))
				toolCalls = mergeToolCallDeltas(toolCalls, choice.Delta.ToolCalls)
			}

			if choice.FinishReason != nil {
				finishReason = choice.FinishReason
			}
		}

		if processor.finished {
			break
		}
	}

	message := &llm.Message{
		Role:    "assistant",
		Content: llm.MessageContent{Content: lo.ToPtr(content.String())},
	}
	for _, toolCall := range toolCalls {
		message.ToolCalls = append(message.ToolCalls, *toolCall)
	}

	if reasoning.Len() > 0 {
		message.ReasoningContent = lo.ToPtr(reasoning.String())
	}

	resp := &llm.Response{
		ID:      processor.id,
		Object:  "chat.completion",
		Created: processor.created,
		Model:   model,
		Choices: []llm.Choice{{Index: 0, Message: message, FinishReason: finishReason}},
		Usage:   usage,
	}

	meta := llm.ResponseMeta{
		ID:                 resp.ID,
		Usage:              usage,
		Completed:          processor.finished,
		AccountingEvidence: llm.AccountingEvidenceUnavailable,
	}
	if processor.finished && usage != nil {
		meta.AccountingEvidence = llm.AccountingEvidenceExplicitFinal
	}

	return resp, meta, nil
}

// mergeToolCallDeltas folds streamed tool call deltas by index: the header
// carries the ID and name, later deltas append arguments. Indexes are assigned
// sequentially by eventProcessor, so the slice position equals the index.
func mergeToolCallDeltas(toolCalls []*llm.ToolCall, deltas []llm.ToolCall) []*llm.ToolCall {
	for _, delta := range deltas {
		if delta.Index >= len(toolCalls) {
			toolCalls = append(toolCalls, &llm.ToolCall{Index: delta.Index, Type: "function"})
		}

		toolCall := toolCalls[delta.Index]
		if delta.ID != "" {
			toolCall.ID = delta.ID
		}

		if delta.Function.Name != "" {
			toolCall.Function.Name = delta.Function.Name
		}

		toolCall.Function.Arguments += delta.Function.Arguments
	}

	return toolCalls
}
