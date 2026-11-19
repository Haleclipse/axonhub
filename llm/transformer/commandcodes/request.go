// 职责: 把统一 llm.Request 映射为 Command Code /alpha/generate 请求体（AI SDK model messages）。
// 关联文件:
//   llm/transformer/commandcodes/outbound.go
//   llm/model.go
//   llm/tools.go
// 注意事项:
//   线格式对齐官方 CLI 的 createModelClient/toWireMessages：图片是 data URL 加 mimeType，
//   工具结果只保留文本，assistant 推理只回传文本。params 只发送 CLI 会发送的字段；
//   上游按 schema 校验字段类型（如 system 只能是字符串或块数组，null 返回 400），content 不支持 file 部件。
//   上游始终以流式 NDJSON 响应，因此 params.stream 恒为 true；stream=false 会被上游当作代理请求拒绝。
//   params.system 缺失或为空字符串时，上游会注入约 7k token 的 CLI 智能体提示，改变模型行为并增加计费；
//   客户端没有 system 时必须发送非空的中性提示。仅含空白的 system 会被 Claude 拒绝，不能用空白占位。
//   只有开头连续的 system/developer 消息进入 params.system。对话中途的 system 消息（Claude Code 会在工具执行、
//   hook、提醒时追加）必须原地转为 user 消息，否则 system 前缀逐轮变化，缓存只能命中静态部分；
//   落在 tool 调用与结果之间时延后到结果之后，避免上游因工具配对被打断而拒绝请求。
//   config 对应 CLI 的 buildServerConfig；网关没有工作区，取值与 CLI 内存运行时的缺省（cwd "/"、platform "linux"）一致。

package commandcodes

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer"
)

const (
	// defaultMaxTokens matches the official CLI default output budget.
	defaultMaxTokens int64 = 64000

	permissionModeStandard = "standard"

	// defaultSystemPrompt keeps the upstream from substituting its CLI agent prompt
	// when the client sends no system message.
	defaultSystemPrompt = "You are a helpful assistant."
)

// defaultInputSchema is sent for function tools that declare no parameters.
var defaultInputSchema = json.RawMessage(`{"type":"object","properties":{}}`)

// supportedReasoningEfforts are the reasoning_effort values the official CLI sends.
var supportedReasoningEfforts = map[string]struct{}{
	"low":    {},
	"medium": {},
	"high":   {},
	"xhigh":  {},
	"max":    {},
}

type generateRequest struct {
	Config generateConfig `json:"config"`
	// Memory, Taste and Skills are CLI-side features; the gateway always sends null.
	Memory         any    `json:"memory"`
	Taste          any    `json:"taste"`
	Skills         any    `json:"skills"`
	PermissionMode string `json:"permissionMode"`
	// ThreadID groups a conversation's turns upstream; see thread.go.
	ThreadID string         `json:"threadId,omitempty"`
	Params   generateParams `json:"params"`
}

// generateConfig mirrors the CLI's buildServerConfig workspace snapshot. The
// gateway has no workspace, so it reports an empty non-git directory.
type generateConfig struct {
	WorkingDir    string   `json:"workingDir"`
	Date          string   `json:"date"`
	Environment   string   `json:"environment"`
	Structure     []string `json:"structure"`
	IsGitRepo     bool     `json:"isGitRepo"`
	CurrentBranch string   `json:"currentBranch"`
	MainBranch    string   `json:"mainBranch"`
	GitStatus     string   `json:"gitStatus"`
	RecentCommits []string `json:"recentCommits"`
}

type generateParams struct {
	Model string `json:"model"`
	// System is either a string or []wireSystemBlock when cache control is requested.
	System          any           `json:"system"`
	Messages        []wireMessage `json:"messages"`
	Tools           []wireTool    `json:"tools"`
	MaxTokens       int64         `json:"max_tokens"`
	Stream          bool          `json:"stream"`
	Temperature     *float64      `json:"temperature,omitempty"`
	ReasoningEffort string        `json:"reasoning_effort,omitempty"`
}

type wireSystemBlock struct {
	Type         string            `json:"type"`
	Text         string            `json:"text"`
	CacheControl *wireCacheControl `json:"cache_control,omitempty"`
}

type wireCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type wireMessage struct {
	Role    string     `json:"role"`
	Content []wirePart `json:"content"`
}

// wirePart is the union of the AI SDK content parts the CLI sends:
// text, reasoning, image, tool-call and tool-result.
type wirePart struct {
	Type       string          `json:"type"`
	Text       string          `json:"text,omitempty"`
	Image      string          `json:"image,omitempty"`
	MimeType   string          `json:"mimeType,omitempty"`
	ToolCallID string          `json:"toolCallId,omitempty"`
	ToolName   string          `json:"toolName,omitempty"`
	Input      json.RawMessage `json:"input,omitempty"`
	Output     *wireToolOutput `json:"output,omitempty"`
}

type wireToolOutput struct {
	Type  string `json:"type"`
	Value string `json:"value"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func buildGenerateRequest(req *llm.Request, now time.Time) (*generateRequest, error) {
	if req.RequestType != "" && req.RequestType != llm.RequestTypeChat {
		return nil, fmt.Errorf("%w: Command Code generate does not support request type %q", transformer.ErrInvalidRequest, req.RequestType)
	}

	if req.Model == "" {
		return nil, fmt.Errorf("%w: model is required", transformer.ErrInvalidRequest)
	}

	system, messages, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	if len(messages) == 0 {
		return nil, fmt.Errorf("%w: at least one non-system message is required", transformer.ErrInvalidRequest)
	}

	maxTokens := lo.FromPtrOr(req.MaxCompletionTokens, lo.FromPtr(req.MaxTokens))
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	return &generateRequest{
		Config: generateConfig{
			WorkingDir:    "/",
			Date:          now.UTC().Format(time.DateOnly),
			Environment:   "linux",
			Structure:     []string{},
			RecentCommits: []string{},
		},
		PermissionMode: permissionModeStandard,
		Params: generateParams{
			Model:           req.Model,
			System:          system,
			Messages:        messages,
			Tools:           convertTools(req.Tools),
			MaxTokens:       maxTokens,
			Stream:          true,
			Temperature:     req.Temperature,
			ReasoningEffort: convertReasoningEffort(req.ReasoningEffort),
		},
	}, nil
}

// convertMessages hoists the leading system/developer block into params.system
// and maps the rest to AI SDK model messages. Consecutive tool results share
// one tool message, matching how the CLI groups the tool_result blocks of a
// user turn.
func convertMessages(msgs []llm.Message) (any, []wireMessage, error) {
	var (
		systemBlocks    []wireSystemBlock
		hasCacheControl bool
		leading         = true
		converter       = messageConverter{
			messages:         make([]wireMessage, 0, len(msgs)),
			toolNames:        make(map[string]string),
			pendingToolCalls: make(map[string]struct{}),
		}
	)

	for _, msg := range msgs {
		isSystem := msg.Role == "system" || msg.Role == "developer"
		if isSystem && leading {
			text := joinText(msg.Content, "\n")
			if strings.TrimSpace(text) == "" {
				continue
			}

			block := wireSystemBlock{Type: "text", Text: text, CacheControl: systemCacheControl(msg)}
			hasCacheControl = hasCacheControl || block.CacheControl != nil
			systemBlocks = append(systemBlocks, block)

			continue
		}

		leading = false

		switch {
		case isSystem:
			converter.addLateSystem(msg)
		case msg.Role == "user":
			parts, err := convertUserContent(msg.Content)
			if err != nil {
				return nil, nil, err
			}

			if len(parts) > 0 {
				converter.append(wireMessage{Role: "user", Content: parts})
			}
		case msg.Role == "assistant":
			converter.addAssistant(msg)
		case msg.Role == "tool":
			converter.addToolResult(msg)
		default:
			return nil, nil, fmt.Errorf("%w: unsupported message role %q", transformer.ErrInvalidRequest, msg.Role)
		}
	}

	converter.flushDeferred()

	return convertSystem(systemBlocks, hasCacheControl), converter.messages, nil
}

// messageConverter keeps the state needed to place late system messages
// in the conversation without splitting a tool call from its results.
type messageConverter struct {
	messages  []wireMessage
	toolNames map[string]string
	// pendingToolCalls are the latest assistant tool calls still waiting for results.
	pendingToolCalls map[string]struct{}
	// deferred holds late system text that arrived while tool results were pending.
	deferred []wirePart
}

// append emits a user or assistant message after any deferred system text.
func (c *messageConverter) append(message wireMessage) {
	c.flushDeferred()
	c.messages = append(c.messages, message)
}

// addLateSystem keeps a mid-conversation system message in place as user text.
// Hoisting it into params.system would change the prompt prefix every time
// the client adds one, leaving only the static system prompt cacheable.
func (c *messageConverter) addLateSystem(msg llm.Message) {
	text := joinText(msg.Content, "\n")
	if strings.TrimSpace(text) == "" {
		return
	}

	part := wirePart{Type: "text", Text: text}
	if len(c.pendingToolCalls) > 0 {
		// Providers reject anything between a tool call and its results.
		c.deferred = append(c.deferred, part)
		return
	}

	c.messages = append(c.messages, wireMessage{Role: "user", Content: []wirePart{part}})
}

func (c *messageConverter) addAssistant(msg llm.Message) {
	parts := convertAssistantMessage(msg, c.toolNames)
	if len(parts) > 0 {
		c.append(wireMessage{Role: "assistant", Content: parts})
	}

	c.pendingToolCalls = make(map[string]struct{}, len(msg.ToolCalls))
	for _, toolCall := range msg.ToolCalls {
		c.pendingToolCalls[toolCall.ID] = struct{}{}
	}
}

func (c *messageConverter) addToolResult(msg llm.Message) {
	part := convertToolResult(msg, c.toolNames)
	if last := len(c.messages) - 1; last >= 0 && c.messages[last].Role == "tool" {
		c.messages[last].Content = append(c.messages[last].Content, part)
	} else {
		c.messages = append(c.messages, wireMessage{Role: "tool", Content: []wirePart{part}})
	}

	delete(c.pendingToolCalls, part.ToolCallID)
	if len(c.pendingToolCalls) == 0 {
		c.flushDeferred()
	}
}

func (c *messageConverter) flushDeferred() {
	for _, part := range c.deferred {
		c.messages = append(c.messages, wireMessage{Role: "user", Content: []wirePart{part}})
	}

	c.deferred = nil
}

func convertSystem(blocks []wireSystemBlock, hasCacheControl bool) any {
	if len(blocks) == 0 {
		return defaultSystemPrompt
	}

	if hasCacheControl {
		return blocks
	}

	return strings.Join(lo.Map(blocks, func(block wireSystemBlock, _ int) string { return block.Text }), "\n\n")
}

func systemCacheControl(msg llm.Message) *wireCacheControl {
	cacheControl := msg.CacheControl
	for _, part := range msg.Content.MultipleContent {
		if cacheControl == nil && part.CacheControl != nil {
			cacheControl = part.CacheControl
		}
	}

	if cacheControl == nil {
		return nil
	}

	return &wireCacheControl{Type: "ephemeral", TTL: cacheControl.TTL}
}

func convertUserContent(content llm.MessageContent) ([]wirePart, error) {
	if content.Content != nil {
		if *content.Content == "" {
			return nil, nil
		}

		return []wirePart{{Type: "text", Text: *content.Content}}, nil
	}

	parts := make([]wirePart, 0, len(content.MultipleContent))
	for _, part := range content.MultipleContent {
		switch part.Type {
		case "text":
			if text := lo.FromPtr(part.Text); text != "" {
				parts = append(parts, wirePart{Type: "text", Text: text})
			}
		case "image_url":
			if part.ImageURL == nil || strings.TrimSpace(part.ImageURL.URL) == "" {
				continue
			}

			image := strings.TrimSpace(part.ImageURL.URL)
			parts = append(parts, wirePart{Type: "image", Image: image, MimeType: imageMimeType(image, part.ImageURL.MIMEType)})
		default:
			return nil, fmt.Errorf("%w: Command Code generate does not support %q content", transformer.ErrInvalidRequest, part.Type)
		}
	}

	return parts, nil
}

// imageMimeType prefers the media type embedded in a data URL.
func imageMimeType(image, declared string) string {
	if rest, ok := strings.CutPrefix(image, "data:"); ok {
		if mediaType, _, found := strings.Cut(rest, ";"); found && mediaType != "" {
			return mediaType
		}
	}

	return declared
}

func convertAssistantMessage(msg llm.Message, toolNames map[string]string) []wirePart {
	parts := make([]wirePart, 0, 2+len(msg.ToolCalls))

	reasoning := lo.FromPtr(msg.ReasoningContent)
	if reasoning == "" {
		reasoning = lo.FromPtr(msg.Reasoning)
	}

	if reasoning != "" {
		parts = append(parts, wirePart{Type: "reasoning", Text: reasoning})
	}

	if text := joinText(msg.Content, ""); text != "" {
		parts = append(parts, wirePart{Type: "text", Text: text})
	}

	for _, toolCall := range msg.ToolCalls {
		toolNames[toolCall.ID] = toolCall.Function.Name
		parts = append(parts, wirePart{
			Type:       "tool-call",
			ToolCallID: toolCall.ID,
			ToolName:   toolCall.Function.Name,
			Input:      toolCallInput(toolCall.Function.Arguments),
		})
	}

	return parts
}

// toolCallInput replays historical arguments as a JSON object. Arguments that
// are not a JSON object cannot become a valid AI SDK tool-call input, so they
// are replaced with an empty object rather than failing the whole turn.
func toolCallInput(arguments string) json.RawMessage {
	arguments = strings.TrimSpace(arguments)
	if !json.Valid([]byte(arguments)) || !strings.HasPrefix(arguments, "{") {
		return json.RawMessage(`{}`)
	}

	return json.RawMessage(arguments)
}

func convertToolResult(msg llm.Message, toolNames map[string]string) wirePart {
	toolCallID := lo.FromPtr(msg.ToolCallID)

	toolName := toolNames[toolCallID]
	if toolName == "" {
		toolName = lo.FromPtr(msg.ToolCallName)
	}

	if toolName == "" {
		toolName = "unknown"
	}

	outputType := "text"
	if lo.FromPtr(msg.ToolCallIsError) {
		outputType = "error-text"
	}

	return wirePart{
		Type:       "tool-result",
		ToolCallID: toolCallID,
		ToolName:   toolName,
		Output:     &wireToolOutput{Type: outputType, Value: joinText(msg.Content, "\n")},
	}
}

func convertTools(tools []llm.Tool) []wireTool {
	result := make([]wireTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "" && tool.Type != llm.ToolTypeFunction {
			continue
		}

		schema := tool.Function.Parameters
		if len(schema) == 0 {
			schema = tool.Function.ParametersJsonSchema
		}

		if len(schema) == 0 || string(schema) == "null" {
			schema = defaultInputSchema
		}

		result = append(result, wireTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: schema,
		})
	}

	return result
}

func convertReasoningEffort(effort string) string {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort == "minimal" {
		return "low"
	}

	if _, ok := supportedReasoningEfforts[effort]; ok {
		return effort
	}

	return ""
}

func joinText(content llm.MessageContent, sep string) string {
	if content.Content != nil {
		return *content.Content
	}

	texts := make([]string, 0, len(content.MultipleContent))
	for _, part := range content.MultipleContent {
		if part.Type == "text" && lo.FromPtr(part.Text) != "" {
			texts = append(texts, *part.Text)
		}
	}

	return strings.Join(texts, sep)
}
