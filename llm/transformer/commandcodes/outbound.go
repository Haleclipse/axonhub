// 职责: Command Code /alpha/generate 的 outbound transformer：组装请求、转换 NDJSON 流与错误，并让非流式请求复用上游流。
// 关联文件:
//   llm/transformer/commandcodes/request.go
//   llm/transformer/commandcodes/stream.go
//   llm/transformer/commandcodes/aggregate.go
//   llm/transformer/commandcodes/decoder.go
//   llm/transformer/commandcodes/thread.go
// 注意事项:
//   上游把 NDJSON 流标为 text/event-stream，按 Content-Type 会走 SSE 解码并丢弃全部事件，
//   请求必须显式携带 NDJSON 解码器。
//   x-command-code-version 不可省略，缺失时上游返回 403 upgrade_required。
//   x-session-id 与请求体 threadId 取同一值，和官方 CLI 一致；客户端自带的 X-Session-Id 会在合并时覆盖它，
//   因此在 FinalizeTransportRequest 中按请求体重新写回。
//   非流式请求也走 DoStream，并在执行器内聚合成统一响应 JSON；原始 NDJSON 不是合法 JSON，
//   不能作为响应体交给流水线落库。
//   Accept 固定为 */*，与官方 CLI 一致，避免上游按 text/event-stream 协商出 SSE。

package commandcodes

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer"
)

const (
	// DefaultBaseURL is the Command Code API origin used by the official CLI.
	DefaultBaseURL = "https://api.commandcode.ai"
	// GenerateEndpointPath is the CLI's native generation route.
	GenerateEndpointPath = "/alpha/generate"
	// CLIVersion is reported in x-command-code-version, as the official CLI does.
	// Channel header overrides can replace it when the upstream requires a newer client.
	CLIVersion = "1.89.2"

	cliVersionHeader = "X-Command-Code-Version"
	sessionIDHeader  = "X-Session-Id"
)

// Config holds the configuration for the Command Code generate transformer.
type Config struct {
	BaseURL string `json:"base_url,omitempty"`
	// EndpointPath overrides GenerateEndpointPath when set. Must start with "/".
	EndpointPath   string              `json:"endpoint_path,omitempty"`
	APIKeyProvider auth.APIKeyProvider `json:"-"`
}

// OutboundTransformer implements transformer.Outbound for /alpha/generate.
type OutboundTransformer struct {
	url            string
	apiKeyProvider auth.APIKeyProvider
	now            func() time.Time
}

var (
	_ transformer.Outbound                  = (*OutboundTransformer)(nil)
	_ transformer.TransportRequestFinalizer = (*OutboundTransformer)(nil)
	_ pipeline.ChannelCustomizedExecutor    = (*OutboundTransformer)(nil)
)

// NewOutboundTransformerWithConfig creates a Command Code generate transformer.
func NewOutboundTransformerWithConfig(config *Config) (transformer.Outbound, error) {
	if config == nil {
		return nil, errors.New("config is required for Command Code generate transformer")
	}

	if strings.TrimSpace(config.BaseURL) == "" {
		return nil, errors.New("base URL is required for Command Code generate transformer")
	}

	if config.APIKeyProvider == nil {
		return nil, errors.New("API key provider is required for Command Code generate transformer")
	}

	baseURL, rawURL := strings.CutSuffix(config.BaseURL, "##")

	return &OutboundTransformer{
		url:            transformer.BuildRequestURL(baseURL, "", GenerateEndpointPath, config.EndpointPath, rawURL),
		apiKeyProvider: config.APIKeyProvider,
		now:            time.Now,
	}, nil
}

func (t *OutboundTransformer) APIFormat() llm.APIFormat {
	return llm.APIFormatCommandCodeGenerate
}

func (t *OutboundTransformer) TransformRequest(ctx context.Context, llmReq *llm.Request) (*httpclient.Request, error) {
	if llmReq == nil {
		return nil, fmt.Errorf("%w: request is nil", transformer.ErrInvalidRequest)
	}

	payload, err := buildGenerateRequest(llmReq, t.now())
	if err != nil {
		return nil, err
	}

	payload.ThreadID = resolveThreadID(ctx, llmReq, &payload.Params)

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Command Code generate request: %w", err)
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/json")
	headers.Set("Accept", "*/*")
	headers.Set(cliVersionHeader, CLIVersion)
	headers.Set(sessionIDHeader, payload.ThreadID)

	return &httpclient.Request{
		Method:  http.MethodPost,
		URL:     t.url,
		Headers: headers,
		Body:    body,
		Auth: &httpclient.AuthConfig{
			Type:   httpclient.AuthTypeBearer,
			APIKey: t.apiKeyProvider.Get(ctx),
		},
		RequestType:   llm.RequestTypeChat.String(),
		APIFormat:     llm.APIFormatCommandCodeGenerate.String(),
		StreamDecoder: newNDJSONDecoder,
	}, nil
}

// FinalizeTransportRequest keeps x-session-id equal to the body threadId after
// inbound client headers have been merged into the request.
func (t *OutboundTransformer) FinalizeTransportRequest(request *httpclient.Request) *httpclient.Request {
	if request == nil {
		return nil
	}

	threadID := gjson.GetBytes(request.Body, "threadId").String()
	if threadID == "" {
		return request
	}

	if request.Headers == nil {
		request.Headers = make(http.Header)
	}

	request.Headers.Set(sessionIDHeader, threadID)

	return request
}

// TransformResponse decodes the unified response produced by generateExecutor.Do.
func (t *OutboundTransformer) TransformResponse(_ context.Context, httpResp *httpclient.Response) (*llm.Response, error) {
	if httpResp == nil {
		return nil, fmt.Errorf("%w: response is nil", transformer.ErrInvalidResponse)
	}

	var resp llm.Response
	if err := json.Unmarshal(httpResp.Body, &resp); err != nil {
		return nil, fmt.Errorf("%w: failed to decode Command Code response: %w", transformer.ErrInvalidResponse, err)
	}

	return &resp, nil
}

func (t *OutboundTransformer) TransformStream(
	_ context.Context,
	req *httpclient.Request,
	stream streams.Stream[*httpclient.StreamEvent],
) (streams.Stream[*llm.Response], error) {
	return newOutboundStream(stream, requestModel(req)), nil
}

func (t *OutboundTransformer) TransformError(_ context.Context, rawErr *httpclient.Error) *llm.ResponseError {
	if rawErr == nil {
		return &llm.ResponseError{
			StatusCode: http.StatusInternalServerError,
			Detail: llm.ErrorDetail{
				Message: http.StatusText(http.StatusInternalServerError),
				Type:    "api_error",
			},
		}
	}

	detail := llm.ErrorDetail{Type: "api_error"}

	if gjson.ValidBytes(rawErr.Body) {
		root := gjson.ParseBytes(rawErr.Body)

		switch errValue := root.Get("error"); {
		case errValue.IsObject():
			detail.Message = errValue.Get("message").String()
			detail.Code = errValue.Get("code").String()
			if errType := errValue.Get("type").String(); errType != "" {
				detail.Type = errType
			}
		case errValue.Type == gjson.String:
			detail.Message = errValue.String()
		}

		if detail.Message == "" {
			detail.Message = root.Get("message").String()
		}
	}

	if detail.Message == "" {
		detail.Message = strings.TrimSpace(string(rawErr.Body))
	}

	if detail.Message == "" {
		detail.Message = http.StatusText(rawErr.StatusCode)
	}

	return &llm.ResponseError{StatusCode: rawErr.StatusCode, Detail: detail}
}

func (t *OutboundTransformer) AggregateStreamChunks(
	_ context.Context,
	req *httpclient.Request,
	chunks []*httpclient.StreamEvent,
) ([]byte, llm.ResponseMeta, error) {
	resp, meta, err := aggregateEvents(requestModel(req), chunks)
	if err != nil {
		return nil, llm.ResponseMeta{}, err
	}

	body, err := json.Marshal(resp)
	if err != nil {
		return nil, llm.ResponseMeta{}, fmt.Errorf("failed to marshal aggregated Command Code response: %w", err)
	}

	return body, meta, nil
}

// CustomizeExecutor makes non-streaming requests consume the upstream stream,
// because /alpha/generate only answers with NDJSON.
func (t *OutboundTransformer) CustomizeExecutor(executor pipeline.Executor) pipeline.Executor {
	return &generateExecutor{inner: executor}
}

type generateExecutor struct {
	inner pipeline.Executor
}

func (e *generateExecutor) DoStream(ctx context.Context, request *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
	return e.inner.DoStream(ctx, request)
}

// Do reads the stream up to the finish event and returns the aggregated
// unified response as a JSON body. A stream that ends before finish is an
// incomplete response, never a successful one.
func (e *generateExecutor) Do(ctx context.Context, request *httpclient.Request) (*httpclient.Response, error) {
	stream, err := e.inner.DoStream(ctx, request)
	if err != nil {
		return nil, err
	}

	defer func() {
		_ = stream.Close()
	}()

	var events []*httpclient.StreamEvent

	for stream.Next() {
		event := stream.Current()
		if event == nil {
			continue
		}

		events = append(events, &httpclient.StreamEvent{Data: bytes.Clone(event.Data)})
		if gjson.GetBytes(event.Data, "type").String() == "finish" {
			break
		}
	}

	if err := stream.Err(); err != nil {
		return nil, err
	}

	resp, meta, err := aggregateEvents(requestModel(request), events)
	if err != nil {
		return nil, err
	}

	if !meta.Completed {
		return nil, llm.ErrStreamIncomplete
	}

	body, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal aggregated Command Code response: %w", err)
	}

	return &httpclient.Response{
		StatusCode: http.StatusOK,
		Headers: httpclient.MergeForwardResponseHeaders(http.Header{
			"Content-Type": []string{"application/json"},
		}, httpclient.GetResponseHeaders(stream)),
		Body:    body,
		Request: request,
	}, nil
}

func requestModel(req *httpclient.Request) string {
	if req == nil {
		return ""
	}

	return gjson.GetBytes(req.Body, "params.model").String()
}
