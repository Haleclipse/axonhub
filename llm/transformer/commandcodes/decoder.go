// 职责: 把 /alpha/generate 的 NDJSON 响应体逐行解码为流事件，一行对应一个事件。
// 关联文件:
//   llm/transformer/commandcodes/outbound.go
//   llm/httpclient/client.go
// 注意事项:
//   复制自 llm/transformer/ollama/decoder.go，刻意不注册全局 Content-Type：
//   上游把 NDJSON 标为 text/event-stream，只能通过请求级 StreamDecoder 使用，注册会劫持所有 SSE 渠道。

package commandcodes

import (
	"bufio"
	"context"
	"errors"
	"io"
	"log/slog"

	"github.com/looplj/axonhub/llm/httpclient"
)

// newNDJSONDecoder creates a decoder that yields one event per NDJSON line.
func newNDJSONDecoder(ctx context.Context, rc io.ReadCloser) httpclient.StreamDecoder {
	return &ndjsonDecoder{
		ctx:    ctx,
		reader: bufio.NewReader(rc),
		body:   rc,
	}
}

type ndjsonDecoder struct {
	ctx     context.Context
	reader  *bufio.Reader
	body    io.ReadCloser
	current *httpclient.StreamEvent
	err     error
	closed  bool
}

var _ httpclient.StreamDecoder = (*ndjsonDecoder)(nil)

// Next advances to the next line.
func (d *ndjsonDecoder) Next() bool {
	if d.err != nil || d.closed {
		return false
	}

	select {
	case <-d.ctx.Done():
		d.err = d.ctx.Err()
		_ = d.Close()

		return false
	default:
	}

	line, err := d.reader.ReadBytes('\n')
	if err != nil {
		if errors.Is(err, io.EOF) {
			// The last line may arrive without a trailing newline.
			if len(line) > 0 {
				d.current = &httpclient.StreamEvent{Data: line}

				return true
			}

			slog.DebugContext(d.ctx, "Command Code NDJSON stream closed")
		} else {
			d.err = err
			slog.ErrorContext(d.ctx, "Command Code NDJSON read error", slog.Any("error", err))
		}

		_ = d.Close()

		return false
	}

	if len(line) > 0 && line[len(line)-1] == '\n' {
		line = line[:len(line)-1]
	}

	if len(line) > 0 && line[len(line)-1] == '\r' {
		line = line[:len(line)-1]
	}

	d.current = &httpclient.StreamEvent{Data: line}

	return true
}

func (d *ndjsonDecoder) Current() *httpclient.StreamEvent {
	return d.current
}

func (d *ndjsonDecoder) Err() error {
	return d.err
}

func (d *ndjsonDecoder) Close() error {
	if d.closed {
		return nil
	}

	d.closed = true
	if d.body != nil {
		return d.body.Close()
	}

	return nil
}
