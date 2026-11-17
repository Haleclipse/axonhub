// 职责: 验证 Command Code NDJSON 解码器的分行行为。
// 关联文件:
//   llm/transformer/commandcodes/decoder.go

package commandcodes

import (
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNDJSONDecoder_SplitsLinesAndKeepsUnterminatedTail(t *testing.T) {
	body := io.NopCloser(strings.NewReader("{\"a\":1}\r\n{\"b\":2}\n{\"c\":3}"))
	decoder := newNDJSONDecoder(t.Context(), body)

	var lines []string
	for decoder.Next() {
		lines = append(lines, string(decoder.Current().Data))
	}

	require.NoError(t, decoder.Err())
	require.Equal(t, []string{`{"a":1}`, `{"b":2}`, `{"c":3}`}, lines)
}
