// 职责: 为 commandcodes 渠道（Command Code CLI 的 /alpha/generate）构建 outbound，并提供该类型专用的 base URL 校验与模型目录地址。
// 关联文件:
//   internal/server/biz/channel_llm.go
//   internal/server/biz/channel.go
//   internal/server/biz/model_fetcher.go
//   llm/transformer/commandcodes/outbound.go
// 注意事项:
//   commandcodes 与 commandcode / commandcode_anthropic 的推理链路刻意隔离，不进入 isCommandCodeChannelType，
//   不参与其 Cookie 设置与模型过滤；需要的函数复制一份独立维护，不复用原有实现。
//   配额例外：同一账号同一套计费接口，归入 commandcode 配额 provider，只用 API Key 采集。
//   base URL 是 API 源站（https://api.commandcode.ai）：推理走 /alpha/generate，模型目录走 /provider/v1/models。

package biz

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/commandcodes"
)

const commandCodesModelsPath = "/provider/v1/models"

// validateCommandCodesBaseURL keeps the account API key on HTTPS.
func validateCommandCodesBaseURL(rawURL string) error {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
		return fmt.Errorf("Command Code (CLI) base URL must use HTTPS")
	}

	return nil
}

func commandCodesModelsURL(baseURL string) string {
	return strings.TrimSuffix(baseURL, "/") + commandCodesModelsPath
}

func buildCommandCodesOutbound(ch *Channel, baseURL string) (transformer.Outbound, error) {
	apiKeyProvider, err := getAPIKeyProvider(ch)
	if err != nil {
		return nil, err
	}

	return commandcodes.NewOutboundTransformerWithConfig(&commandcodes.Config{
		BaseURL:        baseURL,
		APIKeyProvider: apiKeyProvider,
	})
}
