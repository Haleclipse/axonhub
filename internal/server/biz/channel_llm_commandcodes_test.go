// 职责: 锁定 commandcodes 渠道的接线契约：默认端点、outbound 请求、HTTPS 约束、模型目录地址，以及与原有 commandcode 的隔离。
// 关联文件:
//   internal/server/biz/channel_llm_commandcodes.go
//   internal/server/biz/channel_llm.go
//   internal/server/biz/model_fetcher.go

//nolint:exhaustruct_v5 // Test fixtures intentionally set only fields relevant to each scenario.
package biz

import (
	"context"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/ent/enttest"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestCommandCodesOutbound(t *testing.T) {
	client := enttest.NewEntClient(t, "sqlite3", "file:channel_llm_commandcodes?mode=memory&_fk=0")
	t.Cleanup(func() { client.Close() })

	svc := NewChannelServiceForTest(client)
	ctx := context.Background()

	request := &llm.Request{
		Model:    "z-ai/glm-5.3-flash",
		Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("Hello")}}},
	}

	t.Run("generate endpoint with Bearer and CLI version", func(t *testing.T) {
		ch, err := svc.buildChannelWithOutbounds(&ent.Channel{
			ID:          1,
			Name:        "ccs",
			Type:        channel.TypeCommandcodes,
			BaseURL:     "https://api.commandcode.ai",
			Credentials: objects.ChannelCredentials{APIKey: "sk-test"},
		})
		require.NoError(t, err)

		outbound := ch.Outbounds[llm.APIFormatCommandCodeGenerate.String()]
		require.NotNil(t, outbound)
		require.Equal(t, llm.APIFormatCommandCodeGenerate, outbound.APIFormat())

		httpReq, err := outbound.TransformRequest(ctx, request)
		require.NoError(t, err)
		require.Equal(t, "https://api.commandcode.ai/alpha/generate", httpReq.URL)
		require.Equal(t, httpclient.AuthTypeBearer, httpReq.Auth.Type)
		require.Equal(t, "sk-test", httpReq.Auth.APIKey)
		require.NotEmpty(t, httpReq.Headers.Get("x-command-code-version"))
		require.Equal(t, "z-ai/glm-5.3-flash", gjson.GetBytes(httpReq.Body, "params.model").String())
	})

	t.Run("requires an enabled API key", func(t *testing.T) {
		_, err := svc.buildChannelWithOutbounds(&ent.Channel{
			ID:      2,
			Name:    "ccs-nokey",
			Type:    channel.TypeCommandcodes,
			BaseURL: "https://api.commandcode.ai",
		})
		require.ErrorContains(t, err, "missing api key")
	})

	t.Run("insecure base URL is rejected", func(t *testing.T) {
		_, err := svc.buildChannelWithOutbounds(&ent.Channel{
			ID:          3,
			Name:        "ccs-insecure",
			Type:        channel.TypeCommandcodes,
			BaseURL:     "http://api.commandcode.ai",
			Credentials: objects.ChannelCredentials{APIKey: "sk-test"},
		})
		require.ErrorContains(t, err, "HTTPS")
	})
}

func TestCommandCodesIsolatedFromCommandCode(t *testing.T) {
	require.False(t, isCommandCodeChannelType(channel.TypeCommandcodes))
	require.Equal(t,
		[]objects.ChannelEndpoint{{APIFormat: llm.APIFormatCommandCodeGenerate.String()}},
		defaultEndpointsForChannelType[channel.TypeCommandcodes],
	)
}

// Quota is per account, so commandcodes shares the commandcode quota provider
// and authenticates the billing API with its inference API key.
func TestCommandCodesQuotaProvider(t *testing.T) {
	svc := &ProviderQuotaService{}

	withKey := &ent.Channel{Type: channel.TypeCommandcodes, Credentials: objects.ChannelCredentials{APIKeys: []string{"user_key"}}}
	require.Equal(t, "commandcode", svc.getProviderType(withKey))
	require.True(t, hasCredentialsForProvider(withKey))

	require.False(t, hasCredentialsForProvider(&ent.Channel{Type: channel.TypeCommandcodes}))
	require.False(t, hasCredentialsForProvider(&ent.Channel{
		Type:        channel.TypeCommandcodes,
		Credentials: objects.ChannelCredentials{APIKeys: []string{"  "}},
	}))
}

func TestCommandCodesModelsEndpoint(t *testing.T) {
	fetcher := NewModelFetcher(nil, nil)

	for _, baseURL := range []string{"https://api.commandcode.ai", "https://api.commandcode.ai/"} {
		url, headers := fetcher.prepareModelsEndpoint(channel.TypeCommandcodes, baseURL)
		require.Equal(t, "https://api.commandcode.ai/provider/v1/models", url)
		require.Empty(t, headers.Get("Anthropic-Version"))
	}

	result, err := fetcher.FetchModels(context.Background(), FetchModelsInput{
		ChannelType: channel.TypeCommandcodes.String(),
		BaseURL:     "http://api.commandcode.ai",
		APIKey:      lo.ToPtr("sk-test"),
	})
	require.NoError(t, err)
	require.Empty(t, result.Models)
	require.Contains(t, lo.FromPtr(result.Error), "HTTPS")
}
