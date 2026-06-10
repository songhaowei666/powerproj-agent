"""A2AValidator 单元测试。

使用 unittest.mock 对 httpx、a2a-sdk 进行 mock，无需真实 A2A Server。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from a2a_validator.validator import A2AValidator


@pytest.fixture
def validator() -> A2AValidator:
    return A2AValidator(base_url="http://localhost:9999", timeout=5.0)


@pytest.mark.asyncio
async def test_validate_all_passed(validator: A2AValidator) -> None:
    """全部检查项通过的场景。"""
    mock_card = MagicMock()
    mock_card.name = "Test Agent"
    mock_card.capabilities.streaming = True
    mock_card.skills = []

    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200)

    with (
        patch("a2a_validator.validator.A2ACardResolver") as MockResolver,
        patch("a2a_validator.validator.ClientFactory") as MockFactory,
    ):
        resolver_instance = AsyncMock()
        resolver_instance.get_agent_card.return_value = mock_card
        MockResolver.return_value = resolver_instance

        mock_single_response = MagicMock()
        mock_single_response.HasField.return_value = False
        single_client = AsyncMock()
        single_client.send_message.return_value = AsyncMock(
            __aiter__=lambda s: iter([mock_single_response]).__aiter__
        )()

        mock_stream_response = MagicMock()
        mock_stream_response.HasField.side_effect = lambda name: name == "status_update"
        mock_stream_response.status_update.status.HasField.return_value = True
        mock_stream_response.status_update.status.message = MagicMock()
        stream_client = AsyncMock()
        stream_client.send_message.return_value = AsyncMock(
            __aiter__=lambda s: iter([mock_stream_response]).__aiter__
        )()

        factory_instance = MagicMock()
        factory_instance.create.side_effect = [single_client, stream_client]
        MockFactory.return_value = factory_instance

        results = await validator.validate(test_message="hello")

    assert results["connectivity"]["status"] == "passed"
    assert results["agent_card"]["status"] == "passed"
    assert results["agent_card"]["data"] == mock_card
    assert results["single_message"]["status"] == "passed"
    assert results["streaming"]["status"] == "passed"


@pytest.mark.asyncio
async def test_validate_connectivity_failed(validator: A2AValidator) -> None:
    """连通性检查失败，后续所有项应保持 pending。"""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await validator.validate()

    assert results["connectivity"]["status"] == "failed"
    assert "Connection refused" in results["connectivity"]["detail"]
    assert results["agent_card"]["status"] == "pending"
    assert results["single_message"]["status"] == "pending"
    assert results["streaming"]["status"] == "pending"


@pytest.mark.asyncio
async def test_validate_agent_card_failed(validator: A2AValidator) -> None:
    """Agent Card 解析失败，单消息和流式应保持 pending。"""
    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200)

    with (
        patch("httpx.AsyncClient") as MockClient,
        patch("a2a_validator.validator.A2ACardResolver") as MockResolver,
    ):
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        resolver_instance = AsyncMock()
        resolver_instance.get_agent_card.side_effect = ValueError("Invalid JSON")
        MockResolver.return_value = resolver_instance

        results = await validator.validate()

    assert results["connectivity"]["status"] == "passed"
    assert results["agent_card"]["status"] == "failed"
    assert "Invalid JSON" in results["agent_card"]["detail"]
    assert results["single_message"]["status"] == "pending"
    assert results["streaming"]["status"] == "pending"


@pytest.mark.asyncio
async def test_validate_single_message_failed(validator: A2AValidator) -> None:
    """单消息测试失败，流式仍可继续执行。"""
    mock_card = MagicMock()
    mock_card.name = "Test Agent"
    mock_card.capabilities.streaming = True
    mock_card.skills = []

    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200)

    with (
        patch("httpx.AsyncClient") as MockClient,
        patch("a2a_validator.validator.A2ACardResolver") as MockResolver,
        patch("a2a_validator.validator.ClientFactory") as MockFactory,
    ):
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        resolver_instance = AsyncMock()
        resolver_instance.get_agent_card.return_value = mock_card
        MockResolver.return_value = resolver_instance

        single_client = AsyncMock()
        single_client.send_message.side_effect = httpx.ReadTimeout("Server timeout")

        stream_client = AsyncMock()
        stream_client.send_message.return_value = AsyncMock(
            __aiter__=lambda s: iter([]).__aiter__
        )()

        factory_instance = MagicMock()
        factory_instance.create.side_effect = [single_client, stream_client]
        MockFactory.return_value = factory_instance

        results = await validator.validate()

    assert results["connectivity"]["status"] == "passed"
    assert results["agent_card"]["status"] == "passed"
    assert results["single_message"]["status"] == "failed"
    assert "ReadTimeout" in results["single_message"]["detail"]


@pytest.mark.asyncio
async def test_validate_streaming_skipped(validator: A2AValidator) -> None:
    """Agent Card 声明不支持流式，streaming 项标记为 skipped。"""
    mock_card = MagicMock()
    mock_card.name = "Non-streaming Agent"
    mock_card.capabilities.streaming = False
    mock_card.skills = []

    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200)

    with (
        patch("httpx.AsyncClient") as MockClient,
        patch("a2a_validator.validator.A2ACardResolver") as MockResolver,
        patch("a2a_validator.validator.ClientFactory") as MockFactory,
    ):
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        resolver_instance = AsyncMock()
        resolver_instance.get_agent_card.return_value = mock_card
        MockResolver.return_value = resolver_instance

        single_client = AsyncMock()
        single_client.send_message.return_value = AsyncMock(
            __aiter__=lambda s: iter([]).__aiter__
        )()

        factory_instance = MagicMock()
        factory_instance.create.return_value = single_client
        MockFactory.return_value = factory_instance

        results = await validator.validate()

    assert results["connectivity"]["status"] == "passed"
    assert results["agent_card"]["status"] == "passed"
    assert results["single_message"]["status"] == "passed"
    assert results["streaming"]["status"] == "skipped"
    assert "不支持流式传输" in results["streaming"]["detail"]


@pytest.mark.asyncio
async def test_validate_streaming_failed(validator: A2AValidator) -> None:
    """流式测试过程中发生异常。"""
    mock_card = MagicMock()
    mock_card.name = "Test Agent"
    mock_card.capabilities.streaming = True
    mock_card.skills = []

    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=200)

    with (
        patch("httpx.AsyncClient") as MockClient,
        patch("a2a_validator.validator.A2ACardResolver") as MockResolver,
        patch("a2a_validator.validator.ClientFactory") as MockFactory,
    ):
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        resolver_instance = AsyncMock()
        resolver_instance.get_agent_card.return_value = mock_card
        MockResolver.return_value = resolver_instance

        single_client = AsyncMock()
        single_client.send_message.return_value = AsyncMock(
            __aiter__=lambda s: iter([]).__aiter__
        )()

        stream_client = AsyncMock()
        stream_client.send_message.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        factory_instance = MagicMock()
        factory_instance.create.side_effect = [single_client, stream_client]
        MockFactory.return_value = factory_instance

        results = await validator.validate()

    assert results["connectivity"]["status"] == "passed"
    assert results["agent_card"]["status"] == "passed"
    assert results["single_message"]["status"] == "passed"
    assert results["streaming"]["status"] == "failed"
    assert "HTTPStatusError" in results["streaming"]["detail"]


@pytest.mark.asyncio
async def test_validate_well_known_non_200(validator: A2AValidator) -> None:
    """Agent Card 端点返回非 200 状态码，connectivity 标记为 failed。"""
    mock_client = AsyncMock()
    mock_client.get.return_value = MagicMock(status_code=404)

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        results = await validator.validate()

    assert results["connectivity"]["status"] == "failed"
    assert "404" in results["connectivity"]["detail"]
