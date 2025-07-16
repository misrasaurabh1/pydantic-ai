import base64
from collections.abc import Sequence

from mcp import types as mcp_types

from . import exceptions, messages

try:
    from mcp import types as mcp_types
except ImportError as _import_error:
    raise ImportError(
        'Please install the `mcp` package to use the MCP server, '
        'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
    ) from _import_error


def map_from_mcp_params(params: mcp_types.CreateMessageRequestParams) -> list[messages.ModelMessage]:
    """Convert from MCP create message request parameters to pydantic-ai messages."""
    pai_messages: list[messages.ModelMessage] = []
    request_parts: list[messages.ModelRequestPart] = []
    if params.systemPrompt:
        request_parts.append(messages.SystemPromptPart(content=params.systemPrompt))
    response_parts: list[messages.ModelResponsePart] = []
    for msg in params.messages:
        content = msg.content
        if msg.role == 'user':
            # if there are any response parts, add a response message wrapping them
            if response_parts:
                pai_messages.append(messages.ModelResponse(parts=response_parts))
                response_parts = []

            # TODO(Marcelo): We can reuse the `_map_tool_result_part` from the mcp module here.
            if isinstance(content, mcp_types.TextContent):
                user_part_content: str | Sequence[messages.UserContent] = content.text
            else:
                # image content
                user_part_content = [
                    messages.BinaryContent(data=base64.b64decode(content.data), media_type=content.mimeType)
                ]

            request_parts.append(messages.UserPromptPart(content=user_part_content))
        else:
            # role is assistant
            # if there are any request parts, add a request message wrapping them
            if request_parts:
                pai_messages.append(messages.ModelRequest(parts=request_parts))
                request_parts = []

            response_parts.append(map_from_sampling_content(content))

    if response_parts:
        pai_messages.append(messages.ModelResponse(parts=response_parts))
    if request_parts:
        pai_messages.append(messages.ModelRequest(parts=request_parts))
    return pai_messages


def map_from_pai_messages(pai_messages: list[messages.ModelMessage]) -> tuple[str, list[mcp_types.SamplingMessage]]:
    """Convert from pydantic-ai messages to MCP sampling messages.

    Returns:
        A tuple containing the system prompt and a list of sampling messages.
    """
    sampling_msgs: list[mcp_types.SamplingMessage] = []
    system_prompt: list[str] = []
    append_sampling = sampling_msgs.append
    extend_prompt = system_prompt.extend
    for pai_message in pai_messages:
        if isinstance(pai_message, messages.ModelRequest):
            instructions = pai_message.instructions
            if instructions is not None:
                system_prompt.append(instructions)
            parts = pai_message.parts
            for part in parts:
                # Sequence optimized: check type only once
                if isinstance(part, messages.SystemPromptPart):
                    system_prompt.append(part.content)
                elif isinstance(part, messages.UserPromptPart):
                    content = part.content
                    # Fast-path: string user prompt
                    if isinstance(content, str):
                        append_sampling(mcp_types.SamplingMessage(
                            role='user',
                            content=mcp_types.TextContent(type='text', text=content)
                        ))
                    else:
                        for chunk in content:
                            if isinstance(chunk, str):
                                append_sampling(mcp_types.SamplingMessage(
                                    role='user',
                                    content=mcp_types.TextContent(type='text', text=chunk)
                                ))
                            elif isinstance(chunk, messages.BinaryContent) and chunk.is_image:
                                append_sampling(mcp_types.SamplingMessage(
                                    role='user',
                                    content=mcp_types.ImageContent(
                                        type='image',
                                        data=base64.b64decode(chunk.data).decode(),
                                        mimeType=chunk.media_type,
                                    )
                                ))
                            # TODO(Marcelo): Add support for audio content.
                            else:
                                raise NotImplementedError(f'Unsupported content type: {type(chunk)}')
        else:
            # Hotpath: only check type of response once in callee.
            append_sampling(mcp_types.SamplingMessage(
                role='assistant',
                content=map_from_model_response(pai_message)
            ))
    return ''.join(system_prompt), sampling_msgs


def map_from_model_response(model_response: messages.ModelResponse) -> mcp_types.TextContent:
    """Convert from a model response to MCP text content."""
    # Optimize: List comp for max speed, raising as soon as non-TextPart found.
    parts = model_response.parts
    texts = []
    for part in parts:
        if isinstance(part, messages.TextPart):
            texts.append(part.content)
        else:
            raise exceptions.UnexpectedModelBehavior(
                f'Unexpected part type: {type(part).__name__}, expected TextPart'
            )
    return mcp_types.TextContent(type='text', text=''.join(texts))


def map_from_sampling_content(
    content: mcp_types.TextContent | mcp_types.ImageContent | mcp_types.AudioContent,
) -> messages.TextPart:
    """Convert from sampling content to a pydantic-ai text part."""
    if isinstance(content, mcp_types.TextContent):  # pragma: no branch
        return messages.TextPart(content=content.text)
    else:
        raise NotImplementedError('Image and Audio responses in sampling are not yet supported')
