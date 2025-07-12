from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = '<think>'
END_THINK_TAG = '</think>'


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []
    i = 0
    n = len(content)
    stt = START_THINK_TAG
    ett = END_THINK_TAG
    stt_len = len(stt)
    ett_len = len(ett)

    while i < n:
        start_idx = content.find(stt, i)
        if start_idx < 0:
            # No more <think> tags, rest is plain text
            if i < n:
                parts.append(TextPart(content=content[i:]))
            break
        # Add text before <think> tag if any
        if start_idx > i:
            parts.append(TextPart(content=content[i:start_idx]))
        after_start = start_idx + stt_len
        end_idx = content.find(ett, after_start)
        if end_idx < 0:
            # No end tag found, treat rest as plain text (lose <think>)
            parts.append(TextPart(content=content[after_start:]))
            break
        # Add thinking content
        parts.append(ThinkingPart(content=content[after_start:end_idx]))
        i = end_idx + ett_len
    return parts
