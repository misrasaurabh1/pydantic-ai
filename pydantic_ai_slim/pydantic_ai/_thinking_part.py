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
    # Optimize by processing the string in one pass
    tag_len = len(START_THINK_TAG)
    end_tag_len = len(END_THINK_TAG)
    idx = 0
    content_len = len(content)

    while idx < content_len:
        start_index = content.find(START_THINK_TAG, idx)
        if start_index < 0:
            # No more <think>, everything left is plain text
            if idx < content_len:
                parts.append(TextPart(content=content[idx:]))
            break

        if start_index > idx:
            parts.append(TextPart(content=content[idx:start_index]))

        after_think = start_index + tag_len
        end_index = content.find(END_THINK_TAG, after_think)
        if end_index >= 0:
            parts.append(ThinkingPart(content=content[after_think:end_index]))
            idx = end_index + end_tag_len
        else:
            # unmatched <think>; treat the rest as plain text as before
            parts.append(TextPart(content=content[after_think:]))
            break

    return parts
