from typing import Any, Protocol

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool, Tool as BaseTool


class LangChainTool(Protocol):
    # args are like
    # {'dir_path': {'default': '.', 'description': 'Subdirectory to search in.', 'title': 'Dir Path', 'type': 'string'},
    #  'pattern': {'description': 'Unix shell regex, where * matches everything.', 'title': 'Pattern', 'type': 'string'}}
    @property
    def args(self) -> dict[str, JsonSchemaValue]: ...

    def get_input_jsonschema(self) -> JsonSchemaValue: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, *args: Any, **kwargs: Any) -> str: ...


__all__ = ('tool_from_langchain',)


def tool_from_langchain(langchain_tool: LangChainTool) -> Tool:
    """Creates a Pydantic tool proxy from a LangChain tool.

    Args:
        langchain_tool: The LangChain tool to wrap.

    Returns:
        A Pydantic tool that corresponds to the LangChain tool.
    """
    # Local variable cache and minimize attribute lookups
    args = langchain_tool.args
    name = langchain_tool.name
    desc = langchain_tool.description

    # Single pass: required list and defaults
    required, defaults = [], {}
    for n, d in args.items():
        if 'default' not in d:
            required.append(n)
        else:
            defaults[n] = d['default']
    required.sort()

    schema = langchain_tool.get_input_jsonschema()
    # Set keys only if needed (minimize mutation)
    if 'additionalProperties' not in schema:
        schema['additionalProperties'] = False
    if required:
        schema['required'] = required

    # No .copy() needed, as we don't mutate `args`
    def proxy(*args: Any, **kwargs: Any) -> str:
        assert not args, 'This should always be called with kwargs'
        # Avoid temporary dict: update a local copy instead of new dict merge
        merged = defaults.copy()
        merged.update(kwargs)
        return langchain_tool.run(merged)

    return BaseTool.from_schema(
        function=proxy,
        name=name,
        description=desc,
        json_schema=schema,
    )
