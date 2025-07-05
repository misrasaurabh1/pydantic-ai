from typing import Any, Protocol

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool


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
    # Avoid copying and sorting unless there are required params, prealloc locals for perf
    function_name = langchain_tool.name
    function_description = langchain_tool.description
    # Fast dict-comprehension and lazy evaluation for required/defaults
    inputs = langchain_tool.args
    schema: JsonSchemaValue = langchain_tool.get_input_jsonschema()
    # Only set keys if needed
    if 'additionalProperties' not in schema:
        schema['additionalProperties'] = False

    defaults = {}
    required = []

    for name, detail in inputs.items():
        if 'default' in detail:
            defaults[name] = detail['default']
        else:
            required.append(name)

    if required:
        # Sort in place for less memory
        required.sort()
        schema['required'] = required

    # Restructures the arguments to match langchain tool run
    def proxy(*args: Any, **kwargs: Any) -> str:
        assert not args, 'This should always be called with kwargs'
        # Merge defaults and kwargs (kwargs wins), faster than making new dict then update
        # For small numbers of items, {**defaults, **kwargs} is very fast, but in Py3.9+ '|' is fastest
        kwargs = defaults | kwargs if defaults else kwargs
        return langchain_tool.run(kwargs)

    return Tool.from_schema(
        function=proxy,
        name=function_name,
        description=function_description,
        json_schema=schema,
    )
