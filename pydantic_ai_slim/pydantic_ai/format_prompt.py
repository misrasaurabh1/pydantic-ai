from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)


def format_as_xml(
    obj: Any,
    root_tag: str = 'examples',
    item_tag: str = 'example',
    include_root_tag: bool = True,
    none_str: str = 'null',
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        include_root_tag: Whether to include the root tag in the output
            (The root tag is always included if it includes a body - e.g. when the input is a simple value).
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    to_xml_obj = _ToXml(item_tag=item_tag, none_str=none_str)
    el = to_xml_obj.to_xml(obj, root_tag)
    if not include_root_tag and el.text is None:
        # Inline _rootless_xml_elements for performance.
        generator = (
            (ElementTree.indent(sub, space=indent) or ElementTree.tostring(sub, encoding='unicode'))
            if indent is not None
            else ElementTree.tostring(sub, encoding='unicode')
            for sub in el
        )
        join_str = '' if indent is None else '\n'
        return join_str.join(generator)
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        # Path for tag selection
        class_tag = self.item_tag if tag is None else tag
        element = ElementTree.Element(class_tag)

        vtype = type(value)

        # Fast-path for the most common scalar types
        if value is None:
            element.text = self.none_str
            return element
        if vtype is str:
            element.text = value
            return element
        if vtype in (int, float, bool):
            element.text = str(value)
            return element
        if vtype is bytes or vtype is bytearray:
            # decode is slow, but little can be done
            element.text = value.decode(errors='ignore')
            return element
        if isinstance(value, date):
            # datetime is a subclass of date (safe for isoformat)
            element.text = value.isoformat()
            return element
        if isinstance(value, Mapping):
            self._mapping_to_xml(element, value)
            return element

        # For dataclasses
        if is_dataclass(value) and not isinstance(value, type):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            # Try to avoid asdict if possible (fastpath: if it has a __dict__ without slots/properties)
            if hasattr(value, '__dict__') and getattr(value, '__slots__', None) is None:
                dc_dict = value.__dict__  # Avoid asdict's deepcopy
            else:
                dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
            return element

        # For pydantic models
        if isinstance(value, BaseModel):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            # model_dump is required for correct serialization
            self._mapping_to_xml(element, value.model_dump(mode='python'))
            return element

        # For general iterables (lists/tuples/sets)
        # Must check last: str, bytes, mapping, BaseModel, dataclass already handled
        if isinstance(value, Iterable):
            append = element.append
            to_xml = self.to_xml
            for item in value:
                item_el = to_xml(item, None)
                append(item_el)
            return element

        raise TypeError(f'Unsupported type for XML formatting: {type(value)}')

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
