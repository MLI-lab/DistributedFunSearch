# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License - Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing - software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND - either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
  Function: containing all the information we need about functions (name, args,
    body and optionally a return type and a docstring).
  Program: which contains a code preface (which could be imports, global
    variables and classes, etc.) and a list of Functions.
"""

import ast # to parse Python code into its Abstract Syntax Tree
from collections.abc import Iterator, Sequence  # to define types
import copy
import dataclasses # Provides decorators to simplify class structure
import io #  Handling input/output streams.
import tokenize # Breaks python code into tokens
import json
import re # For pattern matching
import logging

logger = logging.getLogger('main_logger')


@dataclasses.dataclass
class Function:
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    hash_value: int | None = None  # Add the hash_value attribute
    # Evolutionary lineage tracking
    program_id: int | None = None  # Unique ID for this program
    parent_ids: list[int] | None = None  # IDs of programs in the prompt that generated this
    generation: int = 0  # Generation number (0 for baseline, increments for offspring)
    timestamp: float | None = None  # When this program was created
    # Algorithm description
    description: str | None = None  # One-sentence algorithm description
    # Reasoning trace from thinking models (e.g. Qwen3)
    thinking_trace: str | None = None


    def __str__(self) -> str:
        """ Maps Function object to str correctly formatted. """
        return_type_str = f' -> {self.return_type}' if self.return_type else ''
        # Only include docstring line if docstring exists (avoid empty line after def)
        if self.docstring:
            docstring_str = f'    """{self.docstring}"""\n'
        else:
            docstring_str = ''
        function_str = f'def {self.name}({self.args}){return_type_str}:\n{docstring_str}{self.body}\n'
        return function_str

    def clean_body(self):
        """Return a cleaned version of the function body for comparison."""
        return self.clean_function_body(self.body)

    def serialize(self) -> dict:
        """Returns a dictionary representing the serializable parts of the function."""
        d = {
            "name": self.name,
            "args": self.args,
            "body": self.body,
            "return_type": self.return_type,
            "docstring": self.docstring,
            "hash_value": self.hash_value,
            "description": self.description
        }
        if self.thinking_trace:
            d["thinking_trace"] = self.thinking_trace
        return d

    @staticmethod
    def deserialize(serialized_str: str):
        """Deserializes the JSON string back to a Function object."""
        data = json.loads(serialized_str)
        return Function(**data)

    def to_dict(self):
        d = {
            "name": self.name,
            "args": self.args,
            "body": self.body,
            "return_type": self.return_type,
            "docstring": self.docstring,
            "hash_value": self.hash_value,
            "description": self.description
        }
        if self.thinking_trace:
            d["thinking_trace"] = self.thinking_trace
        return d

    @staticmethod
    def from_dict(data: dict):
        return Function(
            name=data["name"],
            args=data["args"],
            body=data["body"],
            return_type=data.get("return_type", None),
            docstring=data.get("docstring", None),
            hash_value=data.get("hash_value", None),
            description=data.get("description", None),
            thinking_trace=data.get("thinking_trace", None)
        )

    @staticmethod
    def clean_function_body(body: str) -> str:
        """Remove comments # and normalize whitespace to be in one line."""
        # Remove comments
        body = re.sub(r"#.*", "", body)
        # Normalize whitespace by replacing sequences of whitespace characters with a single space
        body = re.sub(r"\s+", " ", body)
        # Strip leading/trailing whitespace
        body = body.strip()
        return body



@dataclasses.dataclass(frozen=True)
class Program:
  """A parsed Python program."""

  #`preface` is everything from the beginning of the code till the first function is found. (so also class methods)
  preface: str
  functions: list[Function]

  def __str__(self) -> str:
    program = f'{self.preface}\n' if self.preface else ''
    program += '\n'.join([str(f) for f in self.functions])
    return program

  def find_function_index(self, function_name: str) -> int:
    """Returns the index of input function name."""
    function_names = [f.name for f in self.functions]
    count = function_names.count(function_name) # Count occurances of function name in list
    if count == 0:
      raise ValueError(
          f'function {function_name} does not exist in program:\n{str(self)}'
      )
    if count > 1:
      raise ValueError(
          f'function {function_name} exists more than once in program:\n'
          f'{str(self)}'
      )
    index = function_names.index(function_name) #Find index after confirming function exists only once
    return index

  def get_function(self, function_name: str) -> Function:
    index = self.find_function_index(function_name)
    return self.functions[index]

  def serialize(self) -> str:
      """Serializes the program to a JSON string."""
      return json.dumps(dataclasses.asdict(self), default=lambda o: o.serialize() if hasattr(o, 'serialize') else str(o))

  @staticmethod
  def deserialize(serialized_str: str):
      """Deserializes the JSON string back to a Program object."""
      data = json.loads(serialized_str)
      functions = [Function.deserialize(f) for f in data['functions']]
      return Program(preface=data['preface'], functions=functions)


class ProgramVisitor(ast.NodeVisitor):
    def __init__(self, sourcecode: str, remove_classes: bool = False):
        self._remove_classes = remove_classes  # Flag for removing classes
        self._class_lines: set[int] = set()
        self._codelines: list[str] = sourcecode.splitlines() # split the full source code into lines
        self._preface: str = ''
        self._functions: list[Function] = []
        self._current_function: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Marks class definitions for removal."""
        if self._remove_classes:
            # Mark every line of the class for removal (0-indexed)
            for lineno in range(node.lineno - 1, node.end_lineno):
                self._class_lines.add(lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collects function definitions and captures the preface."""
        if node.col_offset == 0:  # Only consider top-level functions.
            self._current_function = node.name
            if not self._functions:
                # Capture preface as all lines before the first function.
                raw_preface = self._codelines[:node.lineno - 1]
                # If remove_classes is enabled, filter out the marked lines.
                if self._remove_classes:
                    raw_preface = [
                        line for idx, line in enumerate(raw_preface)
                        if idx not in self._class_lines
                    ]
                self._preface = "\n".join(raw_preface)
            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1
            # Extract the docstring if available.
            docstring = None
            # Check for docstring (supports both old ast.Str and new ast.Constant)
            if isinstance(node.body[0], ast.Expr):
                # Python 3.8+: string literals are ast.Constant
                if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    docstring = f'  {node.body[0].value.value}'
                    if len(node.body) > 1:
                        body_start_line = node.body[1].lineno - 1
                    else:
                        body_start_line = function_end_line
                # Python 3.7 and earlier: string literals are ast.Str
                elif isinstance(node.body[0].value, ast.Str):
                    docstring = f'  {ast.literal_eval(ast.unparse(node.body[0]))}'
                    if len(node.body) > 1:
                        body_start_line = node.body[1].lineno - 1
                    else:
                        body_start_line = function_end_line

            # Handle one-liner functions where body is on same line as def
            # e.g., "def str2int(s): return int(s, 2)"
            if body_start_line == node.lineno - 1 and function_end_line == node.lineno:
                # One-liner: extract just the body part after the colon
                full_line = self._codelines[body_start_line]
                # Find the colon after the closing parenthesis of args
                colon_idx = full_line.find(':')
                if colon_idx != -1:
                    body_part = full_line[colon_idx + 1:].strip()
                    # Ensure proper indentation for the body
                    body = f'    {body_part}'
                else:
                    body = "\n".join(self._codelines[body_start_line:function_end_line])
            else:
                body = "\n".join(self._codelines[body_start_line:function_end_line])

            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args),
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body=body,
            ))
        self.generic_visit(node)

    def get_clean_code(self) -> str:
        """Return the complete source code with class lines removed."""
        return "\n".join(
            line for idx, line in enumerate(self._codelines)
            if idx not in self._class_lines
        )

    def return_program(self) -> Program:
        # Optionally, rebuild the preface from the cleaned code.
        if self._remove_classes:
            # Assume the preface is the first N lines (as originally captured) in the cleaned version.
            num_preface_lines = len(self._preface.splitlines())
            cleaned_lines = self.get_clean_code().splitlines()
            clean_preface = "\n".join(cleaned_lines[:num_preface_lines])
            logger.debug(f"The clean_preface {clean_preface} and cleaned_lines {cleaned_lines}.")
        else:
            clean_preface = self._preface
        return Program(preface=clean_preface, functions=self._functions)


def text_to_program(text: str, remove_classes: bool = False) -> Program:

    """Parse text into a Program.

    Builds a preface (everything before the first top-level function).
    Collects every top-level function into Function objects.
    When remove_classes is True, any code lines belonging to top-level class
    definitions are discarded before the preface and functions are assembled.

    Returns a Program containing the cleaned preface and functions.
    """

    try:
        tree = ast.parse(text)
        logger.debug("AST parsed successfully.")
    except SyntaxError as e:
        logger.warning(f"Syntax error during AST parsing: {e}")
        raise

    try:
        visitor = ProgramVisitor(text, remove_classes)
        visitor.visit(tree)
        logger.debug("AST visited successfully.")
        return visitor.return_program()
    except Exception as e:
        logger.warning(f"AST visitor error: {e}", exc_info=True)
        raise




def text_to_function(text: str) -> Function:
  """Returns Function object by parsing input text using Python AST."""
  program = text_to_program(text)
  if len(program.functions) != 1:
    raise ValueError(f'Only one function expected, got {len(program.functions)}'
                     f':\n{program.functions}')
  return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
  """Transforms `code` into Python tokens. Each token represents a basic element of Python syntax.
     Used to identify function calls and renaming them.
  """
  code_bytes = code.encode()
  code_io = io.BytesIO(code_bytes)
  return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
  """Transforms a list of Python tokens into code."""
  code_bytes = tokenize.untokenize(tokens)
  return code_bytes.decode()


def _yield_token_and_is_call(
    code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
  """Yields each token with a bool indicating whether it is a function call."""
  try:
    tokens = _tokenize(code)
    prev_token = None
    is_attribute_access = False
    for token in tokens:
      if (prev_token and  # If the previous token exists and
          prev_token.type == tokenize.NAME and  # it is a Python identifier
          token.type == tokenize.OP and  # and the current token is a delimiter
          token.string == '('):  # and in particular it is '('.
        yield prev_token, not is_attribute_access
        is_attribute_access = False
      else:
        if prev_token:
          is_attribute_access = (
              prev_token.type == tokenize.OP and prev_token.string == '.'
          )
          yield prev_token, False
      prev_token = token
    if prev_token:
      yield prev_token, False
  except tokenize.TokenError:
    raise
  except Exception as e:
    logger.warning('Failed parsing %s', code)
    raise e

def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
  """Parses code into tokens, identifies function call tokens, and replaces instances of source_name with target_name."""
  if source_name not in code:
    return code
  try:
    modified_tokens = []
    for token, is_call in _yield_token_and_is_call(code):
      if is_call and token.string == source_name:
        # Replace the function name token
        modified_token = tokenize.TokenInfo(
            type=token.type,
            string=target_name,
            start=token.start,
            end=token.end,
            line=token.line,
        )
        modified_tokens.append(modified_token)
      else:
        # Token doesnt meet criteria for renaming, add unchanged
        modified_tokens.append(token)
    # Untokenize back into code
    return _untokenize(modified_tokens)
  except tokenize.TokenError:
    # ast.parse() and tokenize have different strictness — code that passed
    # AST validation can still fail tokenization. Return unchanged.
    logger.warning(f"Tokenize failed during rename ({source_name} -> {target_name}), returning code unchanged")
    return code


def _find_function_lines(tree: ast.Module, function_name: str) -> tuple[int, int]:
  """Find body start and end line of a function in the AST.

  AST lineno is 1-based, so body_start is converted to 0-based (lineno - 1)
  for direct use in list slicing: lines[body_start:end_line].
  Skips the entire def signature, including multi-line signatures.
  """
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == function_name:
      body_start = node.body[0].lineno - 1  # skip entire def signature
      return body_start, node.end_lineno
  raise ValueError(f"Function '{function_name}' not found in AST")


def _parse_code(code: str) -> tuple[ast.Module | None, str]:
  """Parse Python code, truncating from the first syntax error onward until it compiles.

  This helps when LLMs append trailing junk (bad comments, extra braces, markdown
  artifacts). Keeps truncating until the code compiles or is empty, so a good
  priority function body followed by arbitrarily long rambling will still be
  recovered. Mid function errors will lose the return statement, but that is
  caught later during evaluation.

  Returns:
      (parsed_tree, remaining_code) where parsed_tree is None if parsing failed.
  """
  tree = None
  deletion_count = 0

  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      deletion_count += 1
      lines = code.splitlines()
      deleted_line = lines[e.lineno - 1] if e.lineno <= len(lines) else "(unknown)"
      logger.warning(f"AST SyntaxError at line {e.lineno}: {e.msg}. Deleting: {deleted_line[:100]}")
      code = '\n'.join(lines[:e.lineno - 1])
      if not code.strip():
        logger.error("AST parsing consumed all lines, code is entirely invalid.")
        return None, ''

  if not code:
    logger.warning("AST parsing resulted in empty code after deletions")
    return None, ''

  if deletion_count > 0:
    logger.info(f"AST parsing required {deletion_count} line deletions")

  return tree, code


def _normalize_indent(lines, target=4):
  """Normalize indentation of body lines to target spaces.

  Detects actual indent from first non-empty line and scales proportionally.
  Fixes LLM output that uses 2/3/5/6-space indent mixing with 4-space helpers.
  """
  first = next((l for l in lines if l.strip()), '')
  if not first:
    return lines
  actual = len(first) - len(first.lstrip())
  if actual == target or actual == 0:
    return lines
  result = []
  for line in lines:
    if not line.strip():
      result.append(line)
    else:
      stripped = line.lstrip()
      current = len(line) - len(stripped)
      new_indent = round(current * target / actual)
      result.append(' ' * new_indent + stripped)
  return result


def parse_llm_output(raw_output: str) -> tuple[str, str | None, str | None, str | None]:
  """
  Parse raw LLM output and extract a valid Python function body.

  Pipeline:
  0. Extract <think> reasoning traces (Qwen3 thinking mode)
  1. Extract and strip <description> tags
  2. Extract code: <code> tags > last markdown fence > raw text
  3. Strip any nested markdown fences
  4. Detect structure: has def priority() -> extract body, else treat all as body
  5. Parse with AST (delete invalid lines until it compiles)

  Returns:
      (function_body, description, thinking_trace, failure_reason)
      failure_reason is None on success, or a string describing why parsing failed.
  """
  if not raw_output:
    return '', None, None, 'LLM returned empty output'

  # Extract thinking trace from Qwen3 <think> blocks
  think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
  thinking_trace = think_match.group(1).strip() if think_match else None

  # Extract description and clean input
  description_match = re.search(r'<description>(.*?)</description>', raw_output, re.DOTALL)
  description = description_match.group(1).strip() if description_match else None

  # Remove thinking and description tags from text we'll search for code
  text = re.sub(r'<think>.*?</think>\s*', '', raw_output, flags=re.DOTALL)
  text = re.sub(r'<description>.*?</description>\s*', '', text, flags=re.DOTALL)

  # Extract code by priority: try <code> tags first, then markdown fences, then raw text

  # Try <code> tags first (take last match)
  code_matches = list(re.finditer(r'<code>(.*?)</code>', text, re.DOTALL))
  if not code_matches:
    # Fallback: unclosed <code> tag (output truncated before </code>)
    unclosed = re.search(r'<code>(.*)', text, re.DOTALL)
    if unclosed:
      code_matches = [unclosed]
      logger.debug(f"Extracted code from unclosed <code> tag (truncated output)")
  if code_matches:
    code = code_matches[-1].group(1)
    logger.debug(f"Extracted code from <code> tags ({len(code)} chars, match {len(code_matches)} of {len(code_matches)})")
  else:
    # Try last markdown fence (handles ```python, ```py, ```python3, or plain ```)
    fence_matches = list(re.finditer(r'```(?:python|py|python3)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE))
    if fence_matches:
      code = fence_matches[-1].group(1)
      logger.debug(f"Extracted code from markdown fence {len(fence_matches)} of {len(fence_matches)} ({len(code)} chars)")
    else:
      # Fall back to raw text if it looks like code
      if not re.search(r'\b(def |return |import |from \w+ import )\b', text):
        return '', description, thinking_trace, 'LLM returned no code (no code tags and no code found in output)'
      code = text
      logger.debug(f"Using raw text as code ({len(code)} chars)")

  # Cleanup: strip any fences that might be nested inside (e.g., inside <code> tags)
  fence_in_code = re.search(r'```(?:python|py|python3)?\s*\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
  if fence_in_code:
    code = fence_in_code.group(1)

  # Strip trailing whitespace and leading newlines, but preserve leading indentation
  # (strip() would break indented body code by removing indent from first line only)
  code = code.lstrip('\n').rstrip()
  if not code:
    return '', description, thinking_trace, 'LLM returned no code (only markdown/description tags)'

  # Determine if this is body only or full function

  # First, look for a function starting with "priority" (e.g. priority_improved, priority_v2_degree_based)
  priority_match = re.search(r'^\s*def\s+(priority\w*)\s*\(', code, re.MULTILINE)

  # Otherwise, find all defs and pick the first non-skipped one
  # But only when code starts at column 0 (full function output).
  # If first non-empty line is indented, this is completion-mode output and
  # any def appearing later is a helper or trailing junk, not the priority function.
  if not priority_match:
    first_non_empty = next((l for l in code.splitlines() if l.strip()), '')
    is_completion = first_non_empty and first_non_empty[0].isspace()

    if not is_completion:
      skip_names = {'main', 'evaluate', 'test', 'unused', 'helper', '__init__'}
      all_funcs = list(re.finditer(r'^\s*def\s+(\w+)\s*\(', code, re.MULTILINE))
      candidates = [m for m in all_funcs if m.group(1) not in skip_names]
      if len(candidates) >= 1:
        priority_match = candidates[0]
        logger.info(f"No priority-prefixed function, using '{candidates[0].group(1)}' as priority function")

  if priority_match:
    # Full function output, extract priority body
    function_name = priority_match.group(1)
    logger.debug(f"Found {function_name} definition, extracting body")

    tree, parsed_code = _parse_code(code)
    if tree is None:
      return '', description, thinking_trace, 'code too broken to compile (no valid lines survived AST truncation)'

    parsed_lines = parsed_code.splitlines()

    # Collect module level imports to move inside function body
    imports = []
    for node in tree.body:
      if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append(parsed_lines[node.lineno - 1])
      elif isinstance(node, ast.FunctionDef):
        break

    # Collect helper functions (any function that is not priority)
    # Skip common test/utility function names
    skip_functions = {function_name, 'main', 'evaluate', 'test', 'unused'}
    helper_lines = []
    for node in tree.body:
      if isinstance(node, ast.FunctionDef) and node.name not in skip_functions:
        # Extract the full function source and add 4 spaces to each line
        func_start = node.lineno - 1
        func_end = node.end_lineno
        func_lines = parsed_lines[func_start:func_end]
        # Add 4 spaces to each line to make it a nested function
        indented_func = ['    ' + line for line in func_lines]
        helper_lines.extend(indented_func)
        helper_lines.append('')  # blank line after helper
        logger.debug(f"Including helper function '{node.name}' as nested function")

    # Extract priority body lines (skips entire def signature, including multi-line)
    try:
      start_line, end_line = _find_function_lines(tree, function_name)
    except ValueError:
      # Function was found by regex but removed by _parse_code line deletions
      logger.warning(f"Function '{function_name}' found by regex but lost during AST repair")
      return '', description, thinking_trace, f"AST repair truncated '{function_name}' to no body (whole body had syntax errors)"
    body_lines = parsed_lines[start_line:end_line]

    # Normalize indentation to 4 spaces (LLMs sometimes use 2/3/5/6-space indent)
    body_lines = _normalize_indent(body_lines)

    # Prepend imports and helpers inside function body
    prefix_lines = []
    if imports:
      logger.debug(f"Moving {len(imports)} import(s) inside function body")
      prefix_lines.extend(['    ' + imp for imp in imports])
    if helper_lines:
      prefix_lines.extend(helper_lines)

    if prefix_lines:
      body_lines = prefix_lines + body_lines

    body = '\n'.join(body_lines) + '\n\n'

    # Rename any non-standard function name to 'priority' so the template works
    if function_name != 'priority':
      body = rename_function_calls(body, function_name, 'priority')
      logger.info(f"Renamed recursive calls '{function_name}' -> 'priority' in body")

  else:
    # Completion output, entire code is the body
    logger.debug("No priority function found, treating as body code")

    code_lines = code.splitlines()
    skip_functions = {'main', 'evaluate', 'test', 'unused', '_fake_', 'priority'}

    # Separate module level items (imports, functions at column 0) from body code (indented)
    import_lines = []
    helper_lines = []
    body_code_lines = []

    i = 0
    while i < len(code_lines):
      line = code_lines[i]

      # Skip empty lines at module level
      if not line.strip():
        i += 1
        continue

      # Import at column 0
      if re.match(r'^(import |from \w+ import )', line):
        import_lines.append('    ' + line)
        i += 1
        continue

      # Function at column 0
      func_match = re.match(r'^def\s+(\w+)\s*\(', line)
      if func_match:
        func_name = func_match.group(1)
        # Find end of function (next line at column 0 or end of code)
        func_start = i
        i += 1
        while i < len(code_lines) and (not code_lines[i].strip() or code_lines[i][0].isspace()):
          i += 1
        func_end = i

        if func_name not in skip_functions:
          func_lines = code_lines[func_start:func_end]
          indented_func = ['    ' + fl for fl in func_lines]
          helper_lines.extend(indented_func)
          helper_lines.append('')
          logger.debug(f"Including helper function '{func_name}' as nested function")
        continue

      # Indented code (body) or non indented body
      # Collect all remaining lines until we hit a module level item
      while i < len(code_lines):
        line = code_lines[i]
        if line.strip() and not line[0].isspace():
          # Check if it's a module level item
          if re.match(r'^(import |from \w+ import |def\s+\w+\s*\()', line):
            break
        body_code_lines.append(line)
        i += 1

    body_code = '\n'.join(body_code_lines)

    # Add indentation if code has none
    if body_code and body_code.strip() and not body_code.lstrip('\n')[0].isspace():
      body_code = '    ' + body_code.replace('\n', '\n    ')

    # Validate by wrapping in fake function and parsing
    wrapped = 'def _fake_():\n' + body_code
    tree, parsed_code = _parse_code(wrapped)
    if tree is None:
      return '', description, thinking_trace, 'code too broken to compile (no valid lines survived AST truncation)'

    # Extract validated body
    _, end_line = _find_function_lines(tree, '_fake_')
    body_lines = parsed_code.splitlines()[1:end_line]

    # Normalize indentation to 4 spaces
    body_lines = _normalize_indent(body_lines)

    # Prepend imports and helpers to body
    prefix_lines = import_lines + helper_lines
    if prefix_lines:
      body_lines = prefix_lines + body_lines

    body = '\n'.join(body_lines) + '\n\n'

  logger.debug(f"Parsed LLM output: {len(body)} chars, description={'yes' if description else 'no'}, thinking={'yes' if thinking_trace else 'no'}")
  return body, description, thinking_trace, None


def sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: Program,
    function_to_evolve: str,
) -> tuple[Function, str, str | None, str | None, str | None]:
  """Integrates a generated code as string into a larger program template.

  Returns:
      tuple: (evolved_function, program_str, description, thinking_trace, failure_reason)
  """
  # Parse LLM output: extract from XML tags, strip markdown, validate with AST
  body, description, thinking_trace, failure_reason = parse_llm_output(generated_code)
  if version_generated is not None:

    body = rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)
  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)

  evolved_function.body = body
  evolved_function.description = description
  evolved_function.thinking_trace = thinking_trace
  return evolved_function, str(program), description, thinking_trace, failure_reason
