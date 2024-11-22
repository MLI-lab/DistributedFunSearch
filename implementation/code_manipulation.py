
"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.
- Program, which contains a code preface (which could be imports, global
  variables and classes, ...) and a list of Functions.
"""
import ast # to parse Python code into its Abstract Syntax Tree
from collections.abc import Iterator, MutableSet, Sequence # to define types 
import dataclasses # Provides decorators to simplify class structure 
import io #  Handling input/output streams.
import tokenize # Breaks python code into tokens
import json
import re # For pattern matching
import logging
logger = logging.getLogger('my_logger')


@dataclasses.dataclass
class Function:
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    hash_value: int | None = None  # Add the hash_value attribute


    def __str__(self) -> str:
        """ Maps Function object to str correctly formatted. """
        return_type_str = f' -> {self.return_type}' if self.return_type else ''
        docstring_str = f'    """{self.docstring}"""' if self.docstring else ''
        function_str = f'def {self.name}({self.args}){return_type_str}:\n{docstring_str}\n{self.body}\n'
        return function_str

    def clean_body(self):
        """Return a cleaned version of the function body for comparison."""
        return self.clean_function_body(self.body)

    def serialize(self) -> dict:
        """Returns a dictionary representing the serializable parts of the function."""
        return {
            "name": self.name,
            "args": self.args,
            "body": self.body,
            "return_type": self.return_type,
            "docstring": self.docstring, 
            "hash_value": self.hash_value 
        }

    @staticmethod
    def deserialize(serialized_str: str):
        """Deserializes the JSON string back to a Function object."""
        data = json.loads(serialized_str)
        return Function(**data)

    def to_dict(self):
        return {
            "name": self.name,
            "args": self.args,
            "body": self.body,
            "return_type": self.return_type,
            "docstring": self.docstring, 
            "hash_value": self.hash_value
        }

    @staticmethod
    def from_dict(data: dict):
        return Function(
            name=data["name"],
            args=data["args"],
            body=data["body"],
            return_type=data.get("return_type", None),
            docstring=data.get("docstring", None), 
            hash_value=data.get("hash_value", None) 
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

  #`preface` is everything from the beginning of the code till the first function is found.
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
  """Extends functionality of Python's ast.NodeVisitor. Parses code to collect all required information to produce a `Program`.

  Note that we do not store function decorators.
  """

  def __init__(self, sourcecode: str):
    self._codelines: list[str] = sourcecode.splitlines()
    self._preface: str = ''
    self._functions: list[Function] = []
    self._current_function: str | None = None

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    """Collects all information about the function being parsed."""
    if node.col_offset == 0:  # We only care about first level functions.
      self._current_function = node.name 
      if not self._functions: # Captures preface by checking if no functions processed yet
        self._preface = '\n'.join(self._codelines[:node.lineno - 1]) 
      function_end_line = node.end_lineno
      body_start_line = node.body[0].lineno - 1
      # Extract the docstring.
      docstring = None
      if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value,
                                                           ast.Str):
        docstring = f'  {ast.literal_eval(ast.unparse(node.body[0]))}'
        if len(node.body) > 1:
          body_start_line = node.body[1].lineno - 1
        else:
          body_start_line = function_end_line

      self._functions.append(Function(
          name=node.name,
          args=ast.unparse(node.args),
          return_type=ast.unparse(node.returns) if node.returns else None,
          docstring=docstring,
          body='\n'.join(self._codelines[body_start_line:function_end_line]),
      ))
    self.generic_visit(node)

  def return_program(self) -> Program:
    return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
  """Returns Program object by parsing input text using Python AST."""
  try:
    # We assume that the program is composed of some preface (e.g. imports,
    # classes, assignments, ...) followed by a sequence of functions.
    tree = ast.parse(text)
    visitor = ProgramVisitor(text) # ProgramVisitor instance is initialized with text to extract specific lines of code when visiting nodes
    visitor.visit(tree) # Traverses the AST node by node using visitor pattern collecting inf about functions and preface
    return visitor.return_program() # Use extended method to return program
  except Exception as e:
    logger.warning('Failed parsing %s', text)
    raise e


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
  except Exception as e:
    logger.warning('Failed parsing %s', code)
    raise e

def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
  """Function parses the code into tokens, identifies function call tokens, and replaces instances of source_name with target_name where appropriate. """
  if source_name not in code:
    return code
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
     # if token doesnt meet criteria for renaming its added to #modified_tokens' unchanged  
    else:
      modified_tokens.append(token)
  # The sequence of original and modified tokens is then untokenized back into a coherent piece of code
  return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
  """Returns the set of all functions called in `code`."""
  return set(token.string for token, is_call in
             _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, name: str) -> Iterator[str]:
    """Yields names of functions decorated with `@name` in `code`."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
      #checks whether the current node is an instance of ast.FunctionDef, which is the node type representing a function definition. 
      #If the node is indeed a function definition, then it can potentially have decorators
        if isinstance(node, ast.FunctionDef):
            #For every FunctionDef node, there is a decorator_list attribute. This attribute contains a list of all the decorator nodes that are applied to the function.
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == name:
                    yield node.name


