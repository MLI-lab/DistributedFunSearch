"""Prompt building module for FunSearch, EoH, and ReEvo strategies.

Flow:
    1. load_specification()  Called once at startup, loads templates and content into PromptSpec
    2. select_template()     Called each iteration, returns (template_name, num_programs_needed)
    3. build_prompt()        Called each iteration, fills placeholders and returns prompt string

Placeholders:
    Filled once in load_specification() (static content from spec files):
        {imports}, {problem_description}, {problem_desc}, {func_desc}, {user_generator}

    Filled each iteration in build_prompt() (from sampled programs):
        FunSearch:  {fewshot_examples}, {function_header}
        EoH/ReEvo:  {worse_code}, {better_code}, {thought}
        ReEvo:      {reflection}, {prior_reflection}, {new_reflections}, {initial_reflection}
"""

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import random
import re
import logging

logger = logging.getLogger('main_logger')


class PromptStrategy(Enum):
    """Available prompt strategies."""
    FUNSEARCH = "funsearch"
    EOH = "eoh"
    REEVO = "reevo"


@dataclass
class TemplateRequirements:
    """Requirements for a template/phase."""
    num_programs: int          # 1 if only {better_code}, 2 if also {worse_code}
    needs_thought: bool        # True if {thought} in template
    needs_reflection: bool     # True if {reflection} in template
    samples_per_prompt: int | None = None  # LLM outputs per prompt (None = use sampler default)


@dataclass
class PromptSpec:
    """Immutable specification loaded once at startup."""
    strategy: PromptStrategy

    # Shared content (loaded from files)
    imports: str = ""
    problem_description: str = ""  # For funsearch, eoh
    problem_desc: str = ""         # For reevo (short inline version)
    func_desc: str = ""
    system_message: str | None = None

    # Templates
    templates: dict[str, str] = field(default_factory=dict)  # template_name -> content
    template_requirements: dict[str, TemplateRequirements] = field(default_factory=dict)

    # FunSearch-specific
    fewshot_num_examples: int = 2
    evaluation_preamble: str = ""
    evaluation_script: str = ""
    function_to_evolve: str = "priority"
    function_args: str = ""
    return_type: str = "float"

    # ReEvo-specific
    user_generator: str = ""  # Pre-built from user_generator.txt
    reflector_system_message: str | None = None  # Separate system message for reflector
    initial_reflection: str = ""  # Initial hints for initialization (can be empty)

    # Score display options (shared across all strategies)
    show_scores: bool = False
    score_display_mode: str = "relative"  # "absolute" or "relative"
    best_known_solutions: dict = field(default_factory=dict)  # {(n, s, q): score}

    # Docstring templates (loaded from specification files)
    docstring_baseline: str = ""  # For single/worse function, with {score} placeholder
    docstring_improved: str = ""  # For better function, with {score} placeholder
    score_label_absolute: str = "Scores:"  # Label for absolute score display
    score_label_relative: str = "Relative to baseline:"  # Label for relative score display


def _load_file(path: Path) -> str:
    """Load file content, return empty string if not found."""
    if path.exists():
        return path.read_text()
    logger.warning(f"File not found: {path}")
    return ""


def _load_directory(path: Path) -> dict[str, str]:
    """Load all .txt files from directory as {name: content}."""
    if not path.is_dir():
        logger.warning(f"Directory not found: {path}")
        return {}

    result = {}
    for txt_file in path.glob("*.txt"):
        result[txt_file.stem] = txt_file.read_text().strip()
    return result


def _infer_requirements(template: str) -> TemplateRequirements:
    """Infer template requirements from placeholders used."""
    has_worse_code = "{worse_code}" in template
    has_thought = "{thought}" in template
    has_reflection = "{reflection}" in template or "{prior_reflection}" in template

    # If template uses {worse_code}, it needs 2 programs
    # Otherwise just 1 (the {better_code})
    num_programs = 2 if has_worse_code else 1

    return TemplateRequirements(
        num_programs=num_programs,
        needs_thought=has_thought,
        needs_reflection=has_reflection,
    )


def _extract_function_signature(initial_function_path: Path) -> tuple[str, str]:
    """Extract argument list and return type from initial function file.

    Returns:
        tuple: (args_string, return_type_string)
    """
    import ast

    content = initial_function_path.read_text()

    # Extract code from <code> tags if present
    code_match = re.search(r'<code>(.*?)</code>', content, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()

    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = ", ".join(arg.arg for arg in node.args.args)
            ret_type = ast.unparse(node.returns) if node.returns else "float"
            return args, ret_type

    raise ValueError(f"No function found in {initial_function_path}")


def _format_scores(scores: dict, spec: PromptSpec) -> str:
    """Format scores as absolute or relative string.

    Args:
        scores: Dict mapping (n, s, q) or string keys to score values.
        spec: PromptSpec with score display settings.

    Returns:
        Formatted string like "Scores: {(6,1,2): 8, (7,1,2): 14}" or
        "Relative to baseline: {(6,1,2): +0.0%, (7,1,2): +7.1%}".
        Returns empty string if show_scores is False or scores is empty.
    """
    if not spec.show_scores or not scores:
        return ""

    # Parse string keys to tuples if needed
    parsed_scores = {}
    for k, v in scores.items():
        key = eval(k) if isinstance(k, str) else k
        parsed_scores[key] = v

    if spec.score_display_mode == "absolute":
        items = [f"{k}: {v}" for k, v in sorted(parsed_scores.items())]
        return f"{spec.score_label_absolute} {{{', '.join(items)}}}"

    elif spec.score_display_mode == "relative":
        improvements = []
        for dim in sorted(parsed_scores.keys()):
            score_ours = parsed_scores.get(dim, 0)
            score_baseline = spec.best_known_solutions.get(dim, None)

            if score_baseline is not None and score_baseline != 0:
                rel_improvement = ((score_ours - score_baseline) / abs(score_baseline)) * 100
                improvements.append(f"{dim}: {rel_improvement:+.1f}%")
            else:
                improvements.append(f"{dim}: {score_ours}")

        return f"{spec.score_label_relative} {{{', '.join(improvements)}}}"

    return ""


def load_specification(
    strategy: PromptStrategy,
    spec_dir: str,
    imports_file: str = "imports/networkx.txt",
    # FunSearch options
    funsearch_template: str = "funsearch/template.txt",
    funsearch_problem_desc: str = "funsearch/problem_descriptions/baseline.txt",
    funsearch_system_message: str | None = None,
    funsearch_evaluation_preamble: str | None = None,
    funsearch_evaluation_script: str | None = None,
    fewshot_num_examples: int = 2,
    initial_functions_dir: str = "initial_functions/graph_networkx",
    # EoH options
    eoh_styles_dir: str = "eoh/styles",
    eoh_problem_desc: str = "eoh/problem_descriptions/baseline.txt",
    eoh_func_desc: str = "eoh/func_desc.txt",
    eoh_system_message: str = "eoh/system_message.txt",
    # ReEvo options
    reevo_templates_dir: str = "reevo/templates",
    reevo_problem_desc: str = "reevo/problem/problem_desc.txt",
    reevo_func_desc: str = "reevo/problem/func_desc.txt",
    reevo_generator_system: str = "reevo/system/generator.txt",
    reevo_reflector_system: str = "reevo/system/reflector.txt",
    reevo_initial_reflection: str | None = "reevo/initial_reflection.txt",  # None or path
    # Score display options
    show_scores: bool = False,
    score_display_mode: str = "relative",
    best_known_solutions: dict | None = None,
    # Docstring templates (paths relative to spec_dir)
    docstring_baseline: str = "docstrings/baseline.txt",
    docstring_improved: str = "docstrings/improved.txt",
    score_label_absolute: str = "docstrings/score_label_absolute.txt",
    score_label_relative: str = "docstrings/score_label_relative.txt",
) -> PromptSpec:
    """Load all templates and content once at startup.

    Args:
        strategy: Which prompt strategy to use
        spec_dir: Base directory for specification files
        imports_file: Path to imports file (relative to spec_dir)
        ... (strategy-specific options)

    Returns:
        PromptSpec with all content loaded
    """
    base = Path(spec_dir)

    # Load shared content
    imports = _load_file(base / imports_file)

    # Initialize defaults
    templates = {}
    problem_description = ""
    problem_desc = ""
    func_desc = ""
    system_message = None
    user_generator = ""
    reflector_system_message = None
    initial_reflection = ""
    evaluation_preamble = ""
    evaluation_script = ""
    function_args = ""
    return_type = "float"

    if strategy == PromptStrategy.FUNSEARCH:
        # Load single template
        templates = {"funsearch": _load_file(base / funsearch_template).strip()}
        problem_description = _load_file(base / funsearch_problem_desc).strip()

        if funsearch_system_message:
            system_message = _load_file(base / funsearch_system_message).strip()

        if funsearch_evaluation_preamble:
            evaluation_preamble = _load_file(base / funsearch_evaluation_preamble).strip()

        if funsearch_evaluation_script:
            evaluation_script = _load_file(base / funsearch_evaluation_script)

        # Extract function signature from initial function
        initial_func_dir = base / initial_functions_dir
        if initial_func_dir.exists():
            initial_func_path = next(initial_func_dir.glob("*.txt"), None)
            if initial_func_path:
                function_args, return_type = _extract_function_signature(initial_func_path)

    elif strategy == PromptStrategy.EOH:
        # Load all style templates
        templates = _load_directory(base / eoh_styles_dir)
        problem_description = _load_file(base / eoh_problem_desc).strip()
        func_desc = _load_file(base / eoh_func_desc).strip()
        system_message = _load_file(base / eoh_system_message).strip() or None

    elif strategy == PromptStrategy.REEVO:
        # Load all templates
        templates = _load_directory(base / reevo_templates_dir)
        problem_desc = _load_file(base / reevo_problem_desc).strip()
        func_desc = _load_file(base / reevo_func_desc).strip()
        system_message = _load_file(base / reevo_generator_system).strip() or None
        reflector_system_message = _load_file(base / reevo_reflector_system).strip() or None

        # Load initial reflection (optional, can be None for empty, or path to file)
        initial_reflection = ""
        if reevo_initial_reflection:
            initial_reflection = _load_file(base / reevo_initial_reflection).strip()

        # Pre-build user_generator (used in multiple templates)
        user_gen_template = templates.get("user_generator", "")
        if user_gen_template:
            user_generator = user_gen_template.replace("{problem_desc}", problem_desc)
            user_generator = user_generator.replace("{func_desc}", func_desc)
            user_generator = user_generator.replace("{imports}", imports)
            # Remove user_generator from templates (it's a component, not a prompt template)
            del templates["user_generator"]

    # Infer requirements for each template
    template_requirements = {
        name: _infer_requirements(content)
        for name, content in templates.items()
    }

    # Note: samples_per_prompt for specific templates (e.g., mutation) can be set
    # after loading via: spec.template_requirements["mutation"].samples_per_prompt = N

    # Pre-fill static placeholders in templates (these never change)
    for name in templates:
        templates[name] = templates[name].replace("{imports}", imports)
        templates[name] = templates[name].replace("{problem_description}", problem_description)
        templates[name] = templates[name].replace("{problem_desc}", problem_desc)
        templates[name] = templates[name].replace("{func_desc}", func_desc)
        templates[name] = templates[name].replace("{user_generator}", user_generator)

    # Load docstring templates
    docstring_baseline_content = _load_file(base / docstring_baseline).strip()
    docstring_improved_content = _load_file(base / docstring_improved).strip()
    score_label_absolute_content = _load_file(base / score_label_absolute).strip()
    score_label_relative_content = _load_file(base / score_label_relative).strip()

    logger.info(f"Loaded {strategy.value} spec from {spec_dir}: {len(templates)} templates")

    return PromptSpec(
        strategy=strategy,
        imports=imports,
        problem_description=problem_description,
        problem_desc=problem_desc,
        func_desc=func_desc,
        system_message=system_message,
        templates=templates,
        template_requirements=template_requirements,
        fewshot_num_examples=fewshot_num_examples,
        evaluation_preamble=evaluation_preamble,
        evaluation_script=evaluation_script,
        function_to_evolve="priority",
        function_args=function_args,
        return_type=return_type,
        user_generator=user_generator,
        reflector_system_message=reflector_system_message,
        initial_reflection=initial_reflection,
        # Score display options
        show_scores=show_scores,
        score_display_mode=score_display_mode,
        best_known_solutions=best_known_solutions or {},
        # Docstring templates (loaded from files)
        docstring_baseline=docstring_baseline_content,
        docstring_improved=docstring_improved_content,
        score_label_absolute=score_label_absolute_content,
        score_label_relative=score_label_relative_content,
    )


def load_prompt_spec_from_config(config) -> PromptSpec:
    """Load prompt specification from a Config object.

    Convenience wrapper around load_specification that extracts parameters
    from config.prompt and config.evaluator.

    Args:
        config: Config object with prompt and evaluator attributes

    Returns:
        PromptSpec ready for use
    """
    strategy = PromptStrategy(config.prompt.strategy)

    # Convert absolute initial_functions_dir to relative path (relative to spec_dir)
    spec_dir = Path(config.prompt.spec_dir)
    initial_funcs_abs = Path(config.evaluator.initial_functions_dir)
    try:
        initial_functions_dir = str(initial_funcs_abs.relative_to(spec_dir))
    except ValueError:
        initial_functions_dir = str(initial_funcs_abs)

    spec = load_specification(
        strategy=strategy,
        spec_dir=config.prompt.spec_dir,
        imports_file=config.prompt.imports_file,
        funsearch_template=config.prompt.funsearch_template,
        funsearch_problem_desc=config.prompt.funsearch_problem_desc,
        funsearch_system_message=config.prompt.funsearch_system_message,
        funsearch_evaluation_preamble=config.prompt.funsearch_evaluation_preamble,
        funsearch_evaluation_script=config.prompt.funsearch_evaluation_script,
        fewshot_num_examples=config.prompt.fewshot_num_examples,
        initial_functions_dir=initial_functions_dir,
        eoh_styles_dir=config.prompt.eoh_styles_dir,
        eoh_problem_desc=config.prompt.eoh_problem_desc,
        eoh_func_desc=config.prompt.eoh_func_desc,
        eoh_system_message=config.prompt.eoh_system_message,
        reevo_templates_dir=config.prompt.reevo_templates_dir,
        reevo_problem_desc=config.prompt.reevo_problem_desc,
        reevo_func_desc=config.prompt.reevo_func_desc,
        reevo_generator_system=config.prompt.reevo_generator_system,
        reevo_reflector_system=config.prompt.reevo_reflector_system,
        reevo_initial_reflection=config.prompt.reevo_initial_reflection,
    )

    # Set samples_per_prompt for ReEvo mutation phase if configured
    mutation_samples = getattr(config.sampler, 'samples_per_prompt_mutation', None)
    if mutation_samples is not None and "mutation" in spec.template_requirements:
        spec.template_requirements["mutation"].samples_per_prompt = mutation_samples

    return spec


def select_template(
    spec: PromptSpec,
    state: dict | None = None,
) -> tuple[str, int]:
    """Select template and return how many programs are needed.

    Args:
        spec: Loaded prompt specification
        state: Optional state dict (used for ReEvo phase tracking)

    Returns:
        tuple: (template_name, num_programs_needed)
    """
    if spec.strategy == PromptStrategy.FUNSEARCH:
        # Fixed template, configurable num_programs
        return "funsearch", spec.fewshot_num_examples

    elif spec.strategy == PromptStrategy.EOH:
        # Random selection from styles
        template_name = random.choice(list(spec.templates.keys()))
        num_programs = spec.template_requirements[template_name].num_programs
        return template_name, num_programs

    elif spec.strategy == PromptStrategy.REEVO:
        # Phase-based selection
        phase = _get_reevo_phase(state)
        num_programs = spec.template_requirements[phase].num_programs
        return phase, num_programs

    raise ValueError(f"Unknown strategy: {spec.strategy}")


def _get_reevo_phase(state: dict | None) -> str:
    """Return current ReEvo phase from state dict.

    Args:
        state: Dict with 'num_programs' (int) and 'phase' (str). Managed by ProgramsDatabase.

    Returns:
        "seed", "crossover", or "mutation".
    """
    if state is None:
        return "seed"

    num_programs = state.get("num_programs", 0)

    # Seed phase when no programs exist
    if num_programs == 0:
        return "seed"

    # Return current phase from state (managed by ProgramsDatabase)
    # Default to crossover if phase not set
    return state.get("phase", "crossover")


def _format_function_code(
    func,
    scores: dict | None = None,
    spec: PromptSpec | None = None,
    docstring_template: str | None = None,
    version: int | None = None,
    include_def: bool = True,
    include_thought_tags: bool = False,
) -> str:
    """Format a Function object as code string, with docstring template and scores.

    Args:
        func: Function object with name, args, body, etc.
        scores: Optional scores dict for this function.
        spec: Optional PromptSpec with score display settings.
        docstring_template: Docstring template with {score} and {version} placeholders.
        version: Version number to replace {version} with (e.g., 0 for "priority_v0").
        include_def: Whether to include 'def name(args):' line.
        include_thought_tags: Whether to wrap output with <thought> and <code> tags.

    Returns:
        Formatted function code with placeholders replaced.
    """
    if include_def:
        ret_type = f" -> {func.return_type}" if func.return_type else ""
        header = f"def {func.name}({func.args}){ret_type}:"

        # Use docstring template if provided, otherwise fall back to func.docstring
        docstring_content = docstring_template if docstring_template else (func.docstring or "")

        # Replace {version} placeholder
        if version is not None:
            docstring_content = docstring_content.replace("{version}", str(version))

        # Replace {score} placeholder
        if "{score}" in docstring_content:
            if scores and spec and spec.show_scores:
                score_text = _format_scores(scores, spec)
                docstring_content = docstring_content.replace("{score}", score_text)
            else:
                # Remove {score} placeholder and any preceding newline
                docstring_content = docstring_content.replace("\n{score}", "")
                docstring_content = docstring_content.replace("{score}", "")

        # Clean up and format docstring
        docstring_content = docstring_content.strip()
        if docstring_content:
            # Handle multi-line docstrings (e.g., with scores)
            if '\n' in docstring_content:
                lines = docstring_content.split('\n')
                # Opening """ on own line, all content indented, closing """ on own line
                indented_lines = ['    ' + line for line in lines]
                docstring = '    """\n' + '\n'.join(indented_lines) + '\n    """'
            else:
                docstring = f'    """{docstring_content}"""'
            code = f"{header}\n{docstring}\n{func.body}"
        else:
            code = f"{header}\n{func.body}"
    else:
        code = func.body

    # Wrap with thought and code tags if requested
    if include_thought_tags:
        thought = getattr(func, 'thought', "") or ""
        return f"<thought>{thought}</thought>\n<code>{code}</code>"
    return code


def build_prompt(
    spec: PromptSpec,
    template_name: str,
    programs: list,  # list of (Function, scores_dict) tuples
    state: dict | None = None,
) -> str:
    """Build prompt by filling placeholders.

    Args:
        spec: Loaded prompt specification
        template_name: Which template to use
        programs: List of (Function, scores_dict) tuples, sorted by score (worse first)
        state: Optional state dict (for ReEvo reflections)

    Returns:
        Completed prompt string
    """
    if template_name not in spec.templates:
        raise ValueError(f"Template not found: {template_name}")

    template = spec.templates[template_name]

    # Static placeholders already filled in load_specification()
    # Now fill dynamic placeholders from sampled programs
    prompt = template

    # Strategy-specific placeholders
    if spec.strategy == PromptStrategy.FUNSEARCH:
        # FunSearch: {fewshot_examples} + {function_header} for code completion
        prompt = _fill_code_completion_placeholders(prompt, spec, programs)
    else:
        # EoH/ReEvo: {worse_code} + {better_code} for instruction-based prompts
        prompt = _fill_instruction_placeholders(prompt, spec, programs)

    # ReEvo reflection placeholders (always fill, use spec defaults if not in state)
    state = state or {}
    prompt = prompt.replace("{reflection}", state.get("reflection", ""))
    prompt = prompt.replace("{prior_reflection}", state.get("prior_reflection", ""))

    # new_reflections can be a list (from Database) or string. Convert list to formatted string.
    new_reflections = state.get("new_reflections", "")
    if isinstance(new_reflections, list):
        new_reflections = "\n- ".join(new_reflections) if new_reflections else ""
    prompt = prompt.replace("{new_reflections}", new_reflections)

    # Use spec.initial_reflection as default (can be empty or contain hints)
    prompt = prompt.replace("{initial_reflection}", state.get("initial_reflection", spec.initial_reflection))

    # Clean up empty placeholders and extra whitespace
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)

    return prompt.strip()


def _fill_instruction_placeholders(
    prompt: str,
    spec: PromptSpec,
    programs: list,
) -> str:
    """Fill placeholders for instruction-based models (EoH/ReEvo).

    Builds:
        {worse_code}: priority_v0 (baseline docstring)
        {better_code}: priority_v1 (improved docstring)
        {thought}: from better function
    """
    worse_func, worse_scores = None, {}
    better_func, better_scores = None, {}

    # Version the function names: worse=v0, better=v1 (or single=v0)
    if len(programs) >= 1:
        func, scores = programs[-1]
        better_func = copy.copy(func)
        better_scores = scores
        better_func.name = f"{spec.function_to_evolve}_v0" if len(programs) == 1 else f"{spec.function_to_evolve}_v1"

    if len(programs) >= 2:
        func, scores = programs[0]
        worse_func = copy.copy(func)
        worse_scores = scores
        worse_func.name = f"{spec.function_to_evolve}_v0"

    # EoH uses <thought>/<code> tags to demonstrate expected output format
    use_tags = spec.strategy == PromptStrategy.EOH

    # Format code with docstrings
    if len(programs) == 1:
        better_code = _format_function_code(better_func, better_scores, spec, spec.docstring_baseline, include_thought_tags=use_tags)
    else:
        better_code = _format_function_code(better_func, better_scores, spec, spec.docstring_improved, version=0, include_thought_tags=use_tags)

    worse_code = _format_function_code(worse_func, worse_scores, spec, spec.docstring_baseline, include_thought_tags=use_tags) if worse_func else ""
    thought = getattr(better_func, 'thought', "") or "" if better_func else ""

    prompt = prompt.replace("{worse_code}", worse_code)
    prompt = prompt.replace("{better_code}", better_code)
    prompt = prompt.replace("{thought}", thought)

    return prompt


def _fill_code_completion_placeholders(
    prompt: str,
    spec: PromptSpec,
    programs: list,
) -> str:
    """Fill placeholders for code completion models (FunSearch).

    Builds:
        {fewshot_examples}: priority_v0, priority_v1, ... (all examples combined)
        {function_header}: def priority_vN(...): for model to complete
    """
    # Build fewshot examples with versioned function names
    fewshot_parts = []
    for i, (func, scores) in enumerate(programs):
        func_copy = copy.deepcopy(func)
        func_copy.name = f"{spec.function_to_evolve}_v{i}"
        # First version uses baseline docstring, subsequent use improved (referencing previous version)
        docstring_template = spec.docstring_baseline if i == 0 else spec.docstring_improved
        prev_version = i - 1 if i > 0 else None
        fewshot_parts.append(_format_function_code(func_copy, scores, spec, docstring_template, prev_version))

    fewshot_examples = "\n\n".join(fewshot_parts)

    # Build function header for next version (for code completion)
    next_version = len(programs)
    header_docstring = spec.docstring_improved.replace("{version}", str(next_version - 1))
    header_docstring = header_docstring.replace("\n{score}", "").replace("{score}", "").strip()
    function_header = (
        f"def {spec.function_to_evolve}_v{next_version}({spec.function_args}) -> {spec.return_type}:\n"
        f'    """{header_docstring}"""'
    )

    # Fill FunSearch-specific placeholders
    prompt = prompt.replace("{fewshot_examples}", fewshot_examples)
    prompt = prompt.replace("{function_header}", function_header)
    prompt = prompt.replace("{version}", str(next_version))
    prompt = prompt.replace("{evaluation_preamble}", spec.evaluation_preamble)
    prompt = prompt.replace("{evaluation_script}", spec.evaluation_script)

    return prompt


def get_system_message(spec: PromptSpec, is_reflector: bool = False) -> str | None:
    """Get appropriate system message.

    Args:
        spec: Loaded prompt specification
        is_reflector: If True, return reflector system message (ReEvo only)

    Returns:
        System message string or None
    """
    if is_reflector and spec.reflector_system_message:
        return spec.reflector_system_message
    return spec.system_message


def build_reevo_prompts(
    spec: PromptSpec,
    phase: str,
    programs: list,
    state: dict | None = None,
) -> tuple[str | None, str]:
    """Build two prompts for ReEvo (reflection + generation).

    ReEvo uses two LLM calls:
    1. Reflection: Generate insights about the code
    2. Generation: Use reflection to generate improved code

    Args:
        spec: Loaded prompt specification
        phase: ReEvo phase ("seed", "crossover", or "mutation")
        programs: List of (Function, scores_dict) tuples
        state: Optional state dict with prior reflections

    Returns:
        Tuple of (reflection_prompt, generation_prompt).
        For seed phase, reflection_prompt is None.
        The generation_prompt keeps {reflection} placeholder for sampler to fill.
    """
    state = state or {}

    if phase == "seed":
        # Seed phase: single generation stage, no reflection needed
        # Use seed template (or fall back to mutation template)
        generation_template = spec.templates.get("seed", spec.templates.get("mutation", ""))
        generation_prompt = _fill_reevo_placeholders(generation_template, spec, programs, state)
        # For seed phase, fill {reflection} with initial_reflection directly
        generation_prompt = generation_prompt.replace(
            "{reflection}",
            state.get("initial_reflection", spec.initial_reflection)
        )
        return (None, generation_prompt)

    # Crossover phase: short-term reflection + crossover generation
    if phase == "crossover":
        reflection_template_name = "reflect_st"
        generation_template_name = "crossover"
    # Mutation phase: long-term reflection + mutation generation
    else:  # mutation
        reflection_template_name = "reflect_lt"
        generation_template_name = "mutation"

    # Get templates
    reflection_template = spec.templates.get(reflection_template_name, "")
    generation_template = spec.templates.get(generation_template_name, "")

    if not reflection_template:
        logger.warning(f"ReEvo: Missing template '{reflection_template_name}', falling back to single stage")
        generation_prompt = _fill_reevo_placeholders(generation_template, spec, programs, state)
        return (None, generation_prompt)

    # Build reflection prompt (Stage 1)
    reflection_prompt = _fill_reevo_placeholders(reflection_template, spec, programs, state)

    # Build generation prompt (Stage 2), leave {reflection} unfilled
    # The sampler will fill it with the output from Stage 1
    generation_prompt = _fill_reevo_placeholders(generation_template, spec, programs, state)

    return (reflection_prompt, generation_prompt)


def _fill_reevo_placeholders(
    template: str,
    spec: PromptSpec,
    programs: list,
    state: dict,
) -> str:
    """Fill ReEvo-specific placeholders in a template.

    Handles: {worse_code}, {better_code}, {prior_reflection}, {new_reflections},
             {initial_reflection}, {imports}

    Note: {reflection} is left unfilled, the sampler fills it with Stage 1 output.
    """
    prompt = template

    # Fill code placeholders
    prompt = _fill_instruction_placeholders(prompt, spec, programs)

    # Fill reflection state placeholders
    prompt = prompt.replace("{prior_reflection}", state.get("prior_reflection", ""))

    # new_reflections can be a list (from Database) or string. Convert list to formatted string.
    new_reflections = state.get("new_reflections", "")
    if isinstance(new_reflections, list):
        new_reflections = "\n- ".join(new_reflections) if new_reflections else ""
    prompt = prompt.replace("{new_reflections}", new_reflections)

    prompt = prompt.replace("{initial_reflection}", state.get("initial_reflection", spec.initial_reflection))

    # Clean up empty placeholders and extra whitespace
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)

    return prompt.strip()
