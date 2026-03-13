"""Prompt building module for FunSearch, EoH, and ReEvo strategies.

Flow:
    1. load_specification()  Called once at startup, loads templates and content into PromptSpec
    2. select_template()     Called each iteration, returns (template_name, num_programs_needed)
    3. build_prompt()        Called each iteration, fills placeholders and returns prompt string

Placeholders:
    Filled once in load_specification() (static content from spec files):
        {imports}, {problem_description}, {problem_desc}, {func_desc}, {user_generator}

    Variant placeholders (filled if variant specified, e.g., for ECC folder):
        {string_type}        "binary" or "q-ary"
        {edge_condition}     Graph connectivity rule
        {error_type}         "deletion" or "insertion/deletion/substitution"

    Filled each iteration in build_prompt() (from sampled programs).
    All program placeholders are available to all strategies, usage depends on template:
        {fewshot_examples}   All programs as versioned functions (v0, v1, ...)
        {worse_code}         First program as v0 (when 2 programs provided)
        {better_code}        Last program (v0 or v1 depending on count)
        {function_header}    Next function header for code completion
        {function_signature} Full function signature (def line with parameters)
        {evaluation_script}  Evaluation script content (how functions are scored)
        {version}            Next version number

    Note: {fewshot_examples}, {worse_code}, {better_code} include <code> tags when detected
    in template. <description> tags are included only if detected in template and enabled via
    fewshot_include_description config option.

    ReEvo only (reflection state):
        {reflection}         Filled by sampler with first LLM output, not here
        {prior_reflection}   Previous long term reflection
        {new_reflections}    New short term reflections from crossover phase
        {initial_reflection} Starting reflection from spec
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
    templates: dict[str, str] = field(default_factory=dict)  # template_name to content
    template_requirements: dict[str, TemplateRequirements] = field(default_factory=dict)

    # FunSearch specific
    fewshot_num_examples: int = 2
    fewshot_include_description: bool = True  # Include <description> in few-shot examples
    evaluation_script: str = ""
    function_to_evolve: str = "priority"
    function_args: str = ""
    return_type: str = "float"
    is_multi_turn: bool = False  # True if using two stage prompts (stage1 + stage2)

    # ReEvo specific
    user_generator: str = ""  # Prebuilt from user_generator.txt
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


def _load_variant(spec_dir: Path, variant_name: str) -> dict[str, str]:
    """Load variant placeholders from variants.py.

    Args:
        spec_dir: Base directory containing variants.py
        variant_name: Name of variant to load (e.g., "deletions", "ids")

    Returns:
        Dict of placeholder name to value (e.g., {"string_type": "binary", ...})

    Raises:
        ValueError: If variants.py not found or variant_name not in VARIANTS
    """
    variants_path = spec_dir / "variants.py"
    if not variants_path.exists():
        raise ValueError(f"variants.py not found in {spec_dir}")

    # Load variants module
    import importlib.util
    spec = importlib.util.spec_from_file_location("variants", variants_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get VARIANTS dict
    if not hasattr(module, "VARIANTS"):
        raise ValueError(f"VARIANTS dict not found in {variants_path}")

    variants = module.VARIANTS
    if variant_name not in variants:
        raise ValueError(f"Variant '{variant_name}' not found. Available: {list(variants.keys())}")

    return variants[variant_name]


def _apply_variant(text: str, variant: dict[str, str]) -> str:
    """Apply variant placeholders to text.

    Args:
        text: Text with {placeholder} markers
        variant: Dict mapping placeholder names to values

    Returns:
        Text with placeholders replaced
    """
    for key, value in variant.items():
        text = text.replace(f"{{{key}}}", value)
    return text


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
    has_reflection = "{reflection}" in template or "{prior_reflection}" in template

    # If template uses {worse_code}, it needs 2 programs
    # Otherwise just 1 (the {better_code})
    num_programs = 2 if has_worse_code else 1

    return TemplateRequirements(
        num_programs=num_programs,
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
        scores: Dict mapping (n, s) or string keys to score values.
        spec: PromptSpec with score display settings.

    Returns:
        Formatted string like "Scores: {(6, 1): 8, (7, 1): 14}" or
        "Relative to baseline: {(6, 1): +0.0%, (7, 1): +7.1%}".
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
    variant: str | None = None,  # ECC variant: "deletions" or "ids" (loads from variants.py)
    # FunSearch options
    funsearch_template: str = "funsearch/template.txt",
    funsearch_problem_desc: str = "funsearch/problem_descriptions/baseline.txt",
    funsearch_string_hint: str | None = None,  # Optional hint file, fills {string_hint} placeholder
    funsearch_system_message: str | None = None,
    funsearch_evaluation_script: str | None = None,
    fewshot_num_examples: int = 2,
    fewshot_include_description: bool = True,
    initial_functions_dir: str = "initial_functions/graph",
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
        ... (strategy specific options)

    Returns:
        PromptSpec with all content loaded
    """
    base = Path(spec_dir)

    # Load variant placeholders if specified
    variant_dict = {}
    if variant:
        variant_dict = _load_variant(base, variant)
        logger.info(f"Loaded variant '{variant}': {list(variant_dict.keys())}")

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
    evaluation_script = ""
    function_args = ""
    return_type = "float"

    is_multi_turn = False

    if strategy == PromptStrategy.FUNSEARCH:
        # Load template(s). Supports single file, single turn folder, or multi turn folder
        template_path = base / funsearch_template
        if template_path.is_dir():
            # Check if this is a multi-turn template (has stage1.txt and stage2.txt)
            if (template_path / "stage1.txt").exists() and (template_path / "stage2.txt").exists():
                # Multi turn: load stage1 (reflection) and stage2 (generation) templates
                is_multi_turn = True
                stage1_template = _load_file(template_path / "stage1.txt").strip()
                stage2_template = _load_file(template_path / "stage2.txt").strip()
                templates = {
                    "funsearch_stage1": stage1_template,
                    "funsearch": stage2_template,
                }
                # Load single function variants if they exist (optional, no warning if missing)
                stage1_single_path = template_path / "stage1_single.txt"
                stage2_single_path = template_path / "stage2_single.txt"
                if stage1_single_path.exists():
                    templates["funsearch_stage1_single"] = stage1_single_path.read_text().strip()
                if stage2_single_path.exists():
                    templates["funsearch_single"] = stage2_single_path.read_text().strip()
            else:
                # Single turn folder: load default.txt and single.txt variants
                default_template = _load_file(template_path / "default.txt").strip()
                templates = {"funsearch": default_template}
                # Load single function variant if it exists (optional, no warning if missing)
                single_path = template_path / "single.txt"
                if single_path.exists():
                    templates["funsearch_single"] = single_path.read_text().strip()
        else:
            # Single file (legacy behavior)
            templates = {"funsearch": _load_file(template_path).strip()}
        problem_description = _load_file(base / funsearch_problem_desc).strip()
        string_hint = _load_file(base / funsearch_string_hint).strip() if funsearch_string_hint else ""
        problem_description = problem_description.replace("{string_hint}", string_hint)

        if funsearch_system_message:
            system_message = _load_file(base / funsearch_system_message).strip()

        if funsearch_evaluation_script:
            evaluation_script = _load_file(base / funsearch_evaluation_script)


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

        # Prebuild user_generator
        user_gen_template = templates.get("user_generator", "")
        if user_gen_template:
            user_generator = user_gen_template.replace("{problem_desc}", problem_desc)
            user_generator = user_generator.replace("{func_desc}", func_desc)
            user_generator = user_generator.replace("{imports}", imports)
            # Remove user_generator from templates (it's a component, not a prompt template)
            del templates["user_generator"]

    # Extract function signature from initial function (used by all strategies)
    initial_func_dir = base / initial_functions_dir
    if initial_func_dir.exists():
        initial_func_path = next(initial_func_dir.glob("*.txt"), None)
        if initial_func_path:
            function_args, return_type = _extract_function_signature(initial_func_path)
    function_signature = f"priority({function_args}) -> {return_type}"

    # Infer requirements for each template
    template_requirements = {
        name: _infer_requirements(content)
        for name, content in templates.items()
    }

    # Note: samples_per_prompt for specific templates (e.g., mutation) can be set
    # after loading via: spec.template_requirements["mutation"].samples_per_prompt = N

    # Prefill static placeholders in templates
    for name in templates:
        templates[name] = templates[name].replace("{imports}", imports)
        templates[name] = templates[name].replace("{problem_description}", problem_description)
        templates[name] = templates[name].replace("{problem_desc}", problem_desc)
        templates[name] = templates[name].replace("{func_desc}", func_desc)
        templates[name] = templates[name].replace("{function_signature}", function_signature)
        templates[name] = templates[name].replace("{user_generator}", user_generator)

    # Load docstring templates
    docstring_baseline_content = _load_file(base / docstring_baseline).strip()
    docstring_improved_content = _load_file(base / docstring_improved).strip()
    score_label_absolute_content = _load_file(base / score_label_absolute).strip()
    score_label_relative_content = _load_file(base / score_label_relative).strip()

    # Apply variant placeholders if specified
    if variant_dict:
        problem_description = _apply_variant(problem_description, variant_dict)
        problem_desc = _apply_variant(problem_desc, variant_dict)
        func_desc = _apply_variant(func_desc, variant_dict)
        if system_message:
            system_message = _apply_variant(system_message, variant_dict)
        if reflector_system_message:
            reflector_system_message = _apply_variant(reflector_system_message, variant_dict)
        initial_reflection = _apply_variant(initial_reflection, variant_dict)
        user_generator = _apply_variant(user_generator, variant_dict)
        for name in templates:
            templates[name] = _apply_variant(templates[name], variant_dict)

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
        fewshot_include_description=fewshot_include_description,
        evaluation_script=evaluation_script,
        function_to_evolve="priority",
        function_args=function_args,
        return_type=return_type,
        is_multi_turn=is_multi_turn,
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
        variant=getattr(config.prompt, 'variant', None),
        funsearch_template=config.prompt.funsearch_template,
        funsearch_problem_desc=config.prompt.funsearch_problem_desc,
        funsearch_string_hint=getattr(config.prompt, 'funsearch_string_hint', None),
        funsearch_system_message=config.prompt.funsearch_system_message,
        funsearch_evaluation_script=getattr(config.prompt, 'funsearch_evaluation_script', None),
        fewshot_num_examples=config.prompt.fewshot_num_examples,
        fewshot_include_description=config.prompt.fewshot_include_description,
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
        # Score display options
        show_scores=config.prompt.show_scores,
        score_display_mode=config.prompt.score_display_mode,
        best_known_solutions=config.prompt.best_known_solutions,
        # Docstring templates
        docstring_baseline=config.prompt.docstring_baseline,
        docstring_improved=config.prompt.docstring_improved,
        score_label_absolute=config.prompt.score_label_absolute,
        score_label_relative=config.prompt.score_label_relative,
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
        # Phase based selection
        phase = _get_reevo_phase(state)
        num_programs = spec.template_requirements[phase].num_programs
        return phase, num_programs

    raise ValueError(f"Unknown strategy: {spec.strategy}")


def select_funsearch_variant(spec: PromptSpec, available_programs: int) -> str:
    """Select appropriate FunSearch template variant based on available programs.

    Args:
        spec: Loaded prompt specification
        available_programs: Number of programs actually sampled

    Returns:
        Template name to use ("funsearch" or "funsearch_single")
    """
    if available_programs == 1 and "funsearch_single" in spec.templates:
        return "funsearch_single"
    return "funsearch"


def build_funsearch_prompts(
    spec: PromptSpec,
    programs: list,
) -> tuple[str | None, str]:
    """Build prompts for FunSearch (single-turn or multi-turn).

    Args:
        spec: Loaded prompt specification
        programs: List of (Function, scores_dict) tuples

    Returns:
        Tuple of (reflection_prompt, generation_prompt).
        For single-turn, reflection_prompt is None.
        For multi-turn, generation_prompt keeps {reflection} placeholder for sampler to fill.
    """
    available_programs = len(programs)

    if not spec.is_multi_turn:
        # Single-turn: just build generation prompt
        template_name = select_funsearch_variant(spec, available_programs)
        generation_prompt = build_prompt(spec, template_name, programs)
        return (None, generation_prompt)

    # Multi turn: build stage1 (reflection) and stage2 (generation) prompts
    # Select appropriate templates based on available programs
    if available_programs == 1 and "funsearch_stage1_single" in spec.templates:
        stage1_template_name = "funsearch_stage1_single"
        stage2_template_name = "funsearch_single"
    else:
        stage1_template_name = "funsearch_stage1"
        stage2_template_name = "funsearch"

    # Build stage1 prompt (reflection/thought)
    stage1_prompt = build_prompt(spec, stage1_template_name, programs)

    # Build stage2 template (keep {reflection} placeholder for sampler to fill)
    stage2_template = spec.templates.get(stage2_template_name, "")
    # Fill program placeholders but NOT {reflection}
    stage2_prompt = _fill_program_placeholders(stage2_template, spec, programs)

    return (stage1_prompt, stage2_prompt)


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
    include_description_tags: bool = False,
    include_code_tags: bool = False,
) -> str:
    """Format a Function object as code string, with docstring template and scores.

    Args:
        func: Function object with name, args, body, etc.
        scores: Optional scores dict for this function.
        spec: Optional PromptSpec with score display settings.
        docstring_template: Docstring template with {score} and {version} placeholders.
        version: Version number to replace {version} with (e.g., 0 for "priority_v0").
        include_def: Whether to include 'def name(args):' line.
        include_description_tags: Whether to wrap with <description> tags (implies include_code_tags).
        include_code_tags: Whether to wrap code with <code> tags.

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

        # Append baseline note for initial seed functions (generation == 0)
        # This clarifies when a simple baseline outperforms a more complex function
        is_seed = getattr(func, 'generation', None) == 0
        if is_seed and version is not None:
            # Only append for non-v0 positions (where "Improved version" docstring is used)
            docstring_content += " Simple baseline that assigns equal priority to all nodes."

        # Clean up and format docstring
        docstring_content = docstring_content.strip()
        if docstring_content:
            # Handle multiline docstrings
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

    # Wrap with tags if requested
    if include_description_tags or include_code_tags:
        parts = []
        if include_description_tags:
            description = getattr(func, 'description', "") or ""
            parts.append(f"<description>{description}</description>")
        if include_description_tags or include_code_tags:
            parts.append(f"<code>{code}</code>")
            return "\n".join(parts)
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

    # Fill program placeholders (unified for all strategies)
    prompt = _fill_program_placeholders(prompt, spec, programs)

    # Fill reflection placeholders (ReEvo, does nothing for other strategies)
    state = state or {}
    prompt = _fill_reflection_placeholders(prompt, spec, state, fill_reflection=True)

    # Clean up empty placeholders and extra whitespace
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)

    return prompt.strip()


def _fill_reflection_placeholders(
    prompt: str,
    spec: PromptSpec,
    state: dict,
    fill_reflection: bool = True,
) -> str:
    """Fill reflection-related placeholders.

    Args:
        prompt: Template with placeholders
        spec: Loaded prompt specification
        state: State dict with reflection values
        fill_reflection: Whether to fill {reflection} (False for two stage ReEvo
                         where sampler fills it with Stage 1 output)

    Returns:
        Prompt with reflection placeholders filled.
    """
    if fill_reflection:
        prompt = prompt.replace("{reflection}", state.get("reflection", ""))

    prompt = prompt.replace("{prior_reflection}", state.get("prior_reflection", ""))

    # new_reflections can be a list (from Database) or string
    new_reflections = state.get("new_reflections", "")
    if isinstance(new_reflections, list):
        new_reflections = "\n- ".join(new_reflections) if new_reflections else ""
    prompt = prompt.replace("{new_reflections}", new_reflections)

    prompt = prompt.replace("{initial_reflection}", state.get("initial_reflection", spec.initial_reflection))

    return prompt


def _fill_program_placeholders(
    prompt: str,
    spec: PromptSpec,
    programs: list,
) -> str:
    """Fill all program-related placeholders (unified for FunSearch/EoH/ReEvo).

    Detects tag format from system message (if exists) or falls back to template.
    If <code> or <description> tags are found, few-shot examples are wrapped accordingly.
    Whether to include description in few-shot examples is controlled by
    spec.fewshot_include_description.

    Fills:
        {fewshot_examples}   All programs as versioned functions (v0, v1, ...)
        {worse_code}         First program as v0 (for EoH/ReEvo with 2 programs)
        {better_code}        Last program as v1 (for EoH/ReEvo)
        {function_header}    Next function header for completion
        {version}            Next version number
        {evaluation_script}  FunSearch evaluation context
    """
    # Detect tag format from system message (if exists) or fall back to template
    check_source = spec.system_message if spec.system_message else prompt
    include_description_tags = "<description>" in check_source and spec.fewshot_include_description
    include_code_tags = "<code>" in check_source

    # Build {fewshot_examples}
    fewshot_parts = []
    for i, (func, scores) in enumerate(programs):
        func_copy = copy.deepcopy(func)
        func_copy.name = f"{spec.function_to_evolve}_v{i}"
        docstring_template = spec.docstring_baseline if i == 0 else spec.docstring_improved
        prev_version = i - 1 if i > 0 else None
        fewshot_parts.append(_format_function_code(
            func_copy, scores, spec, docstring_template, prev_version,
            include_description_tags=include_description_tags,
            include_code_tags=include_code_tags,
        ))
    fewshot_examples = "\n\n".join(fewshot_parts)

    # Build {worse_code} and {better_code}
    worse_code = ""
    better_code = ""

    if len(programs) >= 1:
        func, scores = programs[-1]
        better_func = copy.deepcopy(func)
        better_func.name = f"{spec.function_to_evolve}_v0" if len(programs) == 1 else f"{spec.function_to_evolve}_v1"
        docstring = spec.docstring_baseline if len(programs) == 1 else spec.docstring_improved
        version = None if len(programs) == 1 else 0
        better_code = _format_function_code(
            better_func, scores, spec, docstring, version,
            include_description_tags=include_description_tags,
            include_code_tags=include_code_tags,
        )

    if len(programs) >= 2:
        func, scores = programs[0]
        worse_func = copy.deepcopy(func)
        worse_func.name = f"{spec.function_to_evolve}_v0"
        worse_code = _format_function_code(
            worse_func, scores, spec, spec.docstring_baseline, None,
            include_description_tags=include_description_tags,
            include_code_tags=include_code_tags,
        )

    # Build {function_header}
    next_version = len(programs)
    header_docstring = spec.docstring_improved.replace("{version}", str(next_version - 1))
    header_docstring = header_docstring.replace("\n{score}", "").replace("{score}", "").strip()
    function_header = (
        f"def {spec.function_to_evolve}_v{next_version}({spec.function_args}) -> {spec.return_type}:\n"
        f'    """{header_docstring}"""'
    )

    # Build {function_signature} from extracted initial function signature
    function_signature = f"{spec.function_to_evolve}({spec.function_args}) -> {spec.return_type}"

    # Fill all placeholders
    prompt = prompt.replace("{fewshot_examples}", fewshot_examples)
    prompt = prompt.replace("{worse_code}", worse_code)
    prompt = prompt.replace("{better_code}", better_code)
    prompt = prompt.replace("{function_header}", function_header)
    prompt = prompt.replace("{function_signature}", function_signature)
    prompt = prompt.replace("{version}", str(next_version))
    prompt = prompt.replace("{evaluation_script}", spec.evaluation_script)
    prompt = prompt.replace("{version}", str(next_version))  # Also replace {version} in evaluation_script

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

    # Crossover phase, short term reflection and crossover generation
    if phase == "crossover":
        reflection_template_name = "reflect_st"
        generation_template_name = "crossover"
    # Mutation phase, long term reflection and mutation generation
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
    prompt = _fill_program_placeholders(prompt, spec, programs)

    # Fill reflection placeholders (but not {reflection}, sampler fills that)
    prompt = _fill_reflection_placeholders(prompt, spec, state, fill_reflection=False)

    # Clean up empty placeholders and extra whitespace
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)

    return prompt.strip()
