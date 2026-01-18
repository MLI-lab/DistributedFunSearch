# Prompt Building Refactor Plan

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           __main__.py                               │
│  1. Load config                                                     │
│  2. prompt_spec = prompt_builder.load_specification(config)         │
│  3. Create ProgramsDatabase(prompt_spec, ...)                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      prompt_builder.py                              │
│                      (pure utility module - no state, no DB)        │
│                                                                     │
│  load_specification(config) → PromptSpec                            │
│      - Loads ALL templates, placeholders once                       │
│      - Infers template requirements from placeholders               │
│      - Returns immutable PromptSpec object                          │
│                                                                     │
│  select_template(spec, strategy, state) → (template_name, num_progs)│
│      - FunSearch: returns fixed template                            │
│      - EoH: random style selection from styles/                     │
│      - ReEvo: phase-based selection                                 │
│      - Returns which template + how many programs needed            │
│                                                                     │
│  build_prompt(spec, template_name, programs) → str                  │
│      - Pure function: spec + template + programs → prompt string    │
│      - Fills {better_code}, {worse_code}, {thought}, etc.           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     programs_database.py                            │
│                                                                     │
│  __init__(self, prompt_spec, ...):                                  │
│      self.prompt_spec = prompt_spec  # Already loaded               │
│                                                                     │
│  _generate_prompt_for_island(island):                               │
│      1. template_name, num_needed = select_template(spec, strategy) │
│      2. programs = self._sample_programs(island, num_needed)        │
│      3. prompt = build_prompt(spec, template_name, programs)        │
│      return prompt                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

```python
class PromptStrategy(Enum):
    FUNSEARCH = "funsearch"
    EOH = "eoh"
    REEVO = "reevo"


@dataclass
class TemplateRequirements:
    """Requirements inferred from template placeholders."""
    num_programs: int          # 1 if only {better_code}, 2 if also {worse_code}
    needs_thought: bool        # True if {thought} in template
    needs_reflection: bool     # True if {reflection} in template


@dataclass
class PromptSpec:
    """Immutable specification loaded once at startup."""
    strategy: PromptStrategy

    # Shared content (loaded from files)
    imports: str
    problem_description: str  # For funsearch, eoh
    problem_desc: str         # For reevo (short inline version)
    func_desc: str
    system_message: str | None

    # Templates
    templates: dict[str, str]  # template_name → template_content
    template_requirements: dict[str, TemplateRequirements]  # Inferred from placeholders

    # ReEvo-specific
    user_generator: str | None  # Pre-built from user_generator.txt
    reflector_system_message: str | None  # Separate system message for reflector
```

---

## Template Requirements Detection (Option B)

Automatically infer requirements by parsing template for placeholders:

```python
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
```

---

## Strategy Requirements Summary

### FunSearch
- **Templates**: Single `funsearch/template.txt`
- **Programs**: Configurable via `fewshot_num_examples` (default 2)
- **Code format**: Versioned functions `priority_v0`, `priority_v1`, ...
- **Special placeholders**: `{fewshot_examples}`, `{function_header}`, `{version}`
- **Output**: Code completion (raw Python)

### EoH (Evolution of Heuristics)
- **Templates**: Multiple in `eoh/styles/` (i1, e1, e2, m1, m2, m3)
- **Programs**: 1-2 (inferred from template)
- **Code format**: Raw function code (not versioned)
- **Special placeholders**: `{better_code}`, `{worse_code}`, `{thought}`
- **Output**: `<thought>...</thought><code>...</code>`
- **Selection**: Random from available styles

### ReEvo
- **Templates**: Multiple in `reevo/templates/` (seed, crossover, mutation, reflect_st, reflect_lt)
- **Programs**: 1-2 (inferred from template)
- **Code format**: Raw function code
- **Special placeholders**: `{better_code}`, `{worse_code}`, `{reflection}`, `{prior_reflection}`, `{new_reflections}`, `{user_generator}`
- **Output**: ` ```python ... ``` `
- **Selection**: Phase-based (seed → crossover/mutation, periodic reflection)
- **State**: Reflection history tracked in ProgramsDatabase

---

## Responsibility Separation

| Concern | Location | When |
|---------|----------|------|
| Load templates/placeholders | `prompt_builder.load_specification()` | Once at startup |
| Infer template requirements | `prompt_builder._infer_requirements()` | Once at startup |
| Template selection logic | `prompt_builder.select_template()` | Per prompt |
| Program sampling from DB | `ProgramsDatabase._sample_programs()` | Per prompt |
| Placeholder filling | `prompt_builder.build_prompt()` | Per prompt |
| ReEvo reflection state | `ProgramsDatabase.reevo_state` | Persistent |

---

## Config Structure

```python
@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    strategy: PromptStrategy = PromptStrategy.FUNSEARCH
    spec_dir: str = ".../Deletions"  # Base directory for all spec files

    # Shared paths (relative to spec_dir)
    imports_file: str = "imports/networkx.txt"
    system_message_file: str | None = None

    # FunSearch-specific
    funsearch_template: str = "funsearch/template.txt"
    funsearch_problem_desc: str = "funsearch/problem_descriptions/baseline.txt"
    fewshot_num_examples: int = 2
    fewshot_show_thinking: bool = False
    fewshot_show_thought: bool = False

    # EoH-specific
    eoh_styles_dir: str = "eoh/styles"
    eoh_problem_desc: str = "eoh/problem_descriptions/baseline.txt"
    eoh_func_desc: str = "eoh/components/func_desc.txt"
    eoh_system_message: str = "eoh/system_message.txt"

    # ReEvo-specific
    reevo_templates_dir: str = "reevo/templates"
    reevo_problem_desc: str = "reevo/problem/problem_desc.txt"
    reevo_func_desc: str = "reevo/problem/func_desc.txt"
    reevo_generator_system: str = "reevo/system/generator.txt"
    reevo_reflector_system: str = "reevo/system/reflector.txt"
```

---

## Implementation Flow

### load_specification(config) → PromptSpec

```python
def load_specification(config: PromptConfig) -> PromptSpec:
    """Load all templates and content once at startup."""
    spec_dir = Path(config.spec_dir)

    # Load shared content
    imports = _load_file(spec_dir / config.imports_file)

    # Load strategy-specific content
    if config.strategy == PromptStrategy.FUNSEARCH:
        templates = {"funsearch": _load_file(spec_dir / config.funsearch_template)}
        problem_description = _load_file(spec_dir / config.funsearch_problem_desc)
        system_message = None  # Code models don't use system messages
        ...

    elif config.strategy == PromptStrategy.EOH:
        templates = _load_directory(spec_dir / config.eoh_styles_dir)  # {name: content}
        problem_description = _load_file(spec_dir / config.eoh_problem_desc)
        func_desc = _load_file(spec_dir / config.eoh_func_desc)
        system_message = _load_file(spec_dir / config.eoh_system_message)
        ...

    elif config.strategy == PromptStrategy.REEVO:
        templates = _load_directory(spec_dir / config.reevo_templates_dir)
        problem_desc = _load_file(spec_dir / config.reevo_problem_desc)
        func_desc = _load_file(spec_dir / config.reevo_func_desc)
        system_message = _load_file(spec_dir / config.reevo_generator_system)
        reflector_system_message = _load_file(spec_dir / config.reevo_reflector_system)
        user_generator = _build_user_generator(...)  # Pre-fill {problem_desc}, {func_desc}, {imports}
        ...

    # Infer requirements for each template
    template_requirements = {
        name: _infer_requirements(content)
        for name, content in templates.items()
    }

    return PromptSpec(
        strategy=config.strategy,
        templates=templates,
        template_requirements=template_requirements,
        imports=imports,
        problem_description=problem_description,
        ...
    )
```

### select_template(spec, state) → (template_name, num_programs)

```python
def select_template(spec: PromptSpec, state: dict | None = None) -> tuple[str, int]:
    """Select template and return how many programs are needed."""

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
        phase = _get_reevo_phase(state)  # seed, crossover, mutation, reflect_st, reflect_lt
        num_programs = spec.template_requirements[phase].num_programs
        return phase, num_programs
```

### build_prompt(spec, template_name, programs) → str

```python
def build_prompt(
    spec: PromptSpec,
    template_name: str,
    programs: list[tuple[Function, dict]]
) -> str:
    """Build prompt by filling placeholders. Pure function."""
    template = spec.templates[template_name]
    requirements = spec.template_requirements[template_name]

    # Sort programs by score (worse first, better last)
    sorted_programs = sorted(programs, key=lambda p: _get_score(p[1]))

    # Format code blocks
    better_code = _format_function(sorted_programs[-1][0])
    worse_code = _format_function(sorted_programs[0][0]) if len(sorted_programs) > 1 else ""
    thought = sorted_programs[-1][0].thought or ""

    # Fill placeholders
    prompt = template
    prompt = prompt.replace("{problem_description}", spec.problem_description or "")
    prompt = prompt.replace("{problem_desc}", spec.problem_desc or "")
    prompt = prompt.replace("{func_desc}", spec.func_desc or "")
    prompt = prompt.replace("{imports}", spec.imports or "")
    prompt = prompt.replace("{better_code}", better_code)
    prompt = prompt.replace("{worse_code}", worse_code)
    prompt = prompt.replace("{thought}", thought)
    prompt = prompt.replace("{user_generator}", spec.user_generator or "")

    # ReEvo reflection placeholders (from state)
    if requirements.needs_reflection:
        prompt = prompt.replace("{reflection}", state.get("reflection", ""))
        prompt = prompt.replace("{prior_reflection}", state.get("prior_reflection", ""))
        prompt = prompt.replace("{new_reflections}", state.get("new_reflections", ""))
        prompt = prompt.replace("{initial_reflection}", state.get("initial_reflection", ""))

    # Clean up empty placeholders and extra whitespace
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)

    return prompt.strip()
```

---

## Integration with ProgramsDatabase

```python
class ProgramsDatabase:
    def __init__(self, prompt_spec: PromptSpec, ...):
        self.prompt_spec = prompt_spec
        self.reevo_state = {}  # For ReEvo reflection tracking

    def _generate_prompt_for_island(self, island) -> tuple[str, bool, int, list[int]]:
        # 1. Select template (may use reevo_state for phase selection)
        template_name, num_needed = prompt_builder.select_template(
            self.prompt_spec,
            state=self.reevo_state
        )

        # 2. Sample programs from island (existing logic, but with num_needed)
        sampled_programs = self._sample_programs_from_clusters(island, num_needed)

        # 3. Build prompt
        prompt = prompt_builder.build_prompt(
            self.prompt_spec,
            template_name,
            sampled_programs,
            state=self.reevo_state
        )

        # 4. Track metadata
        parent_ids = [p.program_id for p, _ in sampled_programs if p.program_id]
        version = len(sampled_programs)

        return prompt, False, version, parent_ids
```

---

## File Structure After Refactor

```
src/disfun/
├── prompt_builder.py         # Pure utility module (new)
│   ├── load_specification()
│   ├── select_template()
│   ├── build_prompt()
│   └── _infer_requirements()
├── programs_database.py      # Uses prompt_builder functions
├── specification_loader.py   # Keep for backward compat, or merge into prompt_builder
└── __main__.py               # Calls load_specification() at startup
```

---

## Implementation Steps

### Phase 1: Create prompt_builder.py
1. Create new `prompt_builder.py` module
2. Implement `PromptSpec` and `TemplateRequirements` dataclasses
3. Implement `load_specification()` for all three strategies
4. Implement `_infer_requirements()` (Option B)
5. Implement `select_template()` for all three strategies
6. Implement `build_prompt()` with all placeholder filling

### Phase 2: Integrate with ProgramsDatabase
1. Update `__main__.py` to call `load_specification()` at startup
2. Pass `PromptSpec` to `ProgramsDatabase.__init__()`
3. Replace `_generate_prompt()` to use `prompt_builder.build_prompt()`
4. Update `_generate_prompt_for_island()` to use `select_template()`

### Phase 3: Handle FunSearch special case
1. FunSearch uses versioned functions (`priority_v0`, `priority_v1`)
2. FunSearch uses `{fewshot_examples}` placeholder (different from `{better_code}`)
3. May need `_format_fewshot_examples()` helper for FunSearch specifically

### Phase 4: Handle ReEvo state
1. Add `reevo_state` dict to ProgramsDatabase
2. Implement phase progression logic
3. Implement reflection storage/retrieval
4. Handle reflector system message switching

### Phase 5: Testing and cleanup
1. Test each strategy independently
2. Remove legacy code from `specification_loader.py`
3. Update config examples in documentation

---

## Open Questions

1. **FunSearch fewshot format**: FunSearch wraps programs as versioned functions with docstrings. Should this be a separate `_format_fewshot_examples()` function, or should we use a completely different placeholder (`{fewshot_examples}` vs `{better_code}`)?

2. **ReEvo reflection flow**: Reflections are generated by a separate LLM call (with reflector system message). Should this go through the same sampler queue, or be handled differently?

3. **EoH style weights**: Should we support configurable probability weights for style selection (e.g., 60% mutation, 40% exploration)?

4. **Checkpointing**: How to save/restore `reevo_state` in checkpoints?
