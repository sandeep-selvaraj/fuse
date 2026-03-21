"""Zero-shot prompt templates for structured extraction."""

EXTRACTION_SYSTEM = (
    "You are a precise data extraction assistant. "
    "Extract the requested information from the given text. "
    "Output ONLY valid JSON matching the specified schema. "
    "If a field cannot be determined from the text, use null."
)

EXTRACTION_PROMPT = """\
<|system|>
{system}
<|end|>
<|user|>
Extract the following fields from the text below.

Schema:
{schema}

Text:
{text}
<|end|>
<|assistant|>
"""

EXTRACTION_PROMPT_CHATML = """\
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Extract the following fields from the text below.

Schema:
{schema}

Text:
{text}<|im_end|>
<|im_start|>assistant
"""

EXTRACTION_PROMPT_LLAMA = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract the following fields from the text below.

Schema:
{schema}

Text:
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

SCHEMA_INFERENCE_PROMPT = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a schema designer. Given a description of what information to extract, \
output a list of fields with their names and types. \
Use types: string, integer, number, boolean, array. \
For array fields, specify items_type.<|eot_id|><|start_header_id|>user<|end_header_id|>

Design a JSON schema for extracting the following:
{description}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def format_extraction_prompt(
    text: str,
    schema_description: str,
    prompt_format: str = "llama",
) -> str:
    """Format an extraction prompt with the given text and schema.

    Args:
        text: The input text to extract from.
        schema_description: Human-readable description of the schema fields.
        prompt_format: One of "llama", "chatml", or "generic".
    """
    templates = {
        "llama": EXTRACTION_PROMPT_LLAMA,
        "chatml": EXTRACTION_PROMPT_CHATML,
        "generic": EXTRACTION_PROMPT,
    }
    template = templates.get(prompt_format, EXTRACTION_PROMPT_LLAMA)
    return template.format(
        system=EXTRACTION_SYSTEM,
        schema=schema_description,
        text=text,
    )
