"""Zero-shot prompt templates for structured extraction.

Note: Templates must NOT include BOS tokens (e.g. <|begin_of_text|>)
because llama.cpp adds them automatically.
"""

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
<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract the following fields from the text below.

Schema:
{schema}

Text:
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

EVIDENCED_EXTRACTION_SYSTEM = (
    "You are a precise data extraction assistant. "
    "Extract the requested information from the given text. "
    "For EACH field, output a JSON object with three keys:\n"
    '- "value": the extracted value\n'
    '- "evidence": a VERBATIM quote from the source text that supports the value '
    "(copy the exact characters, do not paraphrase)\n"
    '- "is_explicit": true if the value itself appears word-for-word in the text, '
    "false if it was inferred or derived\n\n"
    "If a field cannot be determined from the text, set value to null, "
    'evidence to "", and is_explicit to false.'
)

EVIDENCED_EXTRACTION_PROMPT_LLAMA = """\
<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract the following fields from the text below. For each field, provide the value, \
a verbatim evidence quote from the text, and whether the extraction is explicit.

Schema:
{schema}

Text:
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

EVIDENCED_EXTRACTION_PROMPT_CHATML = """\
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Extract the following fields from the text below. For each field, provide the value, \
a verbatim evidence quote from the text, and whether the extraction is explicit.

Schema:
{schema}

Text:
{text}<|im_end|>
<|im_start|>assistant
"""

EVIDENCED_EXTRACTION_PROMPT = """\
<|system|>
{system}
<|end|>
<|user|>
Extract the following fields from the text below. For each field, provide the value, \
a verbatim evidence quote from the text, and whether the extraction is explicit.

Schema:
{schema}

Text:
{text}
<|end|>
<|assistant|>
"""

SCHEMA_INFERENCE_PROMPT = """\
<|start_header_id|>system<|end_header_id|>

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


def format_evidenced_extraction_prompt(
    text: str,
    schema_description: str,
    prompt_format: str = "llama",
) -> str:
    """Format an evidenced extraction prompt with the given text and schema.

    Like format_extraction_prompt but instructs the model to include
    verbatim evidence quotes and explicit/implicit classification.
    """
    templates = {
        "llama": EVIDENCED_EXTRACTION_PROMPT_LLAMA,
        "chatml": EVIDENCED_EXTRACTION_PROMPT_CHATML,
        "generic": EVIDENCED_EXTRACTION_PROMPT,
    }
    template = templates.get(prompt_format, EVIDENCED_EXTRACTION_PROMPT_LLAMA)
    return template.format(
        system=EVIDENCED_EXTRACTION_SYSTEM,
        schema=schema_description,
        text=text,
    )
