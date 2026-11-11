"""Reading API documents for the API-surface tests of every area.

A fenced block without a language tag (a line of exactly ```) holds public
names and signatures only; blocks with a tag (```python …) are prose examples
and are skipped. Fences are read line by line, so a tagged block's closing
fence is never taken for the opening of a signature block.
"""


def signature_blocks(text):
    """The lines of every untagged fenced block in `text`, block by block."""
    blocks, current, inside = [], None, False
    for line in text.splitlines():
        if line.startswith("```"):
            if not inside:
                inside = True
                current = [] if line.strip() == "```" else None
            else:
                inside = False
                if current is not None:
                    blocks.append(current)
                current = None
        elif inside and current is not None:
            current.append(line)
    return blocks
