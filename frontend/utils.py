import re


def color_timestamps(text: str) -> str:
    return re.sub(
        r'\[(\d{1,2}(?::\d{2}){1,2})\s*-\s*(\d{1,2}(?::\d{2}){1,2})\]',
        lambda m: f'<span style="color:#1E90FF;font-weight:bold;">[{m.group(1)} - {m.group(2)}]</span>',
        text
    )