"""Load and format the YAML prompt bundles shared by every evaluation."""

from pathlib import Path
from typing import Any, Dict

import yaml

_prompts_path = Path(__file__).parent / "prompts.yaml"
_TEMPLATES: Dict[str, Dict[str, str]] = yaml.safe_load(_prompts_path.read_text())

class PromptFactory:
    """Light wrapper to fetch and format raw YAML-based templates."""

    @staticmethod
    def get(prompt_name: str, **kwargs: Any) -> Dict[str, str]:
        """Fetch and format the system/user templates for ``prompt_name``."""
        prompt_type = kwargs.get("prompt_type", None)

        if prompt_name not in _TEMPLATES:
            raise KeyError(f"Unknown prompt: {prompt_name!r}")

        if not prompt_type:
            tpl = _TEMPLATES[prompt_name]
        else:
            tpl = _TEMPLATES[prompt_name][prompt_type]

        if kwargs:
            sys_msg  = tpl["sys"].format(**kwargs)
            user_msg = tpl["user"].format(**kwargs)

            return sys_msg, user_msg

        else:
            return tpl["sys"], tpl["user"]
