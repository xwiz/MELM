"""Prompt engineering pipeline — converts StoryPlan into system+user messages."""

from .assistant_skill_story_planning import StoryPlan

_WORD_COUNT_MAP: dict[str, str] = {
    "short": "under 100 words",
    "medium": "150-250 words",
    "long": "300-500 words",
}


class StoryPromptPipeline:
    def build(self, plan: StoryPlan) -> list[dict[str, str]]:
        return [self._system(plan), self._user(plan)]

    def build_string(self, plan: StoryPlan) -> str:
        """Return a single string combining system + user for template_hint."""
        messages = self.build(plan)
        return f"{messages[0]['content']}\n\n{messages[1]['content']}"

    def _system(self, plan: StoryPlan) -> dict[str, str]:
        parts = [
            f"You are a traditional {plan.cultural_texture} storyteller.",
            f"Write a vivid story in {_WORD_COUNT_MAP.get(plan.length_guide, '150-250 words')}.",
            "Keep sentences clear and warm.",
        ]
        if plan.literary_devices:
            parts.append(f"Weave in a {', '.join(plan.literary_devices)} naturally.")
        if plan.scene_suggestion:
            parts.append(f"The story should have about {plan.scene_suggestion} scenes or paragraphs.")
        return {"role": "system", "content": " ".join(parts)}

    def _user(self, plan: StoryPlan) -> dict[str, str]:
        parts = [f"Tell a story that teaches {plan.lesson}."]
        if plan.themes:
            parts.append(f"Themes: {', '.join(plan.themes)}.")
        if plan.setting_location:
            parts.append(f"Set in {plan.setting_location}.")
        if plan.protagonist_name:
            parts.append(f"The protagonist is {plan.protagonist_name}.")
        if plan.personal_facts:
            parts.append(f"{plan.protagonist_name} {' and '.join(plan.personal_facts)}.")
        if plan.recent_context:
            parts.append(" ".join(plan.recent_context))
        if plan.mood_tone and plan.mood_tone != "neutral":
            parts.append(f"The assistant feels {plan.mood_tone.replace('_', ' ')} — reflect this tone.")
        return {"role": "user", "content": " ".join(parts)}
