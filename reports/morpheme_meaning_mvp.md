# Morpheme Meaning MVP

Source: `benchmarks\morpheme_meaning_mvp.jsonl`
Components: `34`
Known lexemes: `3`
Word cases: `22`
Utterance cases: `6`

- Word inference accuracy: `100.00%`
- Utterance routing accuracy: `100.00%`
- Overall accuracy: `100.00%`

## Word Cases

| Item | Category | Passed | Predicted Components | Gloss |
|---|---|---:|---|---|
| rewelcome | novel_derived | True | prefix:re, root:wil, root:cuma | inferred meaning: again/repeat, desire, greeting, arrival |
| unwelcome | novel_derived | True | prefix:un, root:wil, root:cuma | inferred meaning: not/negation, greeting, arrival, guest |
| helpfulness | known_root_plus_suffix | True | root:help, suffix:ful, suffix:ness | inferred meaning: help/aid, possesses quality, benefit, abstract quality |
| kindness | root_plus_suffix | True | root:kind, suffix:ness | inferred meaning: gentleness, abstract quality, positive feeling, social good |
| hopeless | root_plus_suffix | True | root:hope, suffix:less | inferred meaning: absence, hope, positive feeling, not/negation |
| readable | root_plus_suffix | True | root:read, suffix:able | inferred meaning: reading, can/able, text, understanding |
| unreadable | prefix_root_suffix | True | prefix:un, root:read, suffix:able | inferred meaning: not/negation, reading, can/able, text |
| reread | prefix_root | True | prefix:re, root:read | inferred meaning: again/repeat, reading, text, understanding |
| misread | prefix_root | True | prefix:mis, root:read | inferred meaning: reading, error, incorrectness, text |
| prewrite | prefix_root | True | prefix:pre, root:write | inferred meaning: writing, before, text, creation |
| postlearn | prefix_root | True | prefix:post, root:learn | inferred meaning: after, learning, knowledge gain |
| teacher | agentive_suffix | True | root:teach, suffix:er | inferred meaning: teaching, agent, knowledge transfer, person |
| learner | agentive_suffix | True | root:learn, suffix:er | inferred meaning: learning, agent, knowledge gain, person |
| safely | adverbial_suffix | True | root:safe, suffix:ly | inferred meaning: safety, manner, protection |
| overcareful | prefix_root_suffix | True | prefix:over, root:care, suffix:ful | inferred meaning: possesses quality, concern, excess, attention |
| waterless | root_plus_suffix | True | root:water, suffix:less | inferred meaning: water, absence, fluid, not/negation |
| lightful | root_plus_suffix | True | root:light, suffix:ful | inferred meaning: light, possesses quality, visibility, positive feeling |
| giftable | root_plus_suffix | True | root:gift, suffix:able | inferred meaning: can/able, giving, social exchange, object |
| guarder | agentive_suffix | True | root:guard, suffix:er | inferred meaning: protection, agent, watching, person |
| clearwater | compound_root | True | root:clear, root:water | inferred meaning: water, fluid, clarity, visibility |
| mover | agentive_suffix | True | root:move, suffix:er | inferred meaning: motion, agent, change location, person |
| replay | prefix_root | True | prefix:re, root:play | inferred meaning: again/repeat, play, activity, joy |

## Utterance Cases

| Utterance | Category | Passed | Prediction | Features |
|---|---|---:|---|---|
| Please reread the note. | command_with_novel_word | True | command/action_frame/reread | `{"comprehension": 0.475, "reading": 0.902, "repeat": 0.95, "return": 0.427, "text": 0.57}` |
| What does unreadable mean? | question_about_novel_word | True | question/meaning_answer/unreadable | `{"ability": 0.81, "comprehension": 0.475, "negation": 0.95, "potential": 0.45, "reading": 0.902, "text": 0.57}` |
| A giftable means something that can be given. | knowledge_transfer | True | knowledge_transfer/store_candidate/giftable | `{"ability": 0.81, "giving": 0.81, "object": 0.45, "potential": 0.45, "social_exchange": 0.54}` |
| By wilgift I mean a gift offered with willing intent. | clarification | True | clarification/clarification_update/wilgift | `{"desire": 0.765, "giving": 0.81, "object": 0.45, "positive_valence": 0.595, "social_exchange": 0.54, "willingness": 0.68}` |
| Could you helpfulness the puzzle? | type_mismatch | True | clarification_needed/ask_clarification/helpfulness | `{"abstract_quality": 0.63, "aid": 0.95, "benefit": 0.712, "positive_valence": 0.332, "possesses_quality": 0.81}` |
| Please prewrite the plan. | command_with_novel_word | True | command/action_frame/prewrite | `{"before": 0.9, "creation": 0.38, "planning": 0.36, "text": 0.617, "writing": 0.902}` |
