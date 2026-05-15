# Sound-Symbolism Is Deferred

Sound-symbolism and phonosemantic cues may still be useful for MELM, but they
are not mature enough to carry the active MVP gate.

Current decision:

- do not use encoded sound-symbolism in the active morpheme meaning MVP;
- keep the active corpus focused on higher-confidence roots, productive
  morphemes, compounds, usage definitions, and clarification updates;
- revisit sound-symbolism only as a separate ablation with explicit false
  positive controls.

Future sound-symbolism validation should ask whether weak sound priors improve
novel-word guessing without increasing confident wrong answers. Until that
happens, sound cues should not influence MELM's answer confidence.
