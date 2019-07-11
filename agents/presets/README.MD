To use these presets a link must be made from the RL-Coach preset folder to this location.

The script link_presets.py can recursively link the presets.

Example use:
`python link_presets.py ~/github/Coach/rl_coach/presets/.`

After which the presets can be run using the standard rl-coach commands:
`coach -p ddpg_agent`