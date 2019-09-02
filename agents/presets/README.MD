To use these presets a link must be made from the RL-Coach preset folder to this location.

The script link_presets.py can recursively link the agent presets in this folder to the preset directory used by rl_coach.

Example use:
`python link_presets.py ~/github/Coach/rl_coach/presets/.`

After which the presets can be run using the rl-coach command line commands:
`coach -p ddpg_agent`
