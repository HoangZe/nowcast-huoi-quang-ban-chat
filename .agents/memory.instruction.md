---
applyTo: '**'
---

# Coding Preferences
[To be discovered]

# Project Architecture
- Dataset inspection is performed through the Python scripts in `scripts/`.

# Solutions Repository
- For this project, the full cleaned dataset is hourly for 92,758 consecutive intervals but retains 25 non-hourly gaps; forecasting windows should account for the gap mask or exclude windows crossing those gaps.
