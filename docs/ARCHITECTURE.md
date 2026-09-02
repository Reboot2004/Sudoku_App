# Architecture
Mobile solver/candidates run locally. Backend supplies daily DC data and provides optional personal-puzzle API persistence.

DC ingestion uses direct image assets (`tabpX_Y`) and the proven reference/dimension/template detector; article links are not required.
The daily GitHub workflow runs Monday-Saturday at 08:00 IST and skips Sundays.
