"""Pytest session configuration for the MELM test suite.

Bulk lexicon seeding (``melm/appliance/assistant_lexicon_bulk.py``,
``_resolve_max_entries``) defaults to a cap of 5000 entries. Seeding that many
entries makes the default ``python -m pytest`` run take multiple minutes and can
appear to hang. To keep the default suite fast and reproducible we cap bulk
seeding to a small value at collection time.

We use ``setdefault`` so an explicit environment value still wins. In
particular, a full-fidelity run that exports ``MELM_BULK_MAX_ENTRIES=0``
(0 == unbounded) is *not* overridden here.

Full-fidelity (unbounded seed) invocation for the pre-release gate:

    MELM_BULK_MAX_ENTRIES=0 pytest -m slow
"""

import os

# Cap bulk lexicon seeding small by default so the suite is fast. An explicit
# env value (e.g. MELM_BULK_MAX_ENTRIES=0 for a full-fidelity run) takes
# precedence because setdefault only writes when the key is absent.
os.environ.setdefault("MELM_BULK_MAX_ENTRIES", "200")
