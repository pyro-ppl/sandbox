# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
from hashlib import sha1

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# Ensure directories exist.
for path in [DATA, RESULTS]:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def get_filename(script, args):
    unique = script, sorted(args.__dict__.items())
    fingerprint = sha1(str(unique).encode()).hexdigest()
    cachefile = os.path.join(RESULTS, fingerprint + ".pkl")
    return cachefile
