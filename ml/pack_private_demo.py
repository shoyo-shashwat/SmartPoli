"""Pack the private demo patient (CGMacros, not in the public repo) so it can travel to the deployed app.

Prints the value for the Render environment variable SMARTPOLI_TWIN_DEMO_B_GZ_B64 (gzip + base64 of demo_b.json), or use the
file itself as a Render Secret File named demo_b.json. Run:  python pack_private_demo.py ../backend/twin_demo_data/demo_b.json
"""
import base64, gzip, sys

raw = open(sys.argv[1], "rb").read()
out = base64.b64encode(gzip.compress(raw, 9)).decode()
open("demo_b.packed.txt", "w").write(out)
print(f"{len(raw)} bytes -> {len(out)} characters; written to demo_b.packed.txt (keep it private, do not commit)")
