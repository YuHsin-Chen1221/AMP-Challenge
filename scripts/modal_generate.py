"""Run the full 50k `generate` pipeline on a Modal H100 (CPU is too slow for scoring 50k x 11
endpoints). Weights come from the ampgen-generator volume; outputs (library.fasta / top.fasta)
are written back to it under submission_50k/. This is dev tooling for producing OUR submission --
the actual submission entry point is `uv run generate` (pyproject), which the validator runs.

    modal run scripts/modal_generate.py                # full 50k, seed 42
    modal run scripts/modal_generate.py --n 2000       # quick scale test
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("ampgen-generate")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.40", "peft>=0.11", "pandas", "numpy", "scipy",
                 "Levenshtein")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(str(REPO / "src"), "/repo/src")
    .add_local_file(str(REPO / "data/antibacterial.fasta"), "/repo/data/antibacterial.fasta")
    .add_local_file(str(REPO / "data/length_dist.json"), "/repo/data/length_dist.json")
)
vol = modal.Volume.from_name("ampgen-generator", create_if_missing=True)
cache = modal.Volume.from_name("ampgen-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="H100", volumes={"/out": vol, "/cache": cache}, timeout=60 * 60 * 4)
def generate(n: int = 50_000, top_k: int = 100, seed: int = 42):
    import os, subprocess, sys, shutil

    # wire the volume weights into the layout generate.py expects (/repo/weights/*)
    os.makedirs("/repo/weights", exist_ok=True)
    for link, target in [("/repo/weights/generator", "/out/generator_stage1_expanded"),
                         ("/repo/weights/predictor_encoder", "/out/generator_stage1_all"),
                         ("/repo/weights/predictor_mic.pt", "/out/predictor_v2_deploy/predictor_mic.pt"),
                         ("/repo/weights/predictor_hemo.pt", "/out/predictor_v2_deploy/predictor_hemo.pt")]:
        if not os.path.exists(link):
            os.symlink(target, link)

    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    cmd = [sys.executable, "-m", "ampgen.generate",
           "--n-sequences", str(n), "--top-k", str(top_k), "--seed", str(seed)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/repo", env=env)

    out = "/out/submission_50k"
    os.makedirs(out, exist_ok=True)
    for f in ("library.fasta", "top.fasta"):
        shutil.copy(f"/repo/generate/{f}", f"{out}/{f}")
    vol.commit()
    print("WROTE", out, flush=True)


@app.local_entrypoint()
def main(n: int = 50_000, top_k: int = 100, seed: int = 42):
    generate.remote(n=n, top_k=top_k, seed=seed)
