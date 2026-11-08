"""The in-process GPU backends' pure parts: question mapping, token layout, readout, guards. No torch."""

import hashlib
import json
import math
import subprocess
import sys
import threading

import pytest

from s1.models.backends import gpu, kev, laya, semif
from s1.models.catalog import CATALOG

CHOICE = {
    "type": "choice",
    "instructions": "Pick",
    "criteria": {"opt_1": "red", "opt_2": "blue", "opt_3": ""},
}
NOUL = {"type": "noul", "instructions": "True?", "criteria": {"true": "it holds", "false": "it does not"}}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["none", "some", "lots"]}


def spec(name):
    return next(s for s in CATALOG if s.name == name)


# --- catalog ------------------------------------------------------------------------------------


def test_single_gpu_models_are_ready_and_parallel_ones_planned():
    ready = {s.name: s.backend for s in CATALOG if s.status == "ready"}
    assert ready == {
        "openjev": "vllm_shim",
        "laya.multilingual": "laya",
        "laya.english": "laya",
        "kev-4b.hybrid": "kev",
        "kev-4b.qwen3": "kev",
        "bongard-mini": "bongard",
        "semif.direct": "semif",
        "semif.reranker": "semif",
        "qwen35-9b.zeroshot": "semif",
        "jevk5-9b": "semif",
        "qwen3-8b.zeroshot": "semif",
        "olmoe-1b-7b.zeroshot": "semif",
        "lfm2-8b-a1b.zeroshot": "semif",
        "granite4-h-tiny.zeroshot": "semif",
        "elephant-1": "elephant",
    }


def test_l4_set_shares_semif_readout_and_differs_in_weights():
    names = ["qwen35-9b.zeroshot", "qwen3-8b.zeroshot", "olmoe-1b-7b.zeroshot", "lfm2-8b-a1b.zeroshot"]
    names += ["granite4-h-tiny.zeroshot", "jevk5-9b"]
    specs = [spec(n) for n in names]
    assert {s.options["mode"] for s in specs} == {"direct"}
    assert len({s.repo for s in specs}) == len(specs)
    assert all(s.revision is not None and len(s.revision) == 40 for s in specs)
    assert spec("jevk5-9b").options["semif_temperature"] == 1.316  # jevk5_config.json
    assert all(7e9 < (s.size_bytes or 0) < 20e9 for s in specs)  # one bf16 band that fits a 24 GB L4


def test_bongard_requires_the_rtx_pro_4500():
    assert spec("bongard-mini").options["require_gpu"] == "rtx pro 4500"
    assert all("require_gpu" not in s.options for s in CATALOG if s.name != "bongard-mini")


def test_laya_checkpoints():
    assert spec("laya.english").options["checkpoint"] == ""
    assert spec("laya.multilingual").options["checkpoint"] == "multilingual"


# --- gpu plumbing -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("NVIDIA RTX PRO 4500 Blackwell", True),
        ("NVIDIA RTX Pro 4500", True),
        ("NVIDIA A10G", False),
        ("NVIDIA L4", False),
    ],
)
def test_gpu_matches(name, ok):
    assert gpu.gpu_matches(name, "rtx pro 4500") is ok


def test_local_dir_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    (tmp_path / "there").mkdir()
    assert gpu.local_dir({}, "x_dir", "there", "fix") == tmp_path / "there"
    assert gpu.local_dir({"x_dir": str(tmp_path)}, "x_dir", "nope", "fix") == tmp_path
    with pytest.raises(FileNotFoundError, match=r"download_model\.sh org/m abc m"):
        gpu.local_dir({}, "x_dir", "m", gpu.download_hint("org/m", "abc", "m"))
    assert gpu.download_hint("org/m", None, "m").endswith('org/m "" m')


def test_check_sha256(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    gpu.check_sha256(f, hashlib.sha256(b"x = 1\n").hexdigest())
    with pytest.raises(RuntimeError, match="re-review"):
        gpu.check_sha256(f, "0" * 64)


def test_check_git_commit(tmp_path):
    def git(*a):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *a], capture_output=True, text=True, check=True
        ).stdout

    git("init", "-q")
    (tmp_path / "f").write_text("a")
    git("add", "f")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    head = git("rev-parse", "HEAD").strip()
    gpu.check_git_commit(tmp_path, head)
    with pytest.raises(RuntimeError, match="not the pinned"):
        gpu.check_git_commit(tmp_path, "0" * 40)
    (tmp_path / "f").write_text("b")
    with pytest.raises(RuntimeError, match="local changes"):
        gpu.check_git_commit(tmp_path, head)


def test_backend_modules_do_not_import_torch():
    assert "torch" not in sys.modules


# --- kev ----------------------------------------------------------------------------------------


class FakeServe:
    """Stands in for upstream's kev.serve module: a record is the state, an answer its probabilities."""

    class SystemOneRequest:
        def __init__(self, state, questions):
            self.state, self.questions = state, questions

    @staticmethod
    def prepare(req):
        return req

    @staticmethod
    def to_record(req):
        return {"state": req.state}, list(req.questions)

    @staticmethod
    def to_answers(probs, meta):
        return {key: {"type": "noul", "noul": p[1]} for key, p in zip(meta, probs, strict=True)}


class FakeServer:
    def __init__(self):
        self.encoded, self.lock, self.cleared, self.closed = [], threading.Lock(), False, False
        self.model, self.tok = self, "tok"
        self.prefix_cache = self
        self.queue = self

    def encode(self, tok, rec, **limits):
        self.encoded.append(limits)
        return rec

    def put(self, item):  # the model thread, answered at once
        _, done = item
        done.set_result(([[0.25, 0.75]], {}))

    def wait_idle(self):
        pass

    def clear(self):
        self.cleared = True

    def close(self):
        self.closed = True


def test_kev_encodes_at_our_limit_and_answers_through_upstream():
    server = FakeServer()
    k = kev.Kev(server, FakeServe, max_tokens=16384)
    assert k.decide("s", {"q": NOUL}) == {"q": {"type": "noul", "noul": 0.75}}
    assert server.encoded == [{"max_state": 16384, "max_branch": 16384, "strict": True}]
    k.after_warmup()
    assert server.cleared  # no state survives warm-up
    k.close()
    assert server.closed
    assert server.model is None  # the weights are released for the next run


def test_kev_variants_pin_their_upstream_code():
    for name in ("kev-4b.hybrid", "kev-4b.qwen3"):
        s = spec(name)
        assert s.code_repo == "jaredpalmer/kev"
        assert s.code_revision == "2074cc44256200bb37540103fa4572901faa10f3"


# --- semif --------------------------------------------------------------------------------------


def test_semif_rows_follow_build_typesafe():
    c = semif.to_row(CHOICE, 16)
    assert c.options == ["opt_1: red", "opt_2: blue", "opt_3: "]
    n = semif.to_row(NOUL, 16)
    assert (n.keys, n.options) == (["true", "false"], ["true: it holds", "false: it does not"])
    bare = semif.to_row({"type": "noul", "instructions": "x"}, 16)
    assert bare.options == ["true: The proposition is true.", "false: The proposition is false."]
    assert semif.to_row(SCORE, 16).options == ["0: none", "1: some", "2: lots"]


def test_semif_option_cap_is_upstreams_16():
    many = {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(17)}}
    with pytest.raises(ValueError, match="2-16"):
        semif.to_row(many, 16)
    assert len(semif.to_row(many, 26).options) == 17
    with pytest.raises(ValueError, match="2-26"):
        semif.to_row({**many, "criteria": {f"o{i}": "d" for i in range(27)}}, 99)


def test_semif_direct_prompt_is_one_line_json():
    row = semif.to_row(CHOICE, 16)
    system, user = semif.direct_messages("the state", row)
    assert system["content"].startswith("Apply the supplied criterion")
    payload = json.loads(user["content"])
    assert list(payload) == ["evidence", "criterion", "options"]
    assert payload["options"][1] == {"letter": "B", "description": "opt_2: blue"}
    assert "\n" not in user["content"]


def test_semif_reranker_prompt():
    row = semif.to_row(NOUL, 16)
    text = semif.rerank_text("doc", row, row.options[0])
    assert text.startswith("<|im_start|>system\nJudge whether the Document")
    assert "<Query>: Question: True?\nCandidate answer: true: it holds\n<Document>: doc" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_semif_answers():
    assert semif.answer(semif.to_row(NOUL, 16), [math.log(3), 0.0])["noul"] == pytest.approx(0.75)
    assert semif.answer(semif.to_row(CHOICE, 16), [0.0, 2.0, 1.0])["choice"] == "opt_2"


# --- laya ---------------------------------------------------------------------------------------


def test_laya_refuses_unreviewed_code(tmp_path):
    for name in laya.CODE_SHA256:
        (tmp_path / name).write_text("import os\n")
    with pytest.raises(RuntimeError, match="re-review"):
        laya.import_reviewed(tmp_path)
    assert "rl_agent_api" not in sys.modules
