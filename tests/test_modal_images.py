"""s1.launch.modal_images and modal_stage without the Modal client: build steps recorded by a fake `modal`."""

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.launch import modal_app, modal_stage
from s1.launch import modal_images as mi
from s1.models.catalog import ELEPHANT, KEV_HYBRID

REPO = Path(__file__).resolve().parents[1]
BUILD_STEPS = {"apt_install", "env", "run_commands", "pip_install", "uv_sync"}


class FakeImage:
    """Records every builder call; enforces Modal's rule that non-copy local files come last."""

    def __init__(self, steps=()):
        self.steps = list(steps)

    def _then(self, name, *args, **kwargs):
        if name in BUILD_STEPS or (name.startswith("add_local") and kwargs.get("copy")):
            late = [s for s in self.steps if s[0].startswith("add_local") and not s[2].get("copy")]
            if late:
                raise AssertionError(f"{name} after a non-copy {late[0][0]}: Modal refuses this")
        return type(self)([*self.steps, (name, args, kwargs)])

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self._then(name, *args, **kwargs)

    def names(self):
        return [s[0] for s in self.steps]

    def commands(self):
        return [c for name, args, _ in self.steps if name == "run_commands" for c in args]

    def first(self, name):
        return next(s for s in self.steps if s[0] == name)


def fake_modal():
    image = SimpleNamespace(
        from_registry=lambda ref, **kw: FakeImage([("from_registry", (ref,), kw)]),
        debian_slim=lambda **kw: FakeImage([("debian_slim", (), kw)]),
    )
    return SimpleNamespace(Image=image)


def build(runtime="default", openjev=True):
    img = mi.image(fake_modal(), runtime=runtime, openjev=openjev)
    assert isinstance(img, FakeImage)
    return img


def test_import_does_not_import_modal():
    code = (
        "import sys, s1.launch.modal_images, s1.launch.modal_stage; "
        "assert 'modal' not in sys.modules, 'modal imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_runtimes_and_check():
    assert mi.RUNTIMES == ("default", "elephant", "granite")
    for r in mi.RUNTIMES:
        assert mi.check_runtime(r) == r
    with pytest.raises(ValueError, match="unknown"):
        mi.check_runtime("vllm")


# ---- default ----


def test_default_gpu_image_steps():
    img = build()
    assert img.steps[0] == ("from_registry", (mi.CUDA_DEVEL_IMAGE,), {"add_python": mi.PYTHON})
    apt = img.first("apt_install")[1]
    assert {"gcc", "libc6-dev", "git", "curl", "xz-utils"} <= set(apt)
    sync = img.first("uv_sync")
    assert sync[1] == (str(mi.REPO_ROOT),)
    assert sync[2] == {"extras": ["openjev"], "frozen": True, "uv_version": mi.UV_VERSION}
    names = img.names()
    # nsight before the sync (a lock change does not refetch it), local files last
    assert names.index("run_commands") < names.index("uv_sync")
    assert names[-2:] == ["add_local_dir", "add_local_dir"]
    assert {"CUDA_HOME": "/usr/local/cuda"} in [s[1][0] for s in img.steps if s[0] == "env"]
    assert dict(mi.IMAGE_ENV) in [s[1][0] for s in img.steps if s[0] == "env"]
    assert not any("mamba" in c or "causal" in c for c in img.commands())  # granite's kernels only


def test_default_cpu_image_has_git_and_no_cuda():
    img = build(openjev=False)
    assert img.steps[0][0] == "debian_slim"
    assert "git" in img.first("apt_install")[1]
    assert img.first("uv_sync")[2]["extras"] == []
    assert img.commands() == []  # no nsight, no runtime builds


def test_local_dirs():
    img = build()
    locals_ = [(s[1][0], s[1][1]) for s in img.steps if s[0] == "add_local_dir"]
    assert locals_ == [
        (mi.REPO_ROOT / "src", f"{mi.WORKDIR}/src"),
        (mi.REPO_ROOT / "configs", f"{mi.WORKDIR}/configs"),
    ]


def test_nsight_pins_match_modal_probe():
    """The same archives and sha256s scripts/modal_probe.py ran (parsed: importing it imports modal)."""
    tree = ast.parse((REPO / "scripts" / "modal_probe.py").read_text())
    consts = {
        t.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id in ("REDIST", "TOOLS")
    }
    assert consts["REDIST"] == mi.REDIST
    assert {k: v[:2] for k, v in mi.NSIGHT.items()} == consts["TOOLS"]


def test_nsight_commands_verify_and_link():
    cmds = build().commands()
    text = "\n".join(cmds)
    for tool, (rel, sha, exe) in mi.NSIGHT.items():
        assert f"{mi.REDIST}{rel}" in text
        assert f"echo '{sha}  /tmp/{tool}.tar.xz' | sha256sum -c -" in text
        assert f"test -x {mi.NSIGHT_DIR}/{tool}/{exe}" in text
        for d in mi.NSIGHT_LINKS:
            assert f"ln -sf {mi.NSIGHT_BIN}/{tool} {d}/{tool}" in cmds
    assert "/usr/local/cuda/bin" in mi.NSIGHT_LINKS  # `s1 ncu`'s default --ncu path
    assert cmds[-1].startswith("ncu --version && nsys --version")


def test_s1_ncu_default_path_is_linked():
    src = (REPO / "src" / "s1" / "cli.py").read_text()
    default = re.search(r'ncu_bin: str = typer.Option\("([^"]+)"', src)
    assert default is not None
    assert Path(default.group(1)).parent.as_posix() in mi.NSIGHT_LINKS


def test_fetch_quotes_and_checks():
    pin = mi.Pinned("https://example.org/a%2Bb.whl", "0" * 64)
    assert pin.filename == "a+b.whl"
    cmd = mi.fetch(pin, "/tmp/a+b.whl")
    assert cmd.startswith("curl -fsSL --retry 3 -o /tmp/a+b.whl https://example.org/a%2Bb.whl")
    assert cmd.endswith(f"echo '{'0' * 64}  /tmp/a+b.whl' | sha256sum -c -")


# ---- elephant ----


def test_elephant_image():
    img = build("elephant")
    copies = [s for s in img.steps if s[0] == "add_local_file" and s[2].get("copy")]
    assert [(s[1][0], s[1][1]) for s in copies] == [
        (mi.ELEPHANT_SRC / "pyproject.toml", f"{mi.ELEPHANT_DIR}/pyproject.toml"),
        (mi.ELEPHANT_SRC / "uv.lock", f"{mi.ELEPHANT_DIR}/uv.lock"),
    ]
    worker = img.steps[-1]
    assert worker[0] == "add_local_file" and not worker[2].get("copy")
    assert worker[1] == (mi.ELEPHANT_SRC / "worker.py", f"{mi.ELEPHANT_DIR}/worker.py")
    text = "\n".join(img.commands())
    assert mi.SERVE_PATCH.sha256 in text and mi.SERVE_PATCH.url in text
    assert f'test "$(git -C {mi.ELEPHANT_DIR}/OneJev rev-parse HEAD)" = {mi.ONEJEV_COMMIT}' in text
    assert f"UV_PROJECT_ENVIRONMENT={mi.ELEPHANT_DIR}/.venv" in text  # never the harness's /.uv/.venv
    assert "uv sync --frozen --python 3.11" in text
    assert mi.ELEPHANT_CAUSAL_CONV1D.sha256 in text
    assert "mamba" not in text


def test_elephant_order():
    cmds = build("elephant").commands()
    idx = {
        k: next(i for i, c in enumerate(cmds) if k in c)
        for k in ("serve.patch", "git clone", "git -C", "apply", "uv sync")
    }
    assert idx["serve.patch"] < idx["git clone"] < idx["apply"] < idx["uv sync"]


def test_elephant_pins_match_setup_sh_and_catalog():
    setup = (REPO / "runtimes" / "elephant" / "setup.sh").read_text()
    assert f"ONEJEV={mi.ONEJEV_COMMIT}" in setup
    assert f"PATCH_SHA256={mi.SERVE_PATCH.sha256}" in setup
    assert "causal-conv1d==1.7.0" in setup
    assert "1.7.0" in mi.ELEPHANT_CAUSAL_CONV1D.filename
    assert ELEPHANT.code_revision == mi.ONEJEV_COMMIT
    assert ELEPHANT.revision == mi.ELEPHANT_REVISION
    assert ELEPHANT.code_repo is not None and ELEPHANT.code_repo in mi.ONEJEV_REPO
    project = tomllib.loads((mi.ELEPHANT_SRC / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == f"=={mi.ELEPHANT_PYTHON_VERSION}.*"
    assert "torch==2.8.0" in project["dependencies"]
    cp = "cp" + mi.ELEPHANT_PYTHON_VERSION.replace(".", "")
    assert f"torch2.8cxx11abiTRUE-{cp}-{cp}-linux_x86_64" in mi.ELEPHANT_CAUSAL_CONV1D.filename


def test_elephant_dir_is_backend_default():
    """The image puts the runtime where backends/elephant.py looks inside the container."""
    from s1.models.backends import elephant  # noqa: PLC0415

    rel = elephant.RUNTIME.relative_to(mi.REPO_ROOT).as_posix()
    assert f"{mi.WORKDIR}/{rel}" == mi.ELEPHANT_DIR
    assert f"{mi.ELEPHANT_DIR}/.venv/bin/python" == mi.ELEPHANT_PYTHON


# ---- granite ----


def test_granite_image():
    img = build("granite")
    assert ("apt_install", ("g++",), {}) in img.steps
    names = img.names()
    assert names.index("uv_sync") < names.index("apt_install", names.index("uv_sync"))  # after the sync
    cmds = img.commands()
    text = "\n".join(cmds)
    for pin in [*mi.GRANITE_KERNELS.values(), *mi.GRANITE_BUILD_REQS.values()]:
        assert pin.url in text
        assert f"echo '{pin.sha256}  /tmp/granite-build/{pin.filename}' | sha256sum -c -" in text
    install = next(c for c in cmds if "--no-build-isolation" in c)
    assert install.startswith(
        "CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 CC=gcc CXX=g++ uv pip install"
    )
    assert f"--python {mi.UV_VENV}/bin/python --no-deps" in install  # the harness's torch, no new deps
    assert install.index("causal_conv1d-1.7.0") < install.index("mamba_ssm-2.2.6.post3")
    assert any(
        "import torch, causal_conv1d, causal_conv1d_cuda, selective_scan_cuda, mamba_ssm" in c for c in cmds
    )
    assert names[-2:] == ["add_local_dir", "add_local_dir"]


def test_uv_venv_matches_modal_app():
    assert f"{mi.UV_VENV}/bin/python" == modal_app.VENV_PYTHON


@pytest.mark.parametrize("runtime", ["elephant", "granite"])
def test_gpu_only_runtimes_refuse_cpu(runtime):
    with pytest.raises(ValueError, match="GPU-only"):
        build(runtime, openjev=False)


def test_describe():
    assert mi.describe("default", openjev=False).endswith(f"at {mi.WORKDIR}")
    assert mi.describe("default", openjev=True).endswith("+ ncu/nsys")
    assert "torch 2.8" in mi.describe("elephant", openjev=True)
    assert "mamba-ssm 2.2.6.post3" in mi.describe("granite", openjev=True)
    with pytest.raises(ValueError, match="unknown"):
        mi.describe("nope", openjev=True)


# ---- build CLI ----


def test_build_cli_parses():
    calls = []

    def run(runtime, *, openjev):
        calls.append((runtime, openjev))

    mi.main(["build", "granite"], run=run)
    mi.main(["build", "default", "--cpu"], run=run)
    assert calls == [("granite", True), ("default", False)]
    with pytest.raises(SystemExit):
        mi.main(["build", "elephant", "--cpu"], run=run)
    with pytest.raises(SystemExit):
        mi.main(["build", "vllm"], run=run)
    assert calls == [("granite", True), ("default", False)]


def test_build_cli_help():
    out = subprocess.run(
        [sys.executable, "-m", "s1.launch.modal_images", "build", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout
    assert "{default,elephant,granite}" in out and "--cpu" in out


def test_build_builds_one_image(monkeypatch):
    built = []

    class App:
        def __init__(self, name):
            self.name = name

        def run(self, detach=False):
            return _Ctx()

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Img(FakeImage):
        def build(self, app):
            built.append((app.name, self.names()))

    def from_registry(ref, **kw):
        return Img([("from_registry", (ref,), kw)])

    modal = SimpleNamespace(
        App=App,
        enable_output=_Ctx,
        Image=SimpleNamespace(from_registry=from_registry, debian_slim=None),
    )
    monkeypatch.setattr(modal_app, "_modal", lambda: modal)
    monkeypatch.setattr(modal_app, "_check_python", lambda: None)
    mi.build("granite", openjev=True)
    assert len(built) == 1 and built[0][0] == modal_app.APP_NAME
    assert "run_commands" in built[0][1]


# ---- modal_stage ----

KEV_SHA = "2074cc44256200bb37540103fa4572901faa10f3"


def test_stage_import_has_no_modal_and_help():
    out = subprocess.run(
        [sys.executable, "-m", "s1.launch.modal_stage", "fetch-code", "--help"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    ).stdout
    assert "revision" in out and "dirname" in out


def test_stage_volume_matches_modal_app():
    assert modal_app.VOLUMES[mi.MODELS_MOUNT] == modal_stage.VOLUME


@pytest.mark.parametrize(
    ("args", "want"),
    [
        (("jaredpalmer/kev", KEV_SHA, "kev-code"), "kev-code"),
        (("jaredpalmer/kev", KEV_SHA, None), "kev"),
    ],
)
def test_stage_check_args(args, want):
    assert modal_stage.check_args(*args) == want


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("kev", KEV_SHA, None), "owner"),
        (("a/../b", KEV_SHA, None), "owner"),
        (("https://github.com/a/b", KEV_SHA, None), "owner"),
        (("a/b", "main", None), "40-character"),
        (("a/b", KEV_SHA[:12], None), "40-character"),
        (("a/b", KEV_SHA.upper(), None), "40-character"),
        (("a/b", KEV_SHA, "../etc"), "dirname"),
        (("a/b", KEV_SHA, "x/y"), "dirname"),
        (("a/b", KEV_SHA, ".."), "dirname"),
    ],
)
def test_stage_check_args_rejects(args, match):
    with pytest.raises(ValueError, match=match):
        modal_stage.check_args(*args)


def test_stage_cli():
    calls = []
    modal_stage.main(["fetch-code", "jaredpalmer/kev", KEV_SHA, "kev-code"], run=lambda *a: calls.append(a))
    assert calls == [("jaredpalmer/kev", KEV_SHA, "kev-code")]
    with pytest.raises(SystemExit):
        modal_stage.main(["fetch-code", "jaredpalmer/kev", "main"], run=lambda *a: calls.append(a))
    assert len(calls) == 1


def test_stage_kev_pin_is_catalog():
    assert KEV_HYBRID.code_revision == KEV_SHA and KEV_HYBRID.code_repo == "jaredpalmer/kev"


def fake_git(head, dirty="", log=None):
    log = [] if log is None else log

    def git(*args):
        log.append(args)
        if args[0] == "clone":
            Path(args[-1], ".git").mkdir(parents=True)
        if args[-2:] == ("rev-parse", "HEAD"):
            return head
        if args[-1] == "--porcelain":
            return dirty
        return ""

    return git, log


def test_checkout_clones_then_reuses(tmp_path):
    dest = tmp_path / "kev-code"
    git, log = fake_git(KEV_SHA)
    report = modal_stage.checkout("jaredpalmer/kev", KEV_SHA, dest, git)
    assert log[0] == ("clone", "--quiet", "https://github.com/jaredpalmer/kev.git", str(dest))
    assert ("-C", str(dest), "fetch", "--quiet", "origin", KEV_SHA) in log
    assert ("-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "--quiet", KEV_SHA) in log
    assert modal_stage.verify(report) == report
    git2, log2 = fake_git(KEV_SHA)
    modal_stage.checkout("jaredpalmer/kev", KEV_SHA, dest, git2)
    assert log2[0][0] == "-C"  # already a checkout: no second clone


def test_checkout_refuses_non_git_dir(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "file").write_text("weights?")
    with pytest.raises(FileExistsError):
        modal_stage.checkout("a/b", KEV_SHA, tmp_path / "x", fake_git(KEV_SHA)[0])


def test_verify_refuses_wrong_head_or_dirty(tmp_path):
    wrong = modal_stage.checkout("a/b", KEV_SHA, tmp_path / "a", fake_git("f" * 40)[0])
    with pytest.raises(RuntimeError, match="not the pinned"):
        modal_stage.verify(wrong)
    dirty = modal_stage.checkout("a/b", KEV_SHA, tmp_path / "b", fake_git(KEV_SHA, " M x.py")[0])
    with pytest.raises(RuntimeError, match="local changes"):
        modal_stage.verify(dirty)


def test_fetch_report_roundtrip():
    report = {"repo": "a/b", "revision": KEV_SHA, "head": KEV_SHA, "dest": "/models/b", "dirty": False}
    assert modal_stage.fetch_report(json.dumps(report)) == report
    with pytest.raises(TypeError):
        modal_stage.fetch_report("[]")


def test_checkout_real_git(tmp_path):
    """The real git calls against a local repo (clone over file:// instead of GitHub)."""
    origin = tmp_path / "origin"
    origin.mkdir()

    def g(*a):
        return subprocess.run(
            ["git", "-C", str(origin), *a], capture_output=True, text=True, check=True
        ).stdout

    g("init", "-q")
    (origin / "f.txt").write_text("1")
    g("add", ".")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    sha = g("rev-parse", "HEAD").strip()
    (origin / "f.txt").write_text("2")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    dest = tmp_path / "dest"
    subprocess.run(["git", "clone", "-q", str(origin), str(dest)], check=True)
    report = modal_stage.checkout("a/b", sha, dest)  # already a checkout: fetch + checkout only
    assert modal_stage.verify(report)["head"] == sha
    assert (dest / "f.txt").read_text() == "1"
