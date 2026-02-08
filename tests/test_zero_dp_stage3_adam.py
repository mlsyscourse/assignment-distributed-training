import pathlib
import re
import shutil
import subprocess
import sys

def test_zero_dp_stage3_train(capfd):
    if shutil.which("mpirun") is None:
        # Keep this test portable in environments without MPI launcher.
        return

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        "mpirun",
        "-n",
        "4",
        "python3",
        "zero_dp_train.py",
        "--dp_size",
        "4",
        "--num_epoch",
        "1",
        "--num_train_samples",
        "8000",
        "--num_test_samples",
        "1200",
        "--batch_size",
        "100",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    try:
        assert proc.stdout is not None
        # Disable pytest FD capture so students can see live training logs.
        with capfd.disabled():
            for line in proc.stdout:
                output_lines.append(line)
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
            proc.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("zero_dp_train.py timed out after 1200 seconds")

    output = "".join(output_lines)

    if proc.returncode != 0:
        raise AssertionError(
            "zero_dp_train.py failed. Output:\n" + output
        )

    matches = re.findall(r"Test Acc:\s*([0-9]*\.?[0-9]+)", output)
    if not matches:
        raise AssertionError(
            "Could not find 'Test Acc' in zero_dp_train.py output. Output:\n" + output
        )

    final_acc = float(matches[-1])
    assert final_acc > 0.85, f"Expected final Test Acc > 0.85, got {final_acc}"
