# SPDX-License-Identifier: Apache-2.0
import pytest

from aisl_train.cli import build_parser, main


def test_top_level_help() -> None:
    text = build_parser().format_help()
    for command in ("doctor", "inspect", "smoke", "baseline", "train", "evaluate", "compare"):
        assert command in text


@pytest.mark.parametrize(
    "command",
    ["doctor", "inspect", "smoke", "baseline", "train", "evaluate", "compare"],
)
def test_each_command_help(command: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
