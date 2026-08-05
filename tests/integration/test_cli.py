from typer.testing import CliRunner
from prism_fas.cli.main import app
def test_cli_help():
    result=CliRunner().invoke(app,["--help"]); assert result.exit_code == 0 and "PRISM-FAS-B" in result.output
