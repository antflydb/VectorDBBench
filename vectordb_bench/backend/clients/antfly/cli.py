from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class AntflyTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Antfly host", default="localhost", show_default=True)]
    port: Annotated[int, click.option("--port", type=int, help="Antfly port", default=8080, show_default=True)]
    username: Annotated[str, click.option("--username", type=str, help="Antfly username", default=None)]
    password: Annotated[str, click.option("--password", type=str, help="Antfly password", default=None)]
    num_shards: Annotated[
        int, click.option("--num-shards", type=int, help="Number of shards", default=1, show_default=True)
    ]


class AntflyAKNNTypedDict(CommonTypedDict, AntflyTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(AntflyAKNNTypedDict)
def AntflyAKNN(**parameters: Unpack[AntflyAKNNTypedDict]):
    from .config import AntflyConfig, AntflyIndexConfig

    run(
        db=DB.Antfly,
        db_config=AntflyConfig(
            db_label=parameters["db_label"],
            host=parameters["host"],
            port=parameters["port"],
            username=SecretStr(parameters["username"]) if parameters["username"] else None,
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=parameters["num_shards"],
        ),
        db_case_config=AntflyIndexConfig(
            num_shards=parameters["num_shards"],
        ),
        **parameters,
    )
