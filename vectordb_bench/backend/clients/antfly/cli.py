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
    host: Annotated[
        str, click.option("--host", type=str, help="Antfly metadata API host", default="localhost", show_default=True)
    ]
    port: Annotated[
        int, click.option("--port", type=int, help="Antfly metadata API port", default=8080, show_default=True)
    ]
    store_host: Annotated[str, click.option("--store-host", type=str, help="Antfly store API host", default=None)]
    store_port: Annotated[
        int, click.option("--store-port", type=int, help="Antfly store API port for direct search", default=None)
    ]
    username: Annotated[str, click.option("--username", type=str, help="Antfly username", default=None)]
    password: Annotated[str, click.option("--password", type=str, help="Antfly password", default=None)]
    num_shards: Annotated[
        int, click.option("--num-shards", type=int, help="Number of shards", default=1, show_default=True)
    ]
    use_direct_store_search: Annotated[
        bool,
        click.option(
            "--use-direct-store-search/--no-direct-store-search",
            help="Query Antfly through the store /search API instead of the metadata table query API",
            default=False,
            show_default=True,
        ),
    ]
    pack_query_vectors: Annotated[
        bool,
        click.option(
            "--pack-query-vectors/--no-pack-query-vectors",
            help="Send query vectors in Antfly's packed base64 float32 wire format",
            default=False,
            show_default=True,
        ),
    ]
    search_effort: Annotated[
        float | None,
        click.option(
            "--search-effort",
            type=float,
            help="Search effort 0.0-1.0 (higher=better recall, slower)",
            default=None,
            show_default=True,
        ),
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
            store_host=parameters["store_host"],
            store_port=parameters["store_port"],
            username=SecretStr(parameters["username"]) if parameters["username"] else None,
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=parameters["num_shards"],
            use_direct_store_search=parameters["use_direct_store_search"],
            pack_query_vectors=parameters["pack_query_vectors"],
        ),
        db_case_config=AntflyIndexConfig(
            num_shards=parameters["num_shards"],
            search_effort=parameters["search_effort"],
        ),
        **parameters,
    )
