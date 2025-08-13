"""
Main Antfly client implementation.
"""

import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx

from .exceptions import (
    AntflyConnectionError,
    AntflyHTTPError,
    AntflyIndexError,
    AntflyTableError,
    AntflyValidationError,
)
from .models import (
    AddIndexRequest,
    BatchRequest,
    CreateTableRequest,
    DeleteResult,
    IndexConfig,
    QueryRequest,
    QueryResult,
    Table,
    UpsertResult,
    Vector,
)


class AntflyClient:
    """
    Python client for Antfly vector database.

    This client provides a comprehensive interface to interact with Antfly,
    supporting all database operations including table management, indexing,
    querying, and batch operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        scheme: str = "http",
        default_table: Optional[str] = None,
        default_index: Optional[str] = None,
        timeout: float = 30.0,
        auth: Optional[tuple] = None,
        **httpx_kwargs
    ) -> None:
        """
        Initialize the Antfly client.

        Args:
            host: Hostname of the Antfly service
            port: Port of the Antfly service
            scheme: URL scheme (http or https)
            default_table: Default table name to use for operations
            default_index: Default index name to use for operations
            timeout: Request timeout in seconds
            auth: Optional basic auth tuple (username, password)
            **httpx_kwargs: Additional arguments passed to httpx.Client
        """
        self.base_url = f"{scheme}://{host}:{port}"
        self.default_table = default_table
        self.default_index = default_index
        self.full_text_index = "full_text_index"

        # Configure httpx client
        client_kwargs = {
            "timeout": httpx.Timeout(timeout),
            "base_url": self.base_url,
            **httpx_kwargs
        }

        if auth:
            client_kwargs["auth"] = httpx.BasicAuth(auth[0], auth[1])

        self.client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "AntflyClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        raise_for_status: bool = True
    ) -> httpx.Response:
        """Make an HTTP request with proper error handling."""
        try:
            response = self.client.request(
                method=method,
                url=endpoint,
                json=json_data,
                params=params
            )

            if raise_for_status:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise AntflyHTTPError(
                        f"HTTP {e.response.status_code} error: {e.response.text}",
                        status_code=e.response.status_code,
                        response_text=e.response.text
                    )

            return response

        except httpx.RequestError as e:
            raise AntflyConnectionError(f"Connection error: {str(e)}")

    # Table Management Methods

    def create_table(
        self,
        table_name: str,
        schema: Optional[Dict[str, Any]] = None,
        num_shards: int = 1
    ) -> Table:
        """
        Create a new table.

        Args:
            table_name: Name of the table to create
            schema: Optional table schema definition
            num_shards: Number of shards for the table

        Returns:
            Table information
        """
        request_data = CreateTableRequest(
            shards=num_shards,
            table_schema=schema
        )

        response = self._make_request(
            "POST",
            f"/table/{table_name}",
            json_data=request_data.dict(by_alias=True)
        )

        return Table(**response.json())

    def get_table(self, table_name: Optional[str] = None) -> Table:
        """
        Get details of a specific table.

        Args:
            table_name: Name of the table (uses default if not provided)

        Returns:
            Table information
        """
        table_name = table_name or self.default_table
        if not table_name:
            raise AntflyValidationError("table_name is required")

        response = self._make_request("GET", f"/table/{table_name}")
        return Table(**response.json())

    def list_tables(self) -> List[Table]:
        """
        List all tables.

        Returns:
            List of tables
        """
        response = self._make_request("GET", "/table")
        return [Table(**table_data) for table_data in response.json()]

    def delete_table(self, table_name: str) -> None:
        """
        Delete a table.

        Args:
            table_name: Name of the table to delete
        """
        self._make_request("DELETE", f"/table/{table_name}")

    # Index Management Methods

    def create_index(
        self,
        table_name: str,
        index_name: str,
        dimension: int = 768,
        template: str = "string",
        mem_only: bool = False,
        provider: str = "ollama",
        model: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        """
        Create a vector index on a table.

        Args:
            table_name: Name of the table
            index_name: Name of the index to create
            dimension: Dimension of the vectors
            template: Template for the index
            mem_only: Whether to keep the index in memory only
            provider: Embedding provider (default: ollama)
            model: Model name for embeddings
            url: URL for the embedding service
        """
        # Use environment variables for defaults if not provided
        if model is None:
            model = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
        if url is None:
            ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
            ollama_port = int(os.environ.get("OLLAMA_PORT", 11434))
            url = f"http://{ollama_host}:{ollama_port}"

        request_data = AddIndexRequest(
            template=template,
            memOnly=mem_only,
            dimension=dimension,
            pluginConfig={
                "provider": provider,
                "model": model,
                "url": url
            }
        )

        try:
            self._make_request(
                "POST",
                f"/table/{table_name}/index/{index_name}",
                json_data=request_data.dict(by_alias=True)
            )
        except AntflyHTTPError as e:
            raise AntflyIndexError(f"Failed to create index: {e.message}")

    def get_index(self, table_name: str = None, index_name: str = None) -> IndexConfig:
        """
        Get details of a specific index.

        Args:
            table_name: Name of the table
            index_name: Name of the index

        Returns:
            Index configuration
        """
        table_name = table_name or self.default_table
        index_name = index_name or self.default_index or self.full_text_index

        response = self._make_request("GET", f"/table/{table_name}/index/{index_name}")
        return IndexConfig(**response.json())

    def list_indexes(self, table_name: Optional[str] = None) -> List[IndexConfig]:
        """
        List all indexes for a table.

        Args:
            table_name: Name of the table (uses default if not provided)

        Returns:
            List of index configurations
        """
        table_name = table_name or self.default_table
        if not table_name:
            raise AntflyValidationError("table_name is required")

        response = self._make_request("GET", f"/table/{table_name}/index")
        return [IndexConfig(**index_data) for index_data in response.json()]

    def delete_index(self, table_name: str, index_name: str) -> None:
        """
        Delete an index from a table.

        Args:
            table_name: Name of the table
            index_name: Name of the index to delete
        """
        try:
            self._make_request("DELETE", f"/table/{table_name}/index/{index_name}")
        except AntflyHTTPError as e:
            raise AntflyIndexError(f"Failed to delete index: {e.message}")

    # Embedding Methods

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
        input_type: str = "query"
    ) -> List[float]:
        """
        Generate embeddings for text using Ollama.

        Args:
            text: Text to embed
            model: Model to use (defaults to env var OLLAMA_MODEL)
            input_type: Type of input (currently only "query" is supported)

        Returns:
            Embedding vector

        Raises:
            NotImplementedError: If input_type is not "query"
        """
        if input_type != "query":
            raise NotImplementedError("Only 'query' input_type is currently supported")

        # Use environment variables for Ollama configuration
        ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
        ollama_port = int(os.environ.get("OLLAMA_PORT", 11434))
        ollama_model = model or os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
        ollama_url = f"http://{ollama_host}:{ollama_port}/api/embed"

        try:
            response = self.client.post(
                ollama_url,
                json={"model": ollama_model, "input": text},
                timeout=10.0
            )
            response.raise_for_status()

            result = response.json()

            if 'embeddings' not in result or not result['embeddings']:
                raise AntflyHTTPError(f"Unexpected response from Ollama: {result}", 500)

            return result['embeddings'][0]

        except httpx.RequestError as e:
            raise AntflyConnectionError(f"Failed to connect to Ollama: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise AntflyHTTPError(f"Ollama error: {e.response.text}", e.response.status_code)

    # Query Methods

    def query(
        self,
        table: Optional[str] = None,
        full_text_search: Optional[str] = None,
        semantic_search: Optional[str] = None,
        indexes: Optional[List[str]] = None,
        filter_prefix: Optional[str] = None,
        embeddings: Optional[Dict[str, List[float]]] = None,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Dict[str, bool]] = None,
        count: Optional[bool] = None,
    ) -> QueryResult:
        """
        Execute a query across tables and indexes.

        Args:
            table: Table name (uses default if not provided)
            full_text_search: Full-text search query string
            semantic_search: Semantic search query string
            indexes: List of index names to search
            filter_prefix: Prefix to filter keys by
            embeddings: Dictionary of index name to embedding vectors
            fields: List of fields to return in results
            limit: Maximum number of results to return
            offset: Offset to start returning results from
            order_by: Dictionary specifying result ordering
            count: Whether to return total count of results

        Returns:
            Query results
        """
        table = table or self.default_table
        if not table:
            raise AntflyValidationError("table is required")

        if fields is None:
            fields = ["title"]

        request_data = QueryRequest(
            table=table,
            full_text_search=full_text_search,
            semantic_search=semantic_search,
            indexes=indexes,
            filter_prefix=filter_prefix,
            embeddings=embeddings,
            fields=fields,
            limit=limit,
            offset=offset,
            order_by=order_by,
            count=count,
        )

        # Clean up the request data - remove None values and empty lists
        request_dict = {
            k: v for k, v in request_data.dict().items()
            if v is not None and v != [] and v != ""
        }

        # Handle special case for default full_text_index
        if 'indexes' in request_dict and request_dict['indexes'] == ['full_text_index']:
            del request_dict['indexes']

        response = self._make_request("POST", "/query", json_data=request_dict)
        return QueryResult(**response.json())

    def semantic_search(
        self,
        query_text: str,
        table: Optional[str] = None,
        index_name: Optional[str] = None,
        limit: int = 10,
        fields: Optional[List[str]] = None
    ) -> QueryResult:
        """
        Perform semantic search using text query.

        Args:
            query_text: Text to search for
            table: Table name (uses default if not provided)
            index_name: Index name (uses default if not provided)
            limit: Maximum number of results
            fields: Fields to return

        Returns:
            Query results
        """
        table = table or self.default_table
        index_name = index_name or self.default_index

        if not table:
            raise AntflyValidationError("table is required")
        if not index_name:
            raise AntflyValidationError("index_name is required")

        # Generate embedding for the query
        query_vector = self.embed(query_text)

        return self.query(
            table=table,
            embeddings={index_name: query_vector},
            limit=limit,
            fields=fields or ["title"]
        )

    # Batch Operations

    def batch_upsert(
        self,
        vectors: List[Vector],
        table_name: Optional[str] = None,
    ) -> UpsertResult:
        """
        Batch upsert vectors into a table.

        Args:
            vectors: List of vectors to upsert (minimum 2 required)
            table_name: Name of the table (uses default if not provided)

        Returns:
            Upsert result

        Raises:
            AntflyValidationError: If less than 2 vectors provided
        """
        table_name = table_name or self.default_table
        if not table_name:
            raise AntflyValidationError("table_name is required")

        if len(vectors) < 2:
            raise AntflyValidationError("Batch upsert requires at least 2 vectors")

        # Transform vectors to Antfly's expected format
        inserts = {}
        for vector in vectors:
            doc = {**(vector.metadata or {})}
            if vector.values:
                doc["_embedding"] = vector.values
            inserts[vector.id] = doc

        batch_request = BatchRequest(inserts=inserts)

        try:
            response = self._make_request(
                "POST",
                f"/table/{table_name}/batch",
                json_data=batch_request.dict()
            )

            return UpsertResult(
                upserted_count=len(vectors),
                response=response.json()
            )

        except AntflyHTTPError as e:
            raise AntflyTableError(f"Batch upsert failed: {e.message}")

    def batch_delete(self, keys: List[str], table_name: Optional[str] = None) -> DeleteResult:
        """
        Batch delete documents from a table.

        Args:
            keys: List of document keys to delete
            table_name: Name of the table (uses default if not provided)

        Returns:
            Delete result
        """
        table_name = table_name or self.default_table
        if not table_name:
            raise AntflyValidationError("table_name is required")

        if not keys:
            return DeleteResult(deleted_count=0)

        batch_request = BatchRequest(deletes=keys)

        try:
            response = self._make_request(
                "POST",
                f"/table/{table_name}/batch",
                json_data=batch_request.dict()
            )

            return DeleteResult(
                deleted_count=len(keys),
                response=response.json()
            )

        except AntflyHTTPError as e:
            raise AntflyTableError(f"Batch delete failed: {e.message}")

    # Data Loading (CLI Integration)

    def load_data(
        self,
        file_path: str,
        table_name: str,
        id_field: str,
        batches: int = 10,
        size: int = 1000,
    ) -> None:
        """
        Load data from a JSON file using the antflycli command.

        Args:
            file_path: Path to the JSON file
            table_name: Name of the table to load data into
            id_field: Field in the JSON to use as document ID
            batches: Number of batches to split the data into
            size: Size of each batch

        Raises:
            AntflyValidationError: If antflycli is not found or command fails
        """
        command = [
            "antflycli", "load",
            "--table", table_name,
            "--file-path", file_path,
            "--id-field", id_field,
            "--batches", str(batches),
            "--size", str(size)
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            raise AntflyValidationError(
                "antflycli not found. Make sure it is installed and in your PATH."
            )
        except subprocess.CalledProcessError as e:
            raise AntflyValidationError(f"antflycli failed: {e.stderr}")

    # Utility Methods

    def wait_for_index_ready(
        self,
        table_name: str,
        index_name: str,
        expected_vector_count: Optional[int] = None,
        timeout: float = 120.0,
        poll_interval: float = 3.0
    ) -> None:
        """
        Wait for an index to be ready with vectors.

        Args:
            table_name: Name of the table
            index_name: Name of the index
            expected_vector_count: Expected number of vectors (optional)
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds

        Raises:
            TimeoutError: If index is not ready within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                index = self.get_index(table_name, index_name)
                if index.status and 'active_vectors' in index.status:
                    active_vectors = index.status['active_vectors']

                    if expected_vector_count is None:
                        # Just check if there are any vectors
                        if active_vectors > 0:
                            return
                    else:
                        # Check if we have the expected count
                        if active_vectors >= expected_vector_count:
                            return

                time.sleep(poll_interval)

            except Exception:
                # Continue polling on errors
                time.sleep(poll_interval)

        raise TimeoutError(f"Index {index_name} not ready within {timeout} seconds")

    def health_check(self) -> bool:
        """
        Check if the Antfly service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to list tables as a basic health check
            self.list_tables()
            return True
        except Exception:
            return False
