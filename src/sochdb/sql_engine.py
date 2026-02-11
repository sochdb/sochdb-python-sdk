# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal SQL executor for the Python SDK.

Stores schema and rows in SochDB using path-native keys:
- Schema: _sql/tables/{table}/schema
- Rows:   _sql/tables/{table}/rows/{row_id}
- Seq:    _sql/tables/{table}/sequence
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .errors import DatabaseError
from .query import SQLQueryResult


class SQLExecutor:
    def __init__(self, db_like):
        self._db = db_like

    def execute(self, sql: str) -> SQLQueryResult:
        if not sql or not sql.strip():
            raise DatabaseError("Empty SQL statement")
        sql = sql.strip().rstrip(";")
        keyword = sql.split(None, 1)[0].upper()

        if keyword == "SELECT":
            return self._execute_select(sql)
        if keyword == "INSERT":
            return self._execute_insert(sql)
        if keyword == "UPDATE":
            return self._execute_update(sql)
        if keyword == "DELETE":
            return self._execute_delete(sql)
        if keyword == "CREATE":
            return self._execute_create(sql)
        if keyword == "DROP":
            return self._execute_drop(sql)

        raise DatabaseError(f"Unsupported SQL statement: {keyword}")

    # ---------------------------------------------------------------------
    # CREATE / DROP
    # ---------------------------------------------------------------------

    def _execute_create(self, sql: str) -> SQLQueryResult:
        match = re.match(r"(?is)^create\s+table\s+(if\s+not\s+exists\s+)?(?P<table>\w+)\s*\((?P<body>.*)\)\s*$", sql)
        if not match:
            raise DatabaseError("Only CREATE TABLE is supported")

        table = match.group("table")
        body = match.group("body").strip()
        schema_path = self._schema_path(table)

        if match.group(1):
            if self._get_path(schema_path) is not None:
                return SQLQueryResult(rows_affected=0)

        columns, primary_key = self._parse_table_schema(body)
        schema = {
            "table": table,
            "columns": columns,
            "primary_key": primary_key,
        }
        self._put_path(schema_path, json.dumps(schema).encode("utf-8"))
        return SQLQueryResult(rows_affected=0)

    def _execute_drop(self, sql: str) -> SQLQueryResult:
        match = re.match(r"(?is)^drop\s+table\s+(if\s+exists\s+)?(?P<table>\w+)\s*$", sql)
        if not match:
            raise DatabaseError("Only DROP TABLE is supported")

        table = match.group("table")
        schema_path = self._schema_path(table)

        schema = self._get_schema(table)
        if schema is None:
            if match.group(1):
                return SQLQueryResult(rows_affected=0)
            raise DatabaseError(f"Table not found: {table}")

        for row_key, _ in self._scan_rows(table):
            self._delete_path(row_key)
        self._delete_path(schema_path)
        self._delete_path(self._sequence_path(table))
        return SQLQueryResult(rows_affected=0)

    # ---------------------------------------------------------------------
    # INSERT
    # ---------------------------------------------------------------------

    def _execute_insert(self, sql: str) -> SQLQueryResult:
        match = re.match(
            r"(?is)^insert\s+into\s+(?P<table>\w+)\s*\((?P<cols>[^)]*)\)\s*values\s*\((?P<vals>.*)\)\s*$",
            sql,
        )
        if not match:
            raise DatabaseError("Invalid INSERT syntax")

        table = match.group("table")
        columns = [c.strip() for c in match.group("cols").split(",") if c.strip()]
        values = self._split_csv(match.group("vals"))

        if len(columns) != len(values):
            raise DatabaseError("Column count does not match value count")

        schema = self._require_schema(table)
        row = {col: self._parse_value(val) for col, val in zip(columns, values)}

        primary_key = schema.get("primary_key")
        if primary_key:
            if primary_key in row and row[primary_key] is not None:
                row_id = row[primary_key]
            else:
                row_id = self._next_sequence(table)
                row[primary_key] = row_id
        else:
            row_id = self._next_sequence(table)

        for col in schema["columns"]:
            name = col["name"]
            if name not in row:
                row[name] = None

        row_path = self._row_path(table, row_id)
        self._put_path(row_path, json.dumps(row).encode("utf-8"))
        return SQLQueryResult(rows_affected=1)

    # ---------------------------------------------------------------------
    # SELECT
    # ---------------------------------------------------------------------

    def _execute_select(self, sql: str) -> SQLQueryResult:
        select_match = re.match(
            r"(?is)^select\s+(?P<select>.+?)\s+from\s+(?P<table>\w+)(?P<rest>.*)$",
            sql,
        )
        if not select_match:
            raise DatabaseError("Invalid SELECT syntax")

        table = select_match.group("table")
        select_clause = select_match.group("select").strip()
        rest = select_match.group("rest")

        schema = self._require_schema(table)
        rows = self._load_rows(table)

        clauses = self._split_clauses(rest)
        if clauses.get("where"):
            conditions = self._parse_conditions(clauses["where"])
            rows = [row for row in rows if self._match_row(row, conditions)]

        if select_clause.upper().startswith("COUNT("):
            alias = self._count_alias(select_clause)
            return SQLQueryResult(rows=[{alias: len(rows)}], columns=[alias], rows_affected=0)

        selected_columns = self._parse_select_columns(select_clause, schema)

        if clauses.get("order"):
            order_col, order_desc = self._parse_order_by(clauses["order"])
            rows.sort(key=lambda r: (r.get(order_col) is None, r.get(order_col)), reverse=order_desc)

        offset = int(clauses.get("offset") or 0)
        limit = int(clauses.get("limit") or 0)
        if offset:
            rows = rows[offset:]
        if limit:
            rows = rows[:limit]

        result_rows = []
        for row in rows:
            projected = {col: row.get(col) for col in selected_columns}
            result_rows.append(projected)

        return SQLQueryResult(rows=result_rows, columns=selected_columns, rows_affected=0)

    # ---------------------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------------------

    def _execute_update(self, sql: str) -> SQLQueryResult:
        match = re.match(
            r"(?is)^update\s+(?P<table>\w+)\s+set\s+(?P<set>.+?)(?P<rest>\s+where\s+.+)?$",
            sql,
        )
        if not match:
            raise DatabaseError("Invalid UPDATE syntax")

        table = match.group("table")
        set_clause = match.group("set").strip()
        rest = match.group("rest") or ""

        schema = self._require_schema(table)
        rows = self._load_rows_with_keys(table)

        conditions = []
        if rest:
            where_clause = rest.strip()[5:].strip()  # strip leading WHERE
            conditions = self._parse_conditions(where_clause)

        assignments = self._parse_assignments(set_clause)
        updated = 0

        for row_key, row in rows:
            if conditions and not self._match_row(row, conditions):
                continue
            for col, expr in assignments:
                row[col] = self._evaluate_update_expression(row, col, expr)
            self._put_path(row_key, json.dumps(row).encode("utf-8"))
            updated += 1

        return SQLQueryResult(rows_affected=updated)

    # ---------------------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------------------

    def _execute_delete(self, sql: str) -> SQLQueryResult:
        match = re.match(
            r"(?is)^delete\s+from\s+(?P<table>\w+)(?P<rest>\s+where\s+.+)?$",
            sql,
        )
        if not match:
            raise DatabaseError("Invalid DELETE syntax")

        table = match.group("table")
        rest = match.group("rest") or ""
        conditions = []
        if rest:
            where_clause = rest.strip()[5:].strip()
            conditions = self._parse_conditions(where_clause)

        deleted = 0
        for row_key, row in self._load_rows_with_keys(table):
            if conditions and not self._match_row(row, conditions):
                continue
            self._delete_path(row_key)
            deleted += 1

        return SQLQueryResult(rows_affected=deleted)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _schema_path(self, table: str) -> str:
        return f"_sql/tables/{table}/schema"

    def _rows_prefix(self, table: str) -> str:
        return f"_sql/tables/{table}/rows/"

    def _row_path(self, table: str, row_id: Any) -> str:
        return f"_sql/tables/{table}/rows/{row_id}"

    def _sequence_path(self, table: str) -> str:
        return f"_sql/tables/{table}/sequence"

    def _get_schema(self, table: str) -> Optional[Dict[str, Any]]:
        raw = self._get_path(self._schema_path(table))
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise DatabaseError(f"Invalid schema for table {table}: {exc}") from exc

    def _require_schema(self, table: str) -> Dict[str, Any]:
        schema = self._get_schema(table)
        if schema is None:
            raise DatabaseError(f"Table not found: {table}")
        return schema

    def _parse_table_schema(self, body: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        columns = []
        primary_key = None
        for part in self._split_csv(body):
            part = part.strip()
            if not part:
                continue
            if part.upper().startswith("PRIMARY KEY"):
                match = re.search(r"\((?P<col>[^)]+)\)", part)
                if match:
                    primary_key = match.group("col").strip()
                continue
            tokens = part.split()
            if len(tokens) < 2:
                raise DatabaseError(f"Invalid column definition: {part}")
            name = tokens[0]
            dtype = tokens[1].upper()
            col_primary = any(tok.upper() == "PRIMARY" for tok in tokens)
            if col_primary:
                primary_key = name
            columns.append({
                "name": name,
                "type": dtype,
                "primary_key": col_primary,
            })
        return columns, primary_key

    def _split_csv(self, value: str) -> List[str]:
        reader = csv.reader(io.StringIO(value), skipinitialspace=True)
        return next(reader, [])

    def _parse_value(self, raw: str) -> Any:
        raw = raw.strip()
        if raw.upper() == "NULL":
            return None
        if raw.upper() == "TRUE":
            return True
        if raw.upper() == "FALSE":
            return False
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            return raw[1:-1]
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        if re.fullmatch(r"-?\d+\.\d+", raw):
            return float(raw)
        return raw

    def _parse_select_columns(self, select_clause: str, schema: Dict[str, Any]) -> List[str]:
        if select_clause.strip() == "*":
            return [col["name"] for col in schema["columns"]]
        columns = []
        for part in self._split_csv(select_clause):
            part = part.strip()
            if not part:
                continue
            alias_match = re.match(r"(?is)^(?P<col>\w+)\s+as\s+(?P<alias>\w+)$", part)
            if alias_match:
                columns.append(alias_match.group("alias"))
            else:
                columns.append(part)
        return columns

    def _count_alias(self, select_clause: str) -> str:
        alias_match = re.search(r"(?is)\bas\s+(?P<alias>\w+)$", select_clause)
        if alias_match:
            return alias_match.group("alias")
        return "count"

    def _split_clauses(self, rest: str) -> Dict[str, str]:
        rest = rest or ""
        upper = rest.upper()
        matches = []
        for label, pattern in (
            ("where", r"\bWHERE\b"),
            ("order", r"\bORDER\s+BY\b"),
            ("limit", r"\bLIMIT\b"),
            ("offset", r"\bOFFSET\b"),
        ):
            match = re.search(pattern, upper)
            if match:
                matches.append((match.start(), match.end(), label))
        matches.sort(key=lambda m: m[0])
        clauses = {}
        for idx, (start, end, label) in enumerate(matches):
            next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(rest)
            clauses[label] = rest[end:next_start].strip()
        return clauses

    def _parse_conditions(self, clause: str) -> List[Tuple[str, str, Any]]:
        parts = self._split_and(clause)
        conditions = []
        for part in parts:
            part = part.strip()
            op = self._find_operator(part)
            if not op:
                raise DatabaseError(f"Invalid WHERE clause: {part}")
            left, right = part.split(op, 1)
            conditions.append((left.strip(), op, self._parse_value(right)))
        return conditions

    def _split_and(self, clause: str) -> List[str]:
        parts = []
        current = []
        i = 0
        in_quote = None
        while i < len(clause):
            ch = clause[i]
            if ch in ("'", '"'):
                if in_quote == ch:
                    in_quote = None
                elif in_quote is None:
                    in_quote = ch
            if in_quote is None and clause[i:i+3].upper() == "AND":
                before = clause[i-1] if i > 0 else " "
                after = clause[i+3] if i + 3 < len(clause) else " "
                if before.isspace() and after.isspace():
                    parts.append("".join(current).strip())
                    current = []
                    i += 3
                    continue
            current.append(ch)
            i += 1
        if current:
            parts.append("".join(current).strip())
        return [p for p in parts if p]

    def _find_operator(self, clause: str) -> Optional[str]:
        for op in (">=", "<=", "!=", "<>", "LIKE", "=", ">", "<"):
            if op in clause.upper():
                return op if op != "LIKE" else "LIKE"
        return None

    def _match_row(self, row: Dict[str, Any], conditions: List[Tuple[str, str, Any]]) -> bool:
        for col, op, value in conditions:
            col = col.strip()
            row_value = row.get(col)
            if op == "LIKE":
                if not isinstance(row_value, str):
                    return False
                pattern = re.escape(str(value)).replace("%", ".*").replace("_", ".")
                if not re.match(f"^{pattern}$", row_value):
                    return False
                continue
            if row_value is None or value is None:
                if op in ("=",) and row_value is None and value is None:
                    continue
                if op in ("!=", "<>") and row_value is None and value is None:
                    return False
                return False
            if op in ("=",):
                if row_value != value:
                    return False
            elif op in ("!=", "<>"):
                if row_value == value:
                    return False
            else:
                try:
                    left = float(row_value) if isinstance(row_value, (int, float)) else row_value
                    right = float(value) if isinstance(value, (int, float)) else value
                except Exception:
                    return False
                if op == ">" and not (left > right):
                    return False
                if op == ">=" and not (left >= right):
                    return False
                if op == "<" and not (left < right):
                    return False
                if op == "<=" and not (left <= right):
                    return False
        return True

    def _parse_order_by(self, clause: str) -> Tuple[str, bool]:
        tokens = clause.strip().split()
        if not tokens:
            raise DatabaseError("Invalid ORDER BY clause")
        column = tokens[0]
        desc = len(tokens) > 1 and tokens[1].upper() == "DESC"
        return column, desc

    def _parse_assignments(self, clause: str) -> List[Tuple[str, str]]:
        assignments = []
        for part in self._split_csv(clause):
            if "=" not in part:
                raise DatabaseError(f"Invalid SET clause: {part}")
            left, right = part.split("=", 1)
            assignments.append((left.strip(), right.strip()))
        return assignments

    def _evaluate_update_expression(self, row: Dict[str, Any], col: str, expr: str) -> Any:
        match = re.match(r"(?is)^(?P<base>\w+)\s*(?P<op>[+\-])\s*(?P<val>-?\d+(?:\.\d+)?)$", expr)
        if match and match.group("base") == col:
            base_val = row.get(col, 0) or 0
            delta = float(match.group("val")) if "." in match.group("val") else int(match.group("val"))
            if match.group("op") == "+":
                return base_val + delta
            return base_val - delta
        return self._parse_value(expr)

    def _scan_rows(self, table: str):
        prefix = self._rows_prefix(table).encode("utf-8")
        return list(self._db.scan_prefix(prefix))

    def _load_rows_with_keys(self, table: str) -> List[Tuple[str, Dict[str, Any]]]:
        rows = []
        for key, value in self._scan_rows(table):
            try:
                row = json.loads(value.decode("utf-8"))
            except Exception:
                continue
            rows.append((key.decode("utf-8"), row))
        return rows

    def _load_rows(self, table: str) -> List[Dict[str, Any]]:
        return [row for _, row in self._load_rows_with_keys(table)]

    def _next_sequence(self, table: str) -> int:
        seq_path = self._sequence_path(table)
        raw = self._get_path(seq_path)
        current = int(raw.decode("utf-8")) if raw else 0
        next_val = current + 1
        self._put_path(seq_path, str(next_val).encode("utf-8"))
        return next_val

    def _put_path(self, path: str, value: bytes) -> None:
        self._db.put_path(path, value)

    def _get_path(self, path: str) -> Optional[bytes]:
        return self._db.get_path(path)

    def _delete_path(self, path) -> None:
        # path may be str or bytes (e.g. from scan_prefix)
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        if hasattr(self._db, "delete_path"):
            self._db.delete_path(path)
        else:
            self._db.delete(path.encode("utf-8"))
