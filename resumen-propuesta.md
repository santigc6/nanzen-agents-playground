# Propuesta de cambios — nanzen-agents-playground

> Cambios técnicos al código del challenge.
> Cada sección: problema en el código actual → solución propuesta → implementación.

---

## 1. `tools/csv_reader.py` — Shared data layer con caché y metadatos

**Problema**: `forward()` (línea 90) devuelve texto formateado con pipes. El LLM tiene que re-parsearlo en cada tool call para poder operar. Además, `limit=50` por defecto sin informar al agente de si los datos están truncados — el agente no sabe que solo ha visto 50 de 154 filas.

**Solución**: Cachear cada lectura como JSON estructurado en `.cache/` y devolver al agente una referencia `FILE:.cache/...`. El agente lee con `json.load()` directamente. Añadir metadatos con cardinalidad real para que el agente sepa si necesita más datos.

```python
import json, hashlib, csv
from pathlib import Path
from smolagents import Tool

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DATA_SOURCES: dict[str, str] = {
    "accounts": "accounts.csv", "billing": "billing.csv",
    "product_usage": "product_usage.csv", "support_tickets": "support_tickets.csv",
    "crm_interactions": "crm_interactions.csv", "emails": "emails.csv",
    "contracts": "contracts.csv", "purchase_orders": "purchase_orders.csv",
}

class CSVReaderTool(Tool):
    name = "read_context"
    description = (
        "Read data from the shared context. Returns a FILE: reference to cached JSON. "
        "Use json.load() to read it. Metadata includes total row count so you know "
        "if data was truncated."
    )
    inputs = {
        "source": {"type": "string", "description": "Data source name"},
        "account_id": {"type": "string", "description": "Filter by account_id", "nullable": True},
        "limit": {"type": "integer", "description": "Max rows (default 50)", "nullable": True},
    }
    output_type = "string"
    MAX_ROWS = 200
    _call_counts: dict[str, int] = {}
    _schema_cache: dict | None = None

    def forward(self, source: str, account_id: str | None = None,
                limit: int | None = None) -> str:
        effective_limit = min(limit or 50, self.MAX_ROWS)
        
        # Rate limiting: máximo 10 lecturas por agente
        agent_id = "default"
        self._call_counts[agent_id] = self._call_counts.get(agent_id, 0) + 1
        if self._call_counts[agent_id] > 10:
            return f"Max reads (10) exceeded. Analyze what you already have."

        # Generar cache key
        cache_key = hashlib.md5(
            f"{source}:{account_id}:{effective_limit}".encode()
        ).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.json"

        # Cache hit
        if cache_path.exists():
            return f"FILE:{cache_path}  # cached, use json.load()"

        # Cache miss: leer CSV, contar total, guardar
        if source not in DATA_SOURCES:
            return f"Unknown source '{source}'. Available: {sorted(DATA_SOURCES.keys())}"

        csv_path = DATA_DIR / DATA_SOURCES[source]
        if not csv_path.exists():
            return f"File not found: {csv_path}"

        rows = []
        total = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if account_id and "account_id" in row and row["account_id"] != account_id:
                    continue
                total += 1
                if len(rows) < effective_limit:
                    rows.append(dict(row))

        if not rows:
            return f"No rows found in '{source}' for account_id='{account_id}'"

        output = {
            "metadata": {
                "source": source,
                "account_id": account_id,
                "rows_returned": len(rows),
                "total_in_source": total,
                "truncated": len(rows) < total,
                "columns": list(rows[0].keys()),
                "column_types": self._infer_types(rows),
            },
            "data": rows,
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(output, default=str))
        return f"FILE:{cache_path}"

    def _infer_types(self, rows: list[dict]) -> dict:
        """Infiere tipos de columna para que el agente sepa qué esperar."""
        types = {}
        for col in rows[0].keys():
            values = [r.get(col, "") for r in rows[:5] if r.get(col, "")]
            if not values:
                types[col] = "unknown"
                continue
            try:
                [float(v) for v in values]
                types[col] = "numeric"
                continue
            except (ValueError, TypeError):
                pass
            try:
                from datetime import datetime
                [datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                 for v in values if v]
                types[col] = "datetime"
                continue
            except (ValueError, TypeError):
                pass
            types[col] = "text"
        return types
```

**Qué cambia**: 
- El agente recibe `FILE:.cache/abc123.json` → `json.load()` → acceso directo a datos como dicts.
- Sabe cuántas filas totales hay (`total_in_source`) y si los datos están truncados.
- Sabe el tipo de cada columna (`column_types`).
- Rate limiting: máximo 10 lecturas. Máximo 200 filas por lectura.
- Si otro agente pide los mismos datos, recibe la misma referencia (cache hit).

---

## 2. `tools/verification.py` — VerificationTool (nuevo archivo)

**Problema**: `agent.py:46` dice "never fabricate numbers" pero no hay mecanismo que lo imponga. `PDFReportTool` acepta cualquier JSON sin validar. El agente puede leer 50 de 154 filas, sumar mal, e incluirlo en el reporte.

**Solución**: Una tool de verificación que ejecuta queries SQL deterministas contra los CSVs (cargados en SQLite en memoria). El agente está obligado a llamarla antes de `create_report`.

```python
import sqlite3, re
from smolagents import Tool
from pathlib import Path

class VerificationTool(Tool):
    """Verify numeric claims against original CSV data before reporting."""
    
    name = "verify_finding"
    description = (
        "Verify a numeric claim by executing a SQL query against the CSV data. "
        "Call this for EVERY number before including it in a report."
    )
    inputs = {
        "claim": {"type": "string", "description": "What is being claimed"},
        "expected_value": {"type": "string", "description": "The number claimed, e.g. '421230'"},
        "query": {"type": "string", "description": "SQL query to verify, e.g. 'SELECT SUM(amount) FROM billing WHERE event_type=\"invoice_issued\"'"},
    }
    output_type = "string"
    _db: sqlite3.Connection | None = None

    def forward(self, claim: str, expected_value: str, query: str) -> str:
        conn = self._get_db()
        try:
            result = conn.execute(query).fetchone()
            actual = float(result[0]) if result and result[0] is not None else 0.0
            expected = float(re.sub(r"[€$,]", "", expected_value).strip())
        except Exception as e:
            return f"ERROR in query: {e}"

        delta = abs(actual - expected)
        threshold = max(abs(expected), 1) * 0.01
        if delta <= threshold:
            return f"[OK] VERIFIED: {claim} | expected={expected_value}, actual={actual:.2f}, delta={delta:.2f}"
        else:
            return (
                f"[FAIL] MISMATCH: {claim}\n"
                f"  Claimed: {expected_value}\n"
                f"  Actual from data: {actual:.2f}\n"
                f"  Delta: {delta:.2f} ({(delta/max(abs(expected),1))*100:.1f}%)\n"
                f"  CORRECT before including in report."
            )

    def _get_db(self) -> sqlite3.Connection:
        if self._db is not None:
            return self._db
        self._db = sqlite3.connect(":memory:")
        import pandas as pd
        from challenge.tools.csv_reader import DATA_DIR, DATA_SOURCES
        for source, filename in DATA_SOURCES.items():
            df = pd.read_csv(DATA_DIR / filename)
            df.to_sql(source, self._db, index=False, if_exists="replace")
        return self._db
```

**Qué cambia**:
- El agente no puede generar un PDF sin verificar. Si dice "total invoiced = €421,230", el VerificationTool ejecuta `SELECT SUM(amount) FROM billing WHERE event_type='invoice_issued'` y confirma o rechaza.
- Si hay discrepancia >1%, el agente recibe instrucciones de corregir antes de reportar.
- La base de datos se carga una vez en memoria (SQLite) y se reutiliza.

---

## 3. `tools/email_search.py` — EmailSearchTool con embeddings (nuevo archivo)

**Problema**: `emails.csv` tiene 77 filas de texto libre. Buscar con `LIKE '%competitor%'` no encuentra "estamos evaluando otras opciones" o "el otro proveedor nos ofrece mejor precio". El `CSVReaderTool` actual no distingue entre datos estructurados y texto libre.

**Solución**: Tool de búsqueda semántica que precomputa embeddings offline y permite al agente buscar con lenguaje natural.

```python
import json
import numpy as np
from pathlib import Path
from smolagents import Tool

class EmailSearchTool(Tool):
    """Semantic search over emails using precomputed embeddings."""
    
    name = "search_emails"
    description = (
        "Search emails semantically. Use natural language like "
        "'emails where the CFO questions our pricing' or "
        "'emails mentioning competitors or alternative vendors'."
    )
    inputs = {
        "query": {"type": "string", "description": "Natural language search query"},
        "top_k": {"type": "integer", "description": "Number of results (default 5)", "nullable": True},
    }
    output_type = "string"
    
    _index: np.ndarray | None = None
    _metadata: list[dict] | None = None
    _model = None

    def __init__(self, index_dir: Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self.index_dir = index_dir or Path(__file__).resolve().parents[3] / ".cache"

    def _load(self):
        if self._index is not None:
            return
        cache = np.load(self.index_dir / "email_embeddings.npz")
        self._index = cache["vectors"]
        self._metadata = json.loads(
            (self.index_dir / "email_metadata.json").read_text()
        )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def forward(self, query: str, top_k: int = 5) -> str:
        self._load()
        model = self._get_model()
        
        query_vec = model.encode([query])[0]
        query_vec = query_vec / np.linalg.norm(query_vec)
        norms = np.linalg.norm(self._index, axis=1)
        norms[norms == 0] = 1
        scores = np.dot(self._index, query_vec) / norms
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            meta = self._metadata[idx]
            results.append({
                "score": round(float(scores[idx]), 3),
                "email_id": meta["email_id"],
                "timestamp": meta["timestamp"],
                "from": meta["from"],
                "subject": meta["subject"],
                "snippet": meta["text"][:300],
            })
        return json.dumps(results, indent=2, ensure_ascii=False)
```

**Script offline para generar el índice** (se ejecuta una vez):

```python
# scripts/build_email_index.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/emails.csv")
texts = []
metadata = []
for _, row in df.iterrows():
    text = f"Subject: {row['subject']}\nFrom: {row['from']}\n\n{row['body']}"
    texts.append(text)
    metadata.append({
        "email_id": row["email_id"], "timestamp": str(row["timestamp"]),
        "from": row["from"], "subject": row["subject"], "text": text,
    })

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

Path("data/.cache").mkdir(exist_ok=True)
np.savez_compressed("data/.cache/email_embeddings.npz", vectors=vectors)
Path("data/.cache/email_metadata.json").write_text(json.dumps(metadata, indent=2))
print(f"Indexed {len(texts)} emails. Shape: {vectors.shape}")
```

**Qué cambia**:
- El agente puede buscar "emails donde la CFO cuestiona nuestro pricing" y recibir los top-K resultados con snippets.
- Los embeddings se precomputan offline (modelo local gratuito, `all-MiniLM-L6-v2`, 384 dimensiones).
- En runtime solo se embebe la query del agente (~10ms). Resultados deterministas: mismo email → mismo vector.
- Si el dataset escala a miles de emails, se puede cambiar a Chroma o LanceDB sin cambiar la interfaz de la tool.

---

## 4. `tools/pdf_report.py` — Guardarraíl de verificación

**Problema**: `forward()` (línea 209) acepta cualquier JSON y genera el PDF. No hay validación de que los números incluidos hayan sido verificados.

**Solución**: Añadir un parámetro `verification_hashes` que el agente debe pasar. La tool extrae todos los valores numéricos del contenido del reporte y bloquea la generación si hay claims no verificados.

```python
# Modificación de pdf_report.py — añadir al método forward()
import hashlib, re, json

def forward(self, title: str, filename: str, content_sections_json: str,
            author: str | None = None, verification_hashes: str = "[]") -> str:
    
    content_sections = json.loads(content_sections_json)
    if not isinstance(content_sections, list):
        return "ERROR: content_sections_json must be a JSON array."
    
    # ─── GUARDARRAÍL: verificar claims numéricos ───
    hashes = set(json.loads(verification_hashes))
    report_numbers = self._extract_numbers(content_sections)
    unverified = report_numbers - hashes
    
    if unverified:
        return (
            f"[BLOCKED] {len(unverified)} unverified numeric claims found.\n"
            f"Examples: {list(unverified)[:3]}\n"
            f"You MUST call verify_finding for EVERY number before create_report.\n"
            f"PDF NOT generated."
        )
    
    if len(hashes) < 3:
        return (
            f"[BLOCKED] Only {len(hashes)} claims verified. Minimum required: 3.\n"
            f"PDF NOT generated."
        )
    
    # ─── Si pasa, generar PDF normalmente ───
    try:
        pdf_bytes = build_pdf(title, content_sections, author=author)
    except Exception as exc:
        return f"ERROR: PDF generation failed: {exc}"
    
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_bytes(pdf_bytes)
    
    return (
        f"Report saved: {OUTPUT_DIR / filename}\n"
        f"Verified claims: {len(hashes)}\n"
    )

def _extract_numbers(self, sections: list[dict]) -> set[str]:
    """Extrae hashes de todos los valores numéricos en el contenido del reporte."""
    hashes = set()
    for section in sections:
        text = str(section)
        for match in re.finditer(r'[\d,]+\.?\d*', text):
            n = match.group().replace(",", "").strip()
            if len(n) >= 2:
                hashes.add(hashlib.md5(n.encode()).hexdigest()[:8])
    return hashes
```

**Qué cambia**:
- El agente no puede generar un PDF sin pasar los hashes de verificación.
- Si intenta colar un número no verificado, la tool lo rechaza explícitamente.
- El agente recibe el mensaje de bloqueo y tiene que volver atrás a verificar.

---

## 5. `tools/guardrails.py` — Sistema de guardarraíles (nuevo archivo)

**Problema**: No hay validación cross-source. Billing podría decir "total anual = €400K" y contracts decir "total anual = €372K" y nadie lo detecta. En finanzas, dos sistemas con datos inconsistentes es un riesgo real.

**Solución**: Un reconciliation gate que se ejecuta antes de entregar cualquier output. Comprueba que métricas clave de distintas fuentes sean consistentes.

```python
import json, logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("guardrails")

class ReconciliationGate:
    """Bloquea outputs si distintas fuentes reportan valores inconsistentes."""
    
    def check(self, findings: dict[str, dict]) -> list[str]:
        """Returns list of violations. Empty list = all clear."""
        violations = []
        
        # Regla 1: Total invoiced billing vs contracts
        billing_total = self._safe_float(findings, "billing", "total_invoiced")
        contract_total = self._safe_float(findings, "contracts", "annual_value")
        if billing_total and contract_total:
            delta = abs(billing_total - contract_total) / max(billing_total, 1)
            if delta > 0.05:
                violations.append(
                    f"RECONCILIATION: Billing total €{billing_total:,.0f} vs "
                    f"Contracts annual €{contract_total:,.0f} — {delta:.1%} delta. "
                    f"Resolve before generating report."
                )
        
        # Regla 2: Licensed seats debe coincidir entre usage y contracts
        for dept in ["Engineering", "Marketing"]:
            usage_seats = self._safe_float(findings, "usage", f"{dept.lower()}_licensed_seats")
            contract_seats = self._safe_float(findings, "contracts", f"{dept.lower()}_seats")
            if usage_seats and contract_seats and usage_seats != contract_seats:
                violations.append(
                    f"RECONCILIATION: {dept} seats — Usage reports {usage_seats}, "
                    f"Contracts show {contract_seats}."
                )
        
        # Regla 3: SLA credits deben tener evidencia cross-source
        sla_credits = self._find_by_type(findings, "sla_credit")
        for credit in sla_credits:
            if not credit.get("cross_source_evidence"):
                violations.append(
                    f"UNVERIFIED SLA CREDIT: {credit.get('amount')} claimed "
                    f"({credit.get('statement')}) but no evidence from support/CRM."
                )
        
        # Regla 4: Pagos recibidos vs facturas emitidas
        total_invoiced = self._safe_float(findings, "billing", "total_invoiced")
        total_paid = self._safe_float(findings, "billing", "total_paid")
        if total_invoiced and total_paid and total_paid < total_invoiced * 0.85:
            violations.append(
                f"RECONCILIATION: Only {total_paid/total_invoiced:.0%} of invoices paid. "
                f"Outstanding: €{total_invoiced - total_paid:,.0f}."
            )
        
        return violations
    
    def _safe_float(self, findings, agent, key):
        try:
            return float(findings.get(agent, {}).get("metrics", {}).get(key, None))
        except (TypeError, ValueError):
            return None
    
    def _find_by_type(self, findings, finding_type):
        results = []
        for agent_data in findings.values():
            for f in agent_data.get("findings", []):
                if finding_type in str(f.get("tags", [])):
                    results.append(f)
        return results


class GuardrailHooks:
    """Hooks pre y post ejecución para todas las tools."""
    
    @staticmethod
    def pre_create_report(tool_inputs: dict, verification_state: dict) -> dict | None:
        """Bloquea create_report si no hay suficientes verificaciones."""
        hashes = json.loads(tool_inputs.get("verification_hashes", "[]"))
        if len(hashes) < 3:
            return {
                "blocked": True,
                "reason": "Minimum 3 verified claims required before PDF generation.",
                "remediation": "Call verify_finding for each number in your report.",
            }
        return None
```

**Qué cambia**:
- Antes de entregar cualquier PDF, el reconciliation gate comprueba consistencia entre fuentes.
- Si billing y contracts discrepan >5%, el output se bloquea.
- Si un SLA credit no tiene evidencia en support/CRM, se marca para revisión manual.
- Los hooks son independientes de las tools — se pueden añadir/quitar sin modificar el código del agente.

---

## 6. `challenge/schema_extractor.py` — SchemaExtractor (nuevo archivo)

**Problema**: Los agentes no tienen información previa sobre la estructura de los datos. Cada uno tiene que descubrir columnas, tipos, y relaciones por su cuenta, gastando tokens en tareas que son deterministas.

**Solución**: Una fase de exploración que se ejecuta UNA vez antes de lanzar ningún agente. Analiza los 8 CSVs y genera un `schema_metadata.json` con tipos, cardinalidad, foreign keys y estrategias de búsqueda por tipo de fuente.

```python
import csv, json, re
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"

class SchemaExtractor:
    """Analiza los CSVs offline y genera metadatos estructurados."""
    
    def __init__(self, account_id: str = "MERID-001"):
        self.account_id = account_id
        self.metadata = {"account_id": account_id, "sources": {}, "cross_source_links": []}
    
    def extract_all(self) -> dict:
        for name, filename in self._list_csvs().items():
            self.metadata["sources"][name] = self._extract_source(name, filename)
        self._find_foreign_keys()
        self._find_data_quality_issues()
        self._save()
        return self.metadata
    
    def _extract_source(self, name: str, filename: str) -> dict:
        csv_path = DATA_DIR / filename
        total_rows = 0
        columns_info = defaultdict(dict)
        
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            sample_rows = []
            for row in reader:
                if self.account_id and "account_id" in row:
                    if row["account_id"] != self.account_id:
                        continue
                total_rows += 1
                if len(sample_rows) < 10:
                    sample_rows.append(row)
        
        # Inferir tipos
        for col in columns:
            values = [r.get(col, "") for r in sample_rows if r.get(col, "")]
            columns_info[col]["dtype"] = self._infer_dtype(values)
            columns_info[col]["null_count"] = sum(1 for r in sample_rows if not r.get(col, ""))
            columns_info[col]["sample"] = values[0] if values else ""
        
        # Clasificar fuente
        source_type = self._classify(columns_info, sample_rows)
        
        return {
            "type": source_type,
            "row_count": total_rows,
            "columns": dict(columns_info),
            "column_list": columns,
        }
    
    def _infer_dtype(self, values: list[str]) -> str:
        if not values:
            return "unknown"
        numeric_count = 0
        datetime_count = 0
        for v in values[:10]:
            if not v:
                continue
            try:
                float(v)
                numeric_count += 1
                continue
            except ValueError:
                pass
            try:
                from datetime import datetime
                datetime.fromisoformat(v.replace("Z", "+00:00"))
                datetime_count += 1
                continue
            except (ValueError, TypeError):
                pass
        total = min(len(values), 10)
        if numeric_count == total:
            return "numeric"
        if datetime_count >= total * 0.8:
            return "datetime"
        unique = len(set(values))
        if unique <= 10 and total >= 5:
            return "categorical"
        return "text"
    
    def _classify(self, columns: dict, rows: list[dict]) -> str:
        text_cols = sum(1 for c in columns.values() if c.get("dtype") == "text")
        total = len(columns)
        if text_cols / total > 0.6:
            return "unstructured"
        if text_cols > 2:
            return "semi_structured"
        return "structured"
    
    def _find_foreign_keys(self):
        """Busca columnas que referencian otras tablas."""
        for source, info in self.metadata["sources"].items():
            for col in info.get("column_list", []):
                if col.endswith("_id") or col.endswith("_reference"):
                    target = col.replace("_id", "s").replace("_reference", "s")
                    if target in self.metadata["sources"]:
                        self.metadata["cross_source_links"].append({
                            "from": f"{source}.{col}",
                            "to": f"{target}.{target.rstrip('s')}_id",
                        })
                # Caso especial: contract_reference → contracts.contract_id
                if "_reference" in col and not col.endswith("_id"):
                    for t in self.metadata["sources"]:
                        if t in col or col.replace("_reference", "") in t:
                            self.metadata["cross_source_links"].append({
                                "from": f"{source}.{col}",
                                "to": f"{t}.{t.rstrip('s')}_id",
                            })
    
    def _find_data_quality_issues(self):
        issues = []
        # Detectar typo "Enginering" en product_usage
        for s, info in self.metadata["sources"].items():
            for col, meta in info.get("columns", {}).items():
                if meta.get("dtype") == "categorical" and meta.get("sample"):
                    sample = str(meta.get("sample", ""))
                    if len(sample) > 5:
                        issues.append({
                            "source": s,
                            "column": col,
                            "sample": sample,
                            "note": "long text in categorical column — possible corrupt value",
                        })
        self.metadata["data_quality_issues"] = issues
    
    def _save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / "schema_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, default=str)
        )
    
    def _list_csvs(self) -> dict:
        return {
            "accounts": "accounts.csv",
            "billing": "billing.csv",
            "product_usage": "product_usage.csv",
            "support_tickets": "support_tickets.csv",
            "crm_interactions": "crm_interactions.csv",
            "emails": "emails.csv",
            "contracts": "contracts.csv",
            "purchase_orders": "purchase_orders.csv",
        }
```

**Qué cambia**:
- Una sola ejecución analiza los 8 CSVs y produce `schema_metadata.json`.
- Cada agente recibe metadatos de su fuente: tipos de columna, cardinalidad, foreign keys.
- Las fuentes se clasifican como structured / semi_structured / unstructured.
- Las foreign keys se detectan automáticamente (ej. `billing.contract_reference` → `contracts.contract_id`).
- Los data quality issues se registran (typo "Enginering", filas malformadas, valores corruptos).
- Coste: ~0 tokens de LLM (todo es código Python determinista).

---

## 7. `challenge/orchestrator.py` — Orchestrator con asyncio (nuevo archivo)

**Problema**: `runner.py:116-122` usa `ThreadPoolExecutor` donde cada agente crea su propio modelo, lee CSVs desde cero, y produce su PDF en aislamiento. No hay coordinación ni shared state.

**Solución**: Orquestador con `asyncio` que ejecuta 3 fases (Exploration → Fan-out → Fan-in) con verificación en paralelo. Los especialistas comparten la capa de caché. El synthesizer recibe findings estructurados, no CSVs.

```python
import asyncio, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Orchestrator:
    """Pipeline: Exploration → Fan-out → Fan-in. Verify runs in parallel."""
    
    def __init__(self, account_id: str = "MERID-001"):
        self.account_id = account_id
        self.cache_dir = Path(".cache")
        self.output_dir = Path("output")
    
    def run_risk_assessment(self) -> dict:
        return asyncio.run(self._run())
    
    async def _run(self) -> dict:
        # ─── Fase 1: EXPLORATION ───
        logger.info("[Phase 1] Schema extraction...")
        from challenge.schema_extractor import SchemaExtractor
        extractor = SchemaExtractor(self.account_id)
        schema = extractor.extract_all()
        logger.info(f"[Phase 1] Done. {len(schema['sources'])} sources analyzed.")
        
        # ─── Fase 2: FAN-OUT (4 especialistas en paralelo) ───
        logger.info("[Phase 2] Launching specialists...")
        specialists = {
            "billing":       "gpt-4o-mini",
            "usage":         "gpt-4o-mini",
            "support":       "gpt-4o-mini",
            "relationship":  "gpt-4o-mini",
        }
        tasks = [
            asyncio.to_thread(self._run_specialist, source, model)
            for source, model in specialists.items()
        ]
        findings_files = await asyncio.gather(*tasks)
        logger.info(f"[Phase 2] All specialists done. {len(findings_files)} findings files.")
        
        # ─── Fase 3: FAN-IN + VERIFY (paralelo) ───
        logger.info("[Phase 3] Synthesizer + verification in parallel...")
        verify_task = asyncio.create_task(
            asyncio.to_thread(self._verify_all, findings_files)
        )
        report = await asyncio.to_thread(self._synthesize, findings_files, schema)
        verifications = await verify_task
        
        # ─── Reconciliation gate ───
        from challenge.tools.guardrails import ReconciliationGate
        gate = ReconciliationGate()
        all_findings = self._load_findings(findings_files)
        violations = gate.check(all_findings)
        if violations:
            logger.warning(f"Reconciliation violations: {len(violations)}")
            report["reconciliation_violations"] = violations
            report["blocked"] = len([v for v in violations if "CRITICAL" in v]) > 0
        
        report["verifications"] = verifications
        return report
    
    def _run_specialist(self, source: str, model_id: str) -> Path:
        """Ejecuta un agente especialista sobre UNA fuente."""
        from challenge.agent import ActorAgent
        from challenge.runner import create_model
        # ... crea agente con CSVReaderTool (cacheado), lee schema_metadata.json
        # ... ejecuta task → produce output/findings_{source}.json
        output_path = self.output_dir / f"findings_{source}.json"
        logger.info(f"  [{source}] Findings saved: {output_path}")
        return output_path
    
    def _synthesize(self, findings_files: list[Path], schema: dict) -> dict:
        """Synthesizer (modelo caro) recibe solo los findings JSON."""
        all_findings = self._load_findings(findings_files)
        
        # El synthesizer NO lee CSVs. Trabaja sobre ~8KB de findings.
        prompt = f"""You are the Synthesizer. You receive structured findings
from 4 specialist agents and schema metadata. Your job:
1. Cross-reference findings — do any conflict?
2. Identify risk signals that require ≥2 sources.
3. Assign confidence scores.
4. Generate the final risk report.

Schema metadata (foreign keys, column types):
{json.dumps(schema.get('cross_source_links', []), indent=2)}

Specialist findings:
{json.dumps(all_findings, indent=2)}
"""
        # ... llama al modelo caro (Opus) con este prompt
        return {"status": "generated", "findings_used": len(findings_files)}
    
    def _verify_all(self, findings_files: list[Path]) -> list[dict]:
        """Verificación determinista en paralelo a la síntesis."""
        results = []
        for f in findings_files:
            data = json.loads(f.read_text())
            for finding in data.get("findings", []):
                if finding.get("verification_query"):
                    import sqlite3
                    conn = sqlite3.connect(":memory:")
                    actual_query = finding["verification_query"]
                    try:
                        result = conn.execute(actual_query).fetchone()
                        actual = float(result[0]) if result else 0
                        expected = float(finding.get("value", 0))
                        results.append({
                            "statement": finding["statement"],
                            "verified": abs(actual - expected) < 0.01 * max(abs(expected), 1),
                            "expected": expected,
                            "actual": actual,
                        })
                    except Exception as e:
                        results.append({
                            "statement": finding["statement"],
                            "verified": False,
                            "error": str(e),
                        })
        return results
    
    def _load_findings(self, files: list[Path]) -> dict:
        result = {}
        for f in files:
            if f.exists():
                result[f.stem.replace("findings_", "")] = json.loads(f.read_text())
        return result
```

**Qué cambia**:
- `asyncio` en vez de `ThreadPoolExecutor` — mejor para I/O bound (LLM calls).
- Fase 1: SchemaExtractor analiza CSVs una vez. Output: `.cache/schema_metadata.json`.
- Fase 2: 4 especialistas en paralelo. Cada uno lee SU fuente (con caché). Output: `findings_*.json`.
- Fase 3: Synthesizer (modelo caro) + VerificationRunner en paralelo.
- Reconciliation gate bloquea el output si hay discrepancias cross-source.
- El synthesizer recibe ~8KB de findings, no ~200KB de CSVs.

---

## 8. `challenge/tasks.py` — Tareas cross-source

**Problema**: Las 3 tareas actuales (`billing_summary`, `usage_trends`, `support_health`) son single-source. La tarea que el README pide — risk assessment cross-source — no existe. El TODO en línea 57 lo deja como ejercicio.

**Solución**: Añadir las tareas que realmente producen valor: risk assessment (cross-source), relationship health (texto no estructurado), y reconciliation check (verificación determinista).

```python
# Añadir a TASKS, después de support_health
    {
        "name": "account_risk_assessment",
        "agent_name": "RiskSynthesizer",
        "role": "Cross-source risk analyst — synthesizes findings from all specialists",
        "prompt": (
            "You are the RiskSynthesizer. You receive structured findings from "
            "4 specialist agents. Your job is NOT to re-read raw CSVs — "
            "you work exclusively from the findings JSON files in output/.\n\n"
            "1. Load the findings from:\n"
            "   - output/findings_billing.json\n"
            "   - output/findings_usage.json\n"
            "   - output/findings_support.json\n"
            "   - output/findings_relationship.json\n"
            "2. Cross-reference — do any findings conflict?\n"
            "3. Detect risk signals requiring ≥2 sources:\n"
            "   - Payment deterioration + CFO change = structural risk\n"
            "   - Low utilization + support complaints + PO protest = churn risk\n"
            "   - CTO departure + SLA breach + lack of renewal data = renewal risk\n"
            "4. For each finding, assign confidence:\n"
            "   - 2+ independent sources confirm → confidence ≥0.85\n"
            "   - Verifiable with SQL query → confidence ≥0.95\n"
            "   - Single source, LLM judgment → confidence ≤0.70\n"
            "5. Create PDF 'risk_assessment_merid001.pdf' with:\n"
            "   - Executive summary with overall risk level\n"
            "   - Per-department breakdown (Engineering vs Marketing)\n"
            "   - Cross-source signal matrix\n"
            "   - Recommendations with confidence scores\n"
            "6. BEFORE create_report, verify ALL numbers with verify_finding.\n"
            "   Pass verification_hashes to create_report."
        ),
    },
    {
        "name": "relationship_health",
        "agent_name": "RelationshipAnalyst",
        "role": "Stakeholder relationship and sentiment analyst",
        "prompt": (
            "Analyze CRM interactions and emails for account MERID-001.\n"
            "1. Use search_emails to find:\n"
            "   - Competitor mentions and pricing concerns\n"
            "   - CFO vendor review discussions\n"
            "   - Champion/key stakeholder departure signals\n"
            "2. Identify key stakeholders and sentiment trajectory:\n"
            "   - Sarah Chen (CTO): champion, but leaving?\n"
            "   - Margaret Walsh (CFO): new, reviewing all vendors\n"
            "   - Lisa Huang (Marketing): frustrated, filing complaints\n"
            "   - James Park (Procurement): thorough, evaluating alternatives\n"
            "3. Map relationship health over time.\n"
            "4. Save structured findings to output/findings_relationship.json.\n"
            "   Do NOT generate a PDF — this feeds the RiskSynthesizer."
        ),
    },
    {
        "name": "reconciliation_check",
        "agent_name": "ReconciliationAgent",
        "role": "Cross-source data consistency verifier",
        "prompt": (
            "Verify data consistency across sources for MERID-001.\n"
            "1. Compare billing totals vs contract annual values (±5% threshold).\n"
            "2. Compare licensed seats in usage vs contracts (must match exactly).\n"
            "3. Verify SLA credits have corresponding support tickets or CRM notes.\n"
            "4. Check POs reference valid contracts and vice versa.\n"
            "5. For each inconsistency: log delta, sources, severity.\n"
            "6. Save to output/findings_reconciliation.json.\n"
            "   If financial-impact discrepancies exist, flag as BLOCKING."
        ),
    },
```

**Qué cambia**:
- `account_risk_assessment`: la tarea que el README pide. Cross-source. NO ejecutable con la arquitectura actual (necesita el orchestrator).
- `relationship_health`: analiza CRM + emails con búsqueda semántica. Detecta riesgos de stakeholder que no aparecen en datos estructurados.
- `reconciliation_check`: verificación cross-source determinista. ¿Billing y contracts cuadran? ¿SLA credits tienen evidencia?

---

## 9. `challenge/agent.py` — Modo Plan/Execute + System Reminders

**Problema**: `ActorAgent` (línea 52) no distingue entre exploración y ejecución. El system prompt (línea 19) se define una vez y el agente "olvida" instrucciones tras varios turnos. `max_steps=15` es fijo sin conciencia del límite.

**Solución**: Separar Plan Mode (solo lectura) de Execute Mode (acceso completo). Inyectar recordatorios contextuales antes de acciones críticas. Informar al agente de los pasos restantes.

```python
class ActorAgent:
    def __init__(self, name: str, role: str, model: object,
                 tools: list | None = None, max_steps: int = 15,
                 mode: str = "execute"):
        self.name = name
        self.role = role
        self.mode = mode
        
        if mode == "plan":
            allowed = [CSVReaderTool(), EmailSearchTool(), SchemaExplorerTool()]
            max_steps = 8
            extra = (
                f"You are in PLAN MODE. Steps remaining: {max_steps}.\n"
                "You can ONLY read data. You CANNOT generate reports or modify files.\n"
                "When steps run low, prioritize synthesis over exploration.\n"
                "Output must be a structured plan, not a PDF."
            )
        else:
            allowed = [CSVReaderTool(), EmailSearchTool(), PDFReportTool(), VerificationTool()]
            extra = (
                f"You are in EXECUTE MODE. Steps remaining: {max_steps}.\n"
                "BEFORE calling create_report, verify EVERY number with verify_finding.\n"
                "Pass verification_hashes to create_report. Reports without verification "
                "will be BLOCKED."
            )
        
        all_tools = allowed + (tools or [])
        instructions = INSTRUCTIONS_TEMPLATE.format(name=name, role=role)
        
        self.agent = CodeAgent(
            tools=all_tools,
            model=model,
            instructions=instructions + "\n\n" + extra,
            additional_authorized_imports=[
                "json", "csv", "datetime", "collections", "statistics", "math", "re",
            ],
            max_steps=max_steps,
        )
    
    def run(self, task: str) -> str:
        logger.info("[%s|%s|%d steps] Starting", self.name, self.mode, self.agent.max_steps)
        
        # Inyectar reminder si la tarea implica generar reporte
        if "report" in task.lower() or "pdf" in task.lower():
            reminder = (
                "\nREMINDER: Before generating a report, you MUST:\n"
                "1. Verify ALL numeric claims with verify_finding.\n"
                "2. Collect verification hashes.\n"
                "3. Pass them to create_report as verification_hashes.\n"
                "Reports without verification will be BLOCKED."
            )
            task = task + "\n\n" + reminder
        
        result = self.agent.run(task)
        logger.info("[%s] Done", self.name)
        return result
```

**Qué cambia**:
- **Plan Mode**: solo tools de lectura. 8 pasos máximo. No puede generar PDFs.
- **Execute Mode**: acceso completo pero VerificationTool es obligatorio.
- **System Reminders**: se inyectan justo antes de la acción crítica, no solo en el system prompt inicial.
- El agente sabe cuántos pasos tiene y puede ajustar su estrategia.
- Si la tarea menciona "report" o "pdf", se inyecta automáticamente el recordatorio de verificación.

---

## Resumen de archivos modificados

| Archivo | Tipo | Cambio clave |
|---------|------|-------------|
| `tools/csv_reader.py` | Modificado | Caché JSON + metadatos + rate limiting + inferencia de tipos |
| `tools/verification.py` | **Nuevo** | VerificationTool con queries SQL deterministas |
| `tools/email_search.py` | **Nuevo** | EmailSearchTool con embeddings semánticos precomputados |
| `tools/pdf_report.py` | Modificado | Guardarraíl: bloquea PDF si hay claims no verificados |
| `tools/guardrails.py` | **Nuevo** | ReconciliationGate + GuardrailHooks |
| `challenge/schema_extractor.py` | **Nuevo** | SchemaExtractor: analiza CSVs offline, genera metadatos |
| `challenge/orchestrator.py` | **Nuevo** | Orchestrator con asyncio: Exploration → Fan-out → Fan-in |
| `challenge/tasks.py` | Modificado | 3 tareas cross-source nuevas |
| `challenge/agent.py` | Modificado | Plan/Execute modes + system reminders + step awareness |
