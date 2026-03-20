from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app.schemas.api import FinalResponse, QueryRequest
from app.schemas.competency import CompetencyConfig, CompetencySummary
from app.schemas.ingestion import IngestionResult
from app.schemas.retrieval import HybridRetrievalTrace
from app.schemas.routing import RoutingTrace

router = APIRouter()

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Engineering Agent System</title>
  <style>
    :root {
      --bg: #f4f1e8;
      --panel: #fffdf7;
      --line: #d9d1bf;
      --text: #1c242b;
      --muted: #5c6770;
      --accent: #0b6e4f;
      --accent-strong: #084c38;
      --danger: #a63d40;
      --code-bg: #f0ebde;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(11, 110, 79, 0.10), transparent 34%),
        linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
      color: var(--text);
    }
    .page { max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }
    .hero { margin-bottom: 24px; }
    .hero h1 { margin: 0 0 10px; font-size: 32px; line-height: 1.1; }
    .hero p { margin: 0; color: var(--muted); max-width: 760px; }
    .grid {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(420px, 1fr);
      gap: 18px;
      align-items: start;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 12px 32px rgba(28, 36, 43, 0.06);
    }
    .card h2 { margin: 0 0 14px; font-size: 18px; }
    label { display: block; margin: 0 0 8px; font-weight: 600; }
    .field { margin-bottom: 16px; }
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      color: var(--text);
      background: #fff;
      resize: vertical;
      min-height: 120px;
    }
    input[type="text"], input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }
    .calc-input-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .calc-input-item {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .calc-input-item span {
      font-size: 13px;
      color: var(--muted);
    }
    .actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
    }
    button:hover { background: var(--accent-strong); }
    button.secondary { background: #dfe8e3; color: var(--text); }
    .status { min-height: 24px; color: var(--muted); font-size: 14px; }
    .status.error { color: var(--danger); }
    .response-section { margin-bottom: 18px; }
    .response-section h3 {
      margin: 0 0 8px;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    .answer-box, pre {
      margin: 0;
      background: var(--code-bg);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      line-height: 1.5;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      background: #fcfaf4;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }
    .metric-title { color: var(--muted); font-size: 13px; margin-bottom: 6px; }
    .metric-value { font-weight: 700; }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Engineering Agent System</h1>
      <p>Simple browser UI for sending engineering requests to <code>POST /query</code>.</p>
    </div>

    <div class="grid">
      <section class="card">
        <h2>Request</h2>
        <form id="query-form">
          <div class="field">
            <label for="query">Query</label>
            <textarea id="query" name="query">Подобрать метод для контроля пескопроявления</textarea>
          </div>

          <div class="field">
            <label for="context">Context JSON</label>
            <textarea id="context" name="context">{
  "well_type": "horizontal",
  "production_rate": 100,
  "reservoir_strength": "weak",
  "completion_constraints": "limited workover window"
}</textarea>
          </div>

          <div id="calc-inputs-section" class="field" style="display:none;">
            <label for="calc-inputs-grid">Calculation Inputs</label>
            <div id="calc-inputs-grid" class="calc-input-grid"></div>
          </div>

          <div class="actions">
            <button id="submit-btn" type="submit">Send</button>
            <button id="reset-btn" class="secondary" type="button">Example</button>
          </div>
        </form>

        <div id="status" class="status"></div>
      </section>

      <section class="card">
        <h2>Response</h2>
        <div class="summary">
          <div class="metric">
            <div class="metric-title">Domain</div>
            <div id="domain" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-title">Competency</div>
            <div id="competency" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-title">Intent</div>
            <div id="intent" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-title">Confidence</div>
            <div id="confidence" class="metric-value">-</div>
          </div>
        </div>

        <div class="response-section">
          <h3>Answer</h3>
          <div id="answer" class="answer-box">The engineering answer will appear here.</div>
        </div>

        <div class="response-section">
          <h3>Raw JSON</h3>
          <pre id="raw-json">{}</pre>
        </div>
      </section>
    </div>
  </div>

  <script>
    document.addEventListener("DOMContentLoaded", function () {
      const form = document.getElementById("query-form");
      const queryField = document.getElementById("query");
      const contextField = document.getElementById("context");
      const submitButton = document.getElementById("submit-btn");
      const resetButton = document.getElementById("reset-btn");
      const calcInputsSection = document.getElementById("calc-inputs-section");
      const calcInputsGrid = document.getElementById("calc-inputs-grid");
      const statusBox = document.getElementById("status");
      const domainBox = document.getElementById("domain");
      const competencyBox = document.getElementById("competency");
      const intentBox = document.getElementById("intent");
      const confidenceBox = document.getElementById("confidence");
      const answerBox = document.getElementById("answer");
      const rawJsonBox = document.getElementById("raw-json");

      const exampleQuery = "Подобрать метод для контроля пескопроявления";
      const exampleContext = {
        well_type: "horizontal",
        production_rate: 100,
        reservoir_strength: "weak",
        completion_constraints: "limited workover window"
      };

      function setStatus(message, isError) {
        statusBox.textContent = message;
        statusBox.className = isError ? "status error" : "status";
      }

      function renderAnswer(data) {
        const answerParts = [];
        if (data.answer && data.answer.recommendation) {
          answerParts.push("Recommendation:\\n" + data.answer.recommendation);
        }
        if (data.answer && Array.isArray(data.answer.justification) && data.answer.justification.length) {
          answerParts.push("Justification:\\n- " + data.answer.justification.join("\\n- "));
        }
        if (data.answer && Array.isArray(data.answer.limitations) && data.answer.limitations.length) {
          answerParts.push("Limitations:\\n- " + data.answer.limitations.join("\\n- "));
        }
        if (data.answer && Array.isArray(data.answer.missing_inputs) && data.answer.missing_inputs.length) {
          answerParts.push("Missing inputs:\\n- " + data.answer.missing_inputs.join("\\n- "));
        }
        answerBox.textContent = answerParts.join("\\n\\n") || "Empty response.";
      }

      function humanizeFieldName(name) {
        return name
          .replace(/_/g, " ")
          .replace(/\\b\\w/g, function (char) { return char.toUpperCase(); });
      }

      function parseContextSafe() {
        return contextField.value.trim() ? JSON.parse(contextField.value) : {};
      }

      function mergeCalculationInputs(parsedContext) {
        const merged = { ...parsedContext };
        const inputNodes = calcInputsGrid.querySelectorAll("input[data-field]");
        inputNodes.forEach(function (node) {
          const rawValue = node.value.trim();
          if (!rawValue) {
            return;
          }
          const normalized = rawValue.replace(",", ".");
          if (/^-?\\d+(\\.\\d+)?$/.test(normalized)) {
            merged[node.dataset.field] = Number(normalized);
          } else {
            merged[node.dataset.field] = rawValue;
          }
        });
        contextField.value = JSON.stringify(merged, null, 2);
        return merged;
      }

      function renderCalculationInputs(missingInputs, contextValues) {
        calcInputsGrid.innerHTML = "";
        if (!Array.isArray(missingInputs) || !missingInputs.length) {
          calcInputsSection.style.display = "none";
          return;
        }

        missingInputs.forEach(function (fieldName) {
          const wrapper = document.createElement("label");
          wrapper.className = "calc-input-item";

          const title = document.createElement("span");
          title.textContent = humanizeFieldName(fieldName);

          const input = document.createElement("input");
          input.type = "text";
          input.dataset.field = fieldName;
          input.placeholder = fieldName;
          if (contextValues && contextValues[fieldName] !== undefined && contextValues[fieldName] !== null) {
            input.value = String(contextValues[fieldName]);
          }

          wrapper.appendChild(title);
          wrapper.appendChild(input);
          calcInputsGrid.appendChild(wrapper);
        });

        calcInputsSection.style.display = "block";
      }

      function resetForm() {
        queryField.value = exampleQuery;
        contextField.value = JSON.stringify(exampleContext, null, 2);
        renderCalculationInputs([], exampleContext);
        setStatus("Example request loaded.", false);
      }

      async function submitQuery() {
        setStatus("Sending request...", false);
        submitButton.disabled = true;
        answerBox.textContent = "Waiting for API response...";
        rawJsonBox.textContent = "{}";

        let parsedContext = {};
        try {
          parsedContext = parseContextSafe();
          parsedContext = mergeCalculationInputs(parsedContext);
        } catch (error) {
          setStatus("Context JSON is invalid.", true);
          answerBox.textContent = "Fix JSON in the context field and try again.";
          submitButton.disabled = false;
          return;
        }

        const payload = {
          query: queryField.value.trim(),
          context: parsedContext,
          response_mode: "user"
        };

        try {
          const response = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            cache: "no-store",
            body: JSON.stringify(payload)
          });

          const rawText = await response.text();
          let data = {};
          try {
            data = rawText ? JSON.parse(rawText) : {};
          } catch (parseError) {
            throw new Error("API returned non-JSON response: " + rawText.slice(0, 200));
          }

          if (!response.ok) {
            throw new Error(data.detail || ("HTTP " + response.status));
          }

          domainBox.textContent = data.domain || "-";
          competencyBox.textContent = data.competency_id || "-";
          intentBox.textContent = data.intent || "-";
          confidenceBox.textContent =
            data.confidence !== undefined && data.confidence !== null ? data.confidence : "-";
          renderAnswer(data);
          renderCalculationInputs(
            data.answer && Array.isArray(data.answer.missing_inputs) ? data.answer.missing_inputs : [],
            parsedContext
          );
          rawJsonBox.textContent = JSON.stringify(data, null, 2);
          setStatus("Response received.", false);
        } catch (error) {
          console.error(error);
          answerBox.textContent = "Failed to get response from API.";
          rawJsonBox.textContent = String(error);
          setStatus(error.message || "Request failed.", true);
        } finally {
          submitButton.disabled = false;
        }
      }

      form.addEventListener("submit", function (event) {
        event.preventDefault();
        submitQuery();
      });

      resetButton.addEventListener("click", resetForm);
      resetForm();
    });
  </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Minimal browser UI for interacting with POST /query."""

    return HTMLResponse(
        content=UI_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    service = request.app.state.query_service
    return {
        "status": "ok",
        "competencies_loaded": len(service.list_competencies()),
        "competencies_root": str(service.settings.competencies_root),
        "vector_store_path": str(service.settings.vector_store_path),
    }


@router.get("/competencies", response_model=list[CompetencySummary])
def list_competencies(request: Request) -> list[CompetencySummary]:
    service = request.app.state.query_service
    service.reload_registry()
    return service.list_competencies()


@router.get("/competencies/{competency_id}", response_model=CompetencyConfig)
def get_competency(request: Request, competency_id: str) -> CompetencyConfig:
    service = request.app.state.query_service
    try:
        return service.get_competency(competency_id)
    except KeyError as exc:
        service.reload_registry()
        try:
            return service.get_competency(competency_id)
        except KeyError as retry_exc:
            raise HTTPException(status_code=404, detail=f"Competency `{competency_id}` not found.") from retry_exc


@router.post("/query", response_model=FinalResponse)
def query(request: Request, payload: QueryRequest) -> FinalResponse:
    service = request.app.state.query_service
    try:
        return service.handle_query(payload)
    except KeyError as exc:
        service.reload_registry()
        try:
            return service.handle_query(payload)
        except KeyError as retry_exc:
            raise HTTPException(status_code=404, detail=str(retry_exc)) from retry_exc
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/registry/reload")
def reload_registry(request: Request) -> dict[str, object]:
    """Force reload of competency registry from disk."""

    service = request.app.state.query_service
    service.reload_registry()
    competencies = service.list_competencies()
    return {
        "status": "ok",
        "competencies_loaded": len(competencies),
        "competency_ids": [item.id for item in competencies],
    }


@router.post("/ingest/competency/{competency_id}", response_model=IngestionResult)
def ingest_competency(request: Request, competency_id: str, rebuild: bool = False) -> IngestionResult:
    service = request.app.state.query_service
    try:
        return service.ingest_competency(competency_id, rebuild=rebuild)
    except KeyError as exc:
        service.reload_registry()
        try:
            return service.ingest_competency(competency_id, rebuild=rebuild)
        except KeyError as retry_exc:
            raise HTTPException(status_code=404, detail=f"Competency `{competency_id}` not found.") from retry_exc


@router.post("/ingest/all", response_model=list[IngestionResult])
def ingest_all(request: Request, rebuild: bool = False) -> list[IngestionResult]:
    return request.app.state.query_service.ingest_all(rebuild=rebuild)


@router.get("/routing/debug", response_model=RoutingTrace)
def routing_debug(
    request: Request,
    query: str = Query(...),
    context_json: str | None = Query(default=None),
) -> RoutingTrace:
    service = request.app.state.query_service
    try:
        context = service.parse_context_json(context_json)
        return service.route_debug(query, context)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/retrieval/debug", response_model=HybridRetrievalTrace)
def retrieval_debug(
    request: Request,
    query: str = Query(...),
    competency_id: str | None = Query(default=None),
    context_json: str | None = Query(default=None),
) -> HybridRetrievalTrace:
    service = request.app.state.query_service
    try:
        context = service.parse_context_json(context_json)
        return service.retrieval_debug(query, competency_id=competency_id, context=context)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
