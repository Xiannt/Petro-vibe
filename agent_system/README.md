# Engineering Agent System

Production-minded MVP для инженерной базы знаний и расчетных модулей. Архитектура построена вокруг контролируемого pipeline `registry -> hybrid router -> hybrid retrieval -> tools -> composer -> verifier`, без autonomous agent loop и без произвольного доступа LLM к файловой системе.

## Architecture

- `registry`: загружает competency configs из `Competetions/`.
- `routing`: heuristic shortlist + LLM rerank + confidence-gated fallback.
- `llm`: OpenAI-compatible abstraction для structured routing decisions.
- `ingestion`: PDF parsing, chunking, metadata enrichment, embeddings, vector indexing.
- `embeddings`: provider abstraction с cache.
- `vector_store`: локальный SQLite-backed vector store с metadata filtering.
- `retrieval`: metadata ranking + vector search + rerank + context bundle.
- `tools`: controlled calculations через legacy `tool.yaml + main.py + JSON I/O` и через масштабируемый формат `Calculations/yaml/*.yaml + script_file + callable_name`.
- `composer`: сборка прозрачного ответа с sources и chunks.
- `verifier`: структурная и evidence-level валидация ответа.

## Project Tree

```text
agent_system/
  app/
    api/routes.py
    composer/answer_composer.py
    core/settings.py
    embeddings/
      base.py
      service.py
      providers/
        hash_provider.py
        openai_provider.py
    ingestion/
      pdf_parser.py
      chunker.py
      metadata_enricher.py
      vector_indexer.py
      pipeline.py
    llm/
      base.py
      router_client.py
      providers/openai_provider.py
    orchestrator/query_service.py
    registry/
      competency_registry.py
      registry_loader.py
    retrieval/
      document_catalog.py
      metadata_retriever.py
      vector_retriever.py
      hybrid_retriever.py
      context_builder.py
    routing/
      heuristic_router.py
      llm_router.py
      hybrid_router.py
    schemas/
      api.py
      competency.py
      document.py
      ingestion.py
      retrieval.py
      routing.py
    scripts/
      build_registry.py
      ingest_competency.py
      ingest_all.py
      rebuild_vector_index.py
    tools/
      calculation_runner.py
      executor.py
    vector_store/local_store.py
    verifier/answer_verifier.py
    main.py
  data/examples/
    llm_router_prompt.txt
    retrieval_trace.json
    sample_query.json
    sample_response.json
  tests/
    conftest.py
    test_api.py
    test_ingestion.py
    test_registry.py
    test_router.py
  .env
  .env.example
  requirements.txt
  run_server.bat
```

## Environment

Минимальный `.env`:

```env
COMPETENCIES_ROOT=C:\Users\ShilovDV\Desktop\Agent system\Competetions
VECTOR_STORE_PATH=C:\Users\ShilovDV\Desktop\Agent system\agent_system\data\index\vector_store.sqlite3
EMBEDDINGS_CACHE_PATH=C:\Users\ShilovDV\Desktop\Agent system\agent_system\data\index\embeddings_cache.sqlite3
EMBEDDING_PROVIDER=hash
EMBEDDING_MODEL=hash-embedding-v1
EMBEDDING_DIMENSION=128
PDF_PARSER_BACKEND=pypdf
PDF_PARSER_FALLBACK_BACKEND=raw
LLM_ROUTING_ENABLED=false
LLM_PROVIDER=disabled
LLM_BASE_URL=
LLM_API_KEY=
LLM_ROUTER_MODEL=gpt-4.1-mini
```

Для OpenAI-compatible routing:

```env
LLM_ROUTING_ENABLED=true
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_ROUTER_MODEL=gpt-4.1-mini
```

## Install And Run

```powershell
cd "C:\Users\ShilovDV\Desktop\Agent system\agent_system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Или через `run_server.bat`.

После запуска:

- Browser UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Registry And Ingestion Commands

Собрать registry:

```powershell
python -m app.scripts.build_registry
```

Проиндексировать одну компетенцию:

```powershell
python -m app.scripts.ingest_competency PT_2.6 --rebuild
```

Проиндексировать все:

```powershell
python -m app.scripts.ingest_all --rebuild
```

Полный rebuild vector index:

```powershell
python -m app.scripts.rebuild_vector_index --rebuild
```

## API

### Query

```powershell
$body = @{
  query = "Подобрать метод для контроля пескопроявления"
  context = @{
    well_type = "horizontal"
    production_rate = 100
    reservoir_strength = "weak"
    completion_constraints = "limited workover window"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body $body
```

### Sequential Calculation Flow

Расчетные сценарии работают как последовательный двухшаговый диалог.

1. Пользователь пишет запрос вида `рассчитай запасы месторождения`.
2. `QueryPreprocessor` и router помечают его как `calculation` и выбирают нужную competency.
3. `ToolExecutor` находит подходящий calculation manifest внутри `Calculations/`.
4. Если обязательных входных данных нет, tool не запускается и API возвращает `missing_inputs`.
5. UI показывает отдельные поля для недостающих параметров; при повторной отправке они автоматически попадают в `context`.
6. Когда все обязательные входы заполнены, `CalculationRunner` вызывает Python-скрипт и возвращает структурированный результат в ответ пользователю.

Пример первого запроса:

```powershell
$body = @{
  query = "рассчитай запасы месторождения"
  context = @{}
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body $body
```

Ожидаемое поведение:

- выбирается компетенция `RE_5.1`
- intent становится `calculation`
- в `answer.missing_inputs` возвращаются поля, которые надо заполнить

Пример второго запроса после заполнения входов:

```powershell
$body = @{
  query = "рассчитай запасы месторождения"
  context = @{
    area_km2 = 12.5
    net_pay_m = 18
    ntg = 0.82
    porosity = 0.19
    oil_saturation = 0.76
    oil_density_t_m3 = 0.84
    formation_volume_factor = 1.18
    recovery_factor = 0.33
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body $body
```

После этого пользователь получает уже не `missing_inputs`, а итог расчета, например:

- геологические запасы
- извлекаемые запасы
- запасы в млн т
- классификацию месторождения

### Ingestion

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/ingest/competency/PT_2.6?rebuild=true -Method Post
Invoke-RestMethod -Uri http://127.0.0.1:8000/ingest/all?rebuild=true -Method Post
```

### Debug Endpoints

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/routing/debug?query=Подобрать метод для контроля пескопроявления"
Invoke-RestMethod "http://127.0.0.1:8000/retrieval/debug?query=select sand control method&competency_id=PT_2.6"
```

## How Hybrid Router Works

1. `HeuristicRouter` строит shortlist компетенций по keywords, title, supported_tasks и YAML metadata.
2. `LLMRouter` получает только shortlist, query и context, затем возвращает structured JSON decision.
3. `HybridRouter` принимает LLM result только если:
   - провайдер доступен
   - JSON валиден
   - выбранная competency есть в shortlist
   - confidence выше порога
4. Иначе используется deterministic fallback с trace и причиной fallback.

Пример prompt: [llm_router_prompt.txt](C:\Users\ShilovDV\Desktop\Agent system\agent_system\data\examples\llm_router_prompt.txt)

## How Hybrid Retrieval Works

1. `DocumentCatalog` читает YAML metadata внутри выбранной компетенции.
2. `MetadataRetriever` ранжирует документы на document-level.
3. `VectorRetriever` ищет chunks в локальном vector store, фильтруя по competency/document scope.
4. `HybridRetriever` делает rerank chunks по vector similarity, lexical overlap и document priority.
5. `ContextBuilder` собирает context bundle для composer.

Пример trace: [retrieval_trace.json](C:\Users\ShilovDV\Desktop\Agent system\agent_system\data\examples\retrieval_trace.json)

## How Controlled Calculations Work

Слой calculations не зависит от PDF ingestion и не подменяет retrieval. Он используется только тогда, когда routing решил, что запрос требует расчета.

Поддерживаются два совместимых формата инструмента:

1. Legacy format:
   - `Calculations/<tool_name>/tool.yaml`
   - `Calculations/<tool_name>/main.py`
   - запуск через subprocess с `--input` и `--output`
2. Script manifest format:
   - `Calculations/<script>.py`
   - `Calculations/yaml/<manifest>.yaml`
   - запуск через импорт Python-модуля и вызов `callable_name` или `run(inputs)`

Для script manifest format минимально нужны поля:

```yaml
id: RE_VOL_CALC_01
title: Volumetric Oil Reserves Calculation
script_file: volumetric.py
callable_name: run
keywords:
  - рассчитать запасы
required_inputs:
  - area_km2
  - net_pay_m
```

Рекомендуемый контракт Python-скрипта:

```python
def run(inputs: dict[str, object]) -> dict[str, object]:
    return {
        "status": "success",
        "summary": "Расчет выполнен.",
        "recommendation": "Краткий вывод для пользователя.",
        "outputs": {"value": 123.4},
        "missing_inputs": [],
        "assumptions": [],
        "limitations": [],
    }
```

Если обязательных входов не хватает, скрипт или executor должны вернуть:

```python
{
    "status": "skipped",
    "summary": "Расчет не выполнен: не хватает исходных данных.",
    "missing_inputs": ["area_km2", "net_pay_m"],
    "outputs": {},
}
```

Этот формат масштабируется на новые competency без изменений в orchestrator:

- добавляется новый скрипт в `Calculations/`
- рядом добавляется YAML manifest в `Calculations/yaml/`
- router выбирает competency
- executor сам находит инструмент, спрашивает `missing_inputs` и затем вызывает скрипт

## Add A New Competency

1. Создай `Competetions/<DOMAIN>/<COMPETENCY_FOLDER>/`.
2. Добавь `config.yaml` по текущему стандарту.
3. Положи manuals в `Manuals/`, YAML metadata в `Manuals/yaml/`.
4. Если нужен расчет, используй один из двух форматов:
   - legacy: `Calculations/<tool_name>/tool.yaml` и `Calculations/<tool_name>/main.py`
   - рекомендуемый: `Calculations/<script>.py` и `Calculations/yaml/<manifest>.yaml`
5. Выполни ingestion для новой компетенции.
6. Проверь `routing/debug` и `retrieval/debug`, что shortlist и evidence выглядят ожидаемо.

Для новых расчетных скриптов рекомендуемый путь такой:

1. В `config.yaml` competency добавь русские и английские `keywords`, `supported_tasks` и `calculation_triggers`.
2. В manifest опиши `keywords`, `tasks`, `required_inputs`, `optional_inputs`, `script_file`, `callable_name`.
3. В Python-скрипте реализуй `run(inputs)` с возвратом structured dict.
4. Если пользовательский запрос содержит только intent на расчет, система сначала вернет `missing_inputs`, а после заполнения входов выполнит этот же скрипт без дополнительной настройки backend.

## Tests

```powershell
python -m pytest -q
```

Покрытие в текущем этапе:

- heuristic shortlist
- LLM response validation
- hybrid fallback
- PDF parsing
- chunking
- vector retrieval
- metadata filtering
- end-to-end API flow

## Current Limitations

- Default embedding provider `hash` предназначен для offline MVP и должен быть заменен на production embeddings provider.
- Raw PDF fallback извлекает только простые literal strings и не заменяет полноценный parser.
- LLM router пока используется только для rerank shortlist, а не для open-world discovery.
- Нет отдельного citation generator, который вставляет inline references прямо в текст answer.
