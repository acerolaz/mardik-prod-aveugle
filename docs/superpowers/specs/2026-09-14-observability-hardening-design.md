# Fiabilisation de l'observabilité Mardik — Design

- **Date** : 2026-09-14
- **Branche** : `observability-hardening`
- **Périmètre** : option A — rendre fiable et exploitable de bout en bout l'instrumentation
  existante (traces, métriques, logs structurés). Pas de nouveau backend, pas de sémantique
  `gen_ai.*`, pas de coût/tokens (hors périmètre, cf. options B/C).

## 1. Contexte et problèmes constatés

L'instrumentation est câblée (`telemetry.py`, spans `agent.turn` / `llm.invoke` / `tool.call`,
Jaeger en local) mais n'est pas exploitable en production :

| # | Problème | Conséquence |
|---|---|---|
| P1 | Métriques exportées vers la console en prod ; aucune `Resource` ; `settings.service_name` et `settings.otel_endpoint` ignorés | Métriques perdues, spans sans nom de service |
| P2 | `SimpleSpanProcessor` (export synchrone) et aucun flush/shutdown à la sortie du CLI | Latence ajoutée à chaque span, spans perdus à l'arrêt |
| P3 | `errors_total` jamais incrémenté ; timeout sans statut ERROR ; toute exception autre que `TimeoutError` dans le worker LLM se transforme en `KeyError` | Erreurs invisibles et diagnostic trompeur |
| P4 | Logs sans `trace_id` / `span_id` | Impossible de relier un log à sa trace |
| P5 | `agent.turn` sans `session.id` ; `tool.call` sans statut ; outil inconnu → `KeyError` non typé | Traces peu exploitables |
| P6 | `session_id` utilisé comme attribut de métrique | Explosion de cardinalité (une série par session) |

## 2. Décisions

- **D1 — Approche** : enrichir le bundle `Telemetry` injectable existant (pas de providers
  globaux, pas d'auto-instrumentation). Les tests continuent d'injecter des exporteurs en mémoire.
- **D2 — Destination des métriques** : export sélectionné par `OTEL_METRICS_EXPORTER`
  (`console` | `otlp` | `none`, défaut `console`). Jaeger all-in-one n'ingère pas les métriques :
  en local elles restent sur la console ; `otlp` sert en production derrière un collecteur.
  `docker-compose.yml` est inchangé.
- **D3 — Logs** : JSON sur stdout via structlog, enrichis de `trace_id`/`span_id`. Pas d'export
  OTLP des logs (API Logs OTel Python encore expérimentale).
- **D4 — Exceptions LLM inattendues** : propagées avec leur type d'origine ; `TimeoutError`
  reste convertie en `LLMTimeoutError`.
- **D5 — Cardinalité** : `session_id` n'apparaît jamais dans les attributs de métriques ; il
  reste sur les spans et dans les logs.
- **D6 — Données personnelles** : aucun contenu de message ni argument d'outil dans les spans.

## 3. Configuration (`config.py`)

Nouveaux champs de `Settings` :

| Champ | Variable d'environnement | Défaut | Validation |
|---|---|---|---|
| `environment` | `APP_ENV` | `development` | — |
| `metrics_exporter` | `OTEL_METRICS_EXPORTER` | `console` | `console`, `otlp` ou `none` (insensible à la casse) ; sinon `ValueError` levée par `load_settings()` |
| `metrics_export_interval_ms` | `OTEL_METRIC_EXPORT_INTERVAL` | `60000` | entier ; sinon `ValueError` |

`otel_endpoint` et `service_name` (existants) deviennent effectivement utilisés.

## 4. Télémétrie (`telemetry.py`)

### 4.1 Construction

- `build_resource(settings) -> Resource` : attributs `service.name` (`settings.service_name`),
  `service.version` (`mardik.__version__`), `deployment.environment` (`settings.environment`).
- `build_telemetry(span_exporter=None, metric_reader=None, level="INFO", resource=None, batch=False)` :
  signature existante conservée, deux paramètres optionnels ajoutés.
  - `resource` : appliquée au `TracerProvider` et au `MeterProvider` (défaut : `Resource` par défaut du SDK).
  - `batch=True` → `BatchSpanProcessor`, sinon `SimpleSpanProcessor` (tests).
  - Le `Telemetry` retourné conserve une référence aux deux providers.
- `build_default_telemetry(settings) -> Telemetry` (signature modifiée : reçoit `Settings` au lieu de `level`) :
  - traces : `OTLPSpanExporter(endpoint=settings.otel_endpoint)`, `batch=True` ;
  - métriques : `PeriodicExportingMetricReader(exporter, export_interval_millis=settings.metrics_export_interval_ms)`
    avec `OTLPMetricExporter(endpoint=settings.otel_endpoint)` si `otlp`, `ConsoleMetricExporter()` si
    `console` ; aucun reader si `none` ;
  - `resource=build_resource(settings)`, `level=settings.log_level`.
  - La sélection de l'exporteur de métriques est isolée dans une fonction pure
    `build_metric_reader(settings) -> MetricReader | None`, testable sans réseau.
- `OTLPMetricExporter` provient de `opentelemetry-exporter-otlp-proto-grpc`, déjà en dépendance.

### 4.2 Instruments

| Nom | Type | Unité | Attributs | Émis par |
|---|---|---|---|---|
| `latency_ms` | histogramme | `ms` | `outcome` (`ok` \| `error`) | `track_turn` |
| `errors_total` | compteur | — | `error.type` (nom de classe de l'exception) | `record_error` |
| `tool_calls_total` | compteur | — | `tool.name`, `outcome` (`ok` \| `error`) | `Agent._dispatch_tool` |

Aucun instrument ne porte `session_id` (D5).

### 4.3 API de `Telemetry`

- `record_error(error_type: str) -> None` : incrémente `errors_total`.
- `track_turn(session_id: str)` : context manager (classe avec `__enter__`/`__exit__`, sans
  intercepter l'exception — `__exit__` retourne `False`).
  - À l'entrée : démarre le chronomètre.
  - À la sortie, toujours : enregistre `latency_ms` avec `outcome`.
  - Succès : log `turn.completed` (info) avec `session_id`, `latency_ms`.
  - Erreur : `record_error(exc_type.__name__)` puis log `turn.failed` (error) avec
    `session_id`, `latency_ms`, `error.type`.
- `shutdown() -> None` : `force_flush()` puis `shutdown()` du `TracerProvider` et du
  `MeterProvider` ; idempotent (un second appel ne fait rien).
- `record_latency` est supprimé (remplacé par `track_turn`) ; ses usages sont migrés.

### 4.4 Corrélation logs ↔ traces

Processor structlog `add_trace_context(logger, method_name, event_dict)`, inséré dans
`configure_logging` avant le rendu JSON : si un span valide est actif
(`trace.get_current_span().get_span_context().is_valid`), ajoute `trace_id` (32 caractères hex)
et `span_id` (16 caractères hex). Sinon, n'ajoute rien.

### 4.5 `NoOpTelemetry`

Expose la même interface sans effet : `tracer`, `logger`, `latency_ms`, `errors`, `tool_calls`,
`record_error`, `track_turn` (context manager neutre qui ne supprime pas les exceptions),
`shutdown`. Substituable à `Telemetry` partout où l'agent l'utilise.

## 5. Agent (`agent.py`) et erreurs (`errors.py`)

### 5.1 Appel LLM

- `_invoke_llm` remplace `threading.Thread` + `box` par un `concurrent.futures.ThreadPoolExecutor(max_workers=1)`
  utilisé en context manager ; la tâche est soumise via `ctx.run` sur `contextvars.copy_context()`
  pour conserver la propagation de trace.
- `future.result()` relance l'exception d'origine. Seul `TimeoutError` est intercepté et converti
  en `LLMTimeoutError` (`from exc`) ; toute autre exception se propage inchangée.
- Le span `llm.invoke` passe en ERROR avec un événement d'exception grâce au comportement par
  défaut de `start_as_current_span` (`record_exception=True`, `set_status_on_exception=True`) —
  aucun `try/except` ajouté pour les spans.

### 5.2 Appel d'outil

- Nouvelle exception `ToolNotFoundError(MardikError)` dans `errors.py`.
- `_dispatch_tool` : span `tool.call` avec `tool.name` ; outil absent → `ToolNotFoundError`.
  L'`outcome` est déterminé via `try/finally` et un indicateur de succès (pas de `except` large) :
  attribut de span `tool.outcome` et incrément de `tool_calls_total{tool.name, outcome}`.

### 5.3 Tour complet

`run_turn` ouvre le span `agent.turn`, puis `telemetry.track_turn(session_id)` à l'intérieur (les
logs portent ainsi le `trace_id` du tour). Attributs de span : `session.id`,
`mardik.history.length` (nombre de messages envoyés au LLM), `mardik.tool_calls.count`.

### 5.4 Câblage (`app.py`)

- `build_agent` appelle `build_default_telemetry(settings)`.
- `main()` exécute le tour dans un `try/finally` appelant `agent.telemetry.shutdown()`.

## 6. Tests

Tous avec les fixtures en mémoire existantes (`InMemorySpanExporter`, `InMemoryMetricReader`),
sans réseau.

| Fichier | Test | Vérifie |
|---|---|---|
| `tests/unit/test_config.py` | exporteur invalide, intervalle non entier | `ValueError` |
| `tests/unit/test_telemetry.py` | resource | `service.name`, `service.version`, `deployment.environment` présents sur les spans |
| | `build_metric_reader` | `console` → exporteur console ; `otlp` → `OTLPMetricExporter` ; `none` → `None` |
| | shutdown | en mode `batch=True`, les spans sont dans l'exporteur après `shutdown()` ; second appel sans erreur |
| | corrélation | `trace_id` du log `turn.completed` = `trace_id` du span `agent.turn` |
| `tests/unit/test_agent_errors.py` | timeout | spans `llm.invoke` et `agent.turn` en `StatusCode.ERROR` ; `errors_total{error.type=LLMTimeoutError}` = 1 ; log `turn.failed` |
| | exception inattendue | un LLM levant `RuntimeError` → `RuntimeError` propagée |
| | outil inconnu | `ToolNotFoundError` ; span `tool.call` en ERROR |
| `tests/unit/test_agent_observability.py` | compteur outils | `tool_calls_total{tool.name=lookup_order, outcome=ok}` = 1 |
| | cardinalité | aucun point de métrique n'a d'attribut `session_id` |
| `tests/unit/test_wiring.py` | câblage | `build_agent(llm, settings=...)` retourne un agent dont la télémétrie est un `Telemetry` (exporteur de métriques `none` pour éviter tout réseau) |
| `tests/integration/test_replay.py` | incident timeout | en plus de l'exception, `errors_total` incrémenté |

Les tests existants (traçage outil, métrique de latence, log structuré, propagation inter-threads,
rejeux) doivent continuer de passer.

## 7. Documentation

- `.env.example` : `OTEL_METRICS_EXPORTER=console`, `OTEL_METRIC_EXPORT_INTERVAL=60000`.
- `README.md` : section observabilité — signaux émis (spans, instruments, champs de log),
  choix de l'exporteur de métriques, précision « Jaeger ne stocke que les traces ».

## 8. Critères d'acceptation

1. `make lint`, `make typecheck`, `make test` passent.
2. Avec `make up` puis le CLI (identifiants Azure renseignés dans `.env`), la trace `agent.turn` apparaît dans Jaeger sous le service
   `mardik`, y compris quand le processus se termine immédiatement après le tour.
3. Un tour en erreur produit : span ERROR, `errors_total` incrémenté, log `turn.failed` portant le `trace_id`.

## 9. Hors périmètre

Conventions `gen_ai.*`, tokens/coût, exposition du `trace_id` dans les rapports de tests de
rejeu, OTel Collector, Prometheus/Grafana, Langfuse/Phoenix, export OTLP des logs.
