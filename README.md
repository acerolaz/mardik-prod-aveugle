# Mardik Support Agent

Agent conversationnel de support client connecté à un modèle Azure (Kimi-K2.6),
instrumenté avec OpenTelemetry et validé par rejeu de sessions enregistrées.

## Features

- Réponses aux questions de support via un agent qui appelle des outils métier (statut de commande, centre d'aide).
- Mémoire de session : l'historique d'une conversation est conservé et rejoué d'un tour à l'autre.
- Tracing distribué et métriques via OpenTelemetry, exportés vers un collecteur OTLP (Jaeger en local).
- Journalisation structurée (structlog) en JSON.
- Suite de rejeu : des sessions réelles enregistrées sont rejouées de bout en bout.

## Stack

- Python 3.11, géré avec `uv`
- LangChain 0.3 + `langchain-azure-ai` (modèle Kimi-K2.6)
- OpenTelemetry SDK 1.27 (traces + métriques), exporter OTLP/gRPC
- structlog 24 (logs JSON)
- pytest 8 + pytest-asyncio

## Setup

```bash
make install              # uv sync — install dependencies
cp .env.example .env      # then fill in the values
make up                   # docker compose up -d (Jaeger all-in-one)
make test                 # run the test suite
```

Jaeger UI : http://localhost:16686 — les traces y apparaissent une fois l'application instrumentée.

## Observabilité

Tout passe par le bundle injectable `Telemetry` (`src/mardik/telemetry.py`) :

| Signal | Contenu | Où le voir |
|---|---|---|
| Traces | `agent.turn` (`session.id`) > `llm.invoke` (`gen_ai.usage.input_tokens` / `output_tokens` si le client les fournit), `tool.call` (`tool.name`, `tool.input`, `tool.output` tronqués à 512 caractères, `tool.outcome`) ; statut ERROR si exception | Jaeger http://localhost:16686, service `mardik` |
| Métriques | `latency_ms{outcome}`, `errors_total{error.type}`, `tool_calls_total{tool.name, outcome}` | stdout (`OTEL_METRICS_EXPORTER=console`) ou collecteur OTLP (`otlp`) |
| Logs | JSON : `turn.completed` / `turn.failed` avec `session_id`, `latency_ms`, `trace_id`, `span_id` | stdout |

- Relier un log à sa trace : copier son `trace_id` dans la recherche Jaeger ; inversement `grep <trace_id>` dans les logs.
- `session_id` n'est jamais un attribut de métrique (une série par session sinon) : il est sur les spans et dans les logs.
- Le CLI appelle `telemetry.shutdown()` en sortie pour vider les spans exportés par lots.
- Un appel LLM qui dépasse `MARDIK_LLM_TIMEOUT_S` (30 s par défaut) fait échouer le tour avec `LLMTimeoutError`, sans rien écrire dans la session.
- Instrumenter un nouveau traitement : `with telemetry.tracer.start_as_current_span("nom")`, et `with telemetry.track_turn(session_id)` pour un tour complet.

## Layout

```
src/mardik/
  agent.py        Orchestration d'un tour : LLM, appels d'outils, télémétrie
  app.py          Assemblage de l'agent + point d'entrée CLI
  config.py       Configuration lue depuis l'environnement
  errors.py       Exceptions de domaine
  llm.py          Fabrique du client Azure (Kimi-K2.6)
  runner.py       Rejeu de sessions enregistrées
  session.py      Stockage des sessions partagé entre tours concurrents
  telemetry.py    Tracer, logger et instruments de métriques
sessions/         Sessions enregistrées utilisées par les tests de rejeu
tests/            Tests unitaires + tests d'intégration (rejeu)
docker-compose.yml  Jaeger all-in-one pour la collecte locale des traces
```

## Useful commands

```bash
make fmt        # ruff format + autofix
make lint       # ruff check
make typecheck  # mypy
make check      # format + lint + mypy + tests (mêmes étapes que la CI)
make down       # stop docker services
```

## Known issues

La tenue sous charge est couverte par les tests : `tests/integration/test_observability.py` (24 tours simultanés sur des sessions distinctes) et `tests/integration/test_replay_concurrent_session.py` (24 tours simultanés sur une même session). Limites restantes :

- `llm.py` renvoie le modèle LangChain brut, qui produit un `AIMessage` et non le `Reply` attendu par `Agent` : un adaptateur (contenu, `tool_calls`, `usage_metadata` → `usage`) reste à écrire avant un usage en production.
- Aucune évaluation de la qualité des réponses (LLM-as-judge, groundedness) : les tests vérifient le comportement et les signaux, pas la pertinence sémantique.

## License

MIT

## Contact

es.agwu.19@eigsi.fr
