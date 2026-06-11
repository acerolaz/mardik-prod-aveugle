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
make down       # stop docker services
```

## Known issues

L'observabilité est câblée mais n'a pas encore été validée de bout en bout sous charge.

## License

MIT

## Contact

es.agwu.19@eigsi.fr
