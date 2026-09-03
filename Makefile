ROLE ?= operator
USER ?= local-user

.PHONY: install test eval eval-update-baseline serve chat pending

install:
	pip install -e ".[dev]"

test:
	pytest -q

eval:
	python -m eval.run_eval

eval-update-baseline:
	python -m eval.run_eval --update-baseline

retrieval-benchmark:
	python -m eval.retrieval_benchmark

observability-report:
	python -m copilot.observability.report

serve:
	uvicorn api.server:app --reload

chat:
	python -m copilot.cli chat --query "$(QUERY)" --role "$(ROLE)" --user "$(USER)"

pending:
	python -m copilot.cli pending
