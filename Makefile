CODE = src tests

.PHONY: lint pretty test lock

lint:
	ruff check $(CODE)
	ruff format --check $(CODE)
	mypy $(CODE)

pretty:
	ruff check --fix $(CODE)
	ruff format $(CODE)

test:
	pytest -q tests

lock:
	rm -f poetry.lock
	poetry lock
