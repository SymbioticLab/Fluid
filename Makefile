# lists all available targets
list:
	@sh -c "$(MAKE) -p no_targets__ | \
		awk -F':' '/^[a-zA-Z0-9][^\$$#\/\\t=]*:([^=]|$$)/ {\
			split(\$$1,A,/ /);for(i in A)print A[i]\
		}' | grep -v '__\$$' | grep -v 'make\[1\]' | grep -v 'Makefile' | sort"
# required for list
no_targets__:

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	@poetry run isort fluid/ tests/ workloads/
	@poetry run black fluid/ tests/ workloads/

# install all dependencies
setup:
	@poetry install

setup-no-dev:
	@poetry install --no-dev

# test your application (tests in the tests/ directory)
test:
	@poetry run pytest

format-lint:
	@poetry run isort --diff --check fluid/ tests/ workloads/
	@poetry run black --diff --color --check fluid/ tests/ workloads/

# stop the build if there are Python syntax errors or undefined names
# then exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
lint: format-lint
	@poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@poetry run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

build:
	@poetry build

publish: build
	@poetry publish
