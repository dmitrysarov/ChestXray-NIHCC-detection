#!/bin/bash

: '
Usage:
`./.lint/lint.sh <command> <path-to-code-directory>`,
example `./.lint/lint.sh format .`

Default <path-to-code-directory> equals to CWD, so you may execute this script from project root dir:
`./.lint/lint.sh format`

Commands:

install - install all required dependecies via pip install

check - check all *.py files by all intsalled linters according to setup.cfg/pyproject.toml

format - format all code files via isort and black

check-black - black linter
check-isort - import sorting check
check-flake8 - flake8 check
check-mypy - mypy typings check

diff-isort - show difference of import sorting
diff-black - show difference of black linters
'

SCRIPT_NAME=$0
COMMAND=$1
FILEPATH=${2:-.}

function handle_exit {
    EXIT_CODE=$1
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo $2
    fi
    exit $EXIT_CODE
}

case ${COMMAND} in
    check-black)
        black --config ./pyproject.toml --check ${FILEPATH}
        handle_exit $? "Formatting error! Run \`$SCRIPT_NAME format\` to format the code"
        ;;

    check-isort)
        isort --profile black --settings ./pyproject.toml ${FILEPATH} --check-only
        handle_exit $? "Isort error! Run \`$SCRIPT_NAME format\` to format the code"
        ;;

    check-flake8)
        flake8 --config ./setup.cfg ${FILEPATH}
        handle_exit $? "Flake8 error!"
        ;;

    check-mypy)
        mypy --config ./setup.cfg ${FILEPATH}
        handle_exit $? "Mypy error!"
        ;;

    check)
        set -e
        $SCRIPT_NAME check-black
        $SCRIPT_NAME check-isort
        $SCRIPT_NAME check-flake8
        # $SCRIPT_NAME check-mypy
        ;;

    diff-isort)
        isort --profile black --settings ./pyproject.toml ${FILEPATH} --diff
        ;;

    diff-black)
        black --config ./pyproject.toml --diff ${FILEPATH}
        ;;

    format)
        isort --profile black --settings ./pyproject.toml ${FILEPATH}
        black --config ./pyproject.toml ${FILEPATH}
        ;;

    install)
        pip install -r ./requirements.dev.txt
        pre-commit install
        ;;
    *)
        echo $"Usage: $SCRIPT_NAME {check|check-black|check-isort|check-flake8|check-mypy|diff-isort|diff-black|format|install}"
        exit 1
esac
