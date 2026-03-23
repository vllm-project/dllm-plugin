#!/bin/sh
# Called by pre-commit (prepare-commit-msg): $1 is the path to the commit message file.

if [ -z "$1" ]; then
    echo "Error: Commit message file path not provided." >&2
    exit 1
fi

git interpret-trailers --if-exists doNothing --trailer "Signed-off-by: $(git config user.name) <$(git config user.email)>" --in-place "$1"
