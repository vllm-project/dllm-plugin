#!/bin/sh
# Called by pre-commit (prepare-commit-msg): $1 is the path to the commit message file.

if [ -z "$1" ]; then
    echo "Error: Commit message file path not provided." >&2
    exit 1
fi

name=$(git config user.name)
email=$(git config user.email)
if [ -z "$name" ] || [ -z "$email" ]; then
    echo "Error: git config user.name and user.email must be non-empty for DCO Signed-off-by." >&2
    echo "Set them once, e.g.: git config --global user.name 'Your Name' && git config --global user.email 'you@example.com'" >&2
    exit 1
fi

git interpret-trailers --if-exists doNothing --trailer "Signed-off-by: ${name} <${email}>" --in-place "$1"
