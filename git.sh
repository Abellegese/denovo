#!/bin/bash

git add .

if [ -z "$1" ]; then
    commit_message="updates"
else
    commit_message="$1"
fi

git commit -m "$commit_message"

git push -u origin master