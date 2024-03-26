#!/bin/bash

# Add all changes
git add .

# Check if a commit message argument is provided
if [ -z "$1" ]; then
    # If no commit message is provided, use a default message
    commit_message="updates"
else
    # If a commit message is provided, use it
    commit_message="$1"
fi

# Commit changes with the specified message
git commit -m "$commit_message"

# Merge 'main' branch into 'master'
git merge main

# Push changes to the 'master' branch on the remote repository
git push -u origin master
