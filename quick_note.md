## update master and remove old remote branches
git fetch origin --prune

## Delete local merged branches: 
git branch --merged master | grep -v '^\*' | grep -v master | xargs git branch -d