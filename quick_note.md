## update master and remove old remote branches
git fetch origin --prunee

## Delete local merged branches: 
git branch --merged master | grep -v '^\*' | grep -v master | xargs git branch -d

## Reset branches
Resetting to another branch's state: git reset --hard <another_branch>
Resetting to a specific commit hash: git reset --hard <commit_hash>
Resetting to the point of divergence: git reset --hard origin/<base_branch> or <base_branch>