# Removing Personal Identifiable Information (PII) from Git History

## Issue
This repository contains personal identifiable information (PII) in the git commit history, specifically in commit author names and email addresses.

## Why PII Cannot Be Completely Removed Automatically

Git commit history is immutable by design. The PII exists in:
- Commit author information
- Commit metadata
- Git objects stored in `.git/objects/`

Removing PII from git history requires:
1. **Rewriting the entire repository history** using tools like:
   - `git filter-repo` (recommended)
   - `git filter-branch` (deprecated)
2. **Force pushing** the rewritten history to GitHub
3. **All collaborators must delete and re-clone** the repository
4. **All forks become incompatible** with the rewritten history
5. **Existing references** in issues, PRs, and external tools may break

## What This PR Does

Since automated agents cannot perform force pushes, this PR documents the issue and provides instructions for repository maintainers who have the authority to rewrite history.

## For Repository Maintainers: How to Remove PII

### Option 1: Using git-filter-repo (Recommended)

```bash
# Install git-filter-repo
# See: https://github.com/newren/git-filter-repo

# Create a mailmap file (not committed)
cat > /tmp/mailmap << 'EOF'
Anonymous Contributor <noreply@github.com> <mihirwaknis282@gmail.com>
Anonymous Contributor <noreply@github.com> Waknis <mihirwaknis282@gmail.com>
EOF

# Filter the repository
git filter-repo --mailmap /tmp/mailmap --force

# Force push (WARNING: This rewrites history!)
git push --force --all
git push --force --tags
```

### Option 2: Using git filter-branch (Deprecated)

```bash
git filter-branch --env-filter '
if [ "$GIT_AUTHOR_EMAIL" = "mihirwaknis282@gmail.com" ]; then
    export GIT_AUTHOR_NAME="Anonymous Contributor"
    export GIT_AUTHOR_EMAIL="noreply@github.com"
fi
if [ "$GIT_COMMITTER_EMAIL" = "mihirwaknis282@gmail.com" ]; then
    export GIT_COMMITTER_NAME="Anonymous Contributor"
    export GIT_COMMITTER_EMAIL="noreply@github.com"
fi
' --tag-name-filter cat -- --branches --tags

# Force push (WARNING: This rewrites history!)
git push --force --all
git push --force --tags
```

## After Rewriting History

1. **Notify all contributors** to delete their local clones
2. **Have contributors re-clone** the repository
3. **Update any CI/CD systems** that reference old commit SHAs
4. **Contact GitHub support** if you need to remove the old commits from GitHub's cache

## Alternative: Use .mailmap (Display-Only Anonymization)

If you don't want to rewrite history, you can add a `.mailmap` file to anonymize the display of author information:

```
# .mailmap
Anonymous Contributor <noreply@github.com> Waknis <mihirwaknis282@gmail.com>
```

This approach:
- ✅ Anonymizes display in `git log --use-mailmap` and `git shortlog`
- ✅ Doesn't break existing clones or forks
- ❌ Doesn't remove PII from git objects
- ❌ Adds another instance of PII to the repository (in the `.mailmap` file itself)

## Recommendations

For **complete PII removal**, use git-filter-repo and coordinate with all repository collaborators.

For **display anonymization only** without disruption, use `.mailmap` (note: the PII will still exist in git history and in the `.mailmap` file).
