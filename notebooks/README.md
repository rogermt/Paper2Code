```bash
 ==========================================
 ```
 
## 🧭 Working with notebooks (Kaggle) and nbstripout

If you edit notebooks from a cloud GPU notebook (e.g., Kaggle), follow these quick tips so commits stay clean and reproducible:

- Clone the feature branch on the remote machine:

```bash
git clone --branch feature/litellm-support --single-branch https://github.com/rogermt/Paper2Code.git
cd Paper2Code
```

- Configure git identity (one-time per session):

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

- We configured `.gitattributes` to use `nbstripout` for notebooks. Run this on the machine once to enable the hook (Kaggle has Python available):

```bash
pip install nbstripout
nbstripout --install
```

- Commit safely from the notebook (stage tracked changes only so untracked files like large outputs aren't accidentally added):

```bash
# stage tracked changes only
git add -u
git commit -m "WIP: incremental update from Kaggle"
git push origin $(git rev-parse --abbrev-ref HEAD)
```

- If you prefer not to push WIP history, push frequently as backups then squash (via interactive rebase) on your laptop before the final PR, or use GitHub's "Squash and merge" when merging.

- If notebooks grow large, consider running `nbstripout` to remove outputs before committing and/or use Git LFS for very large artifacts.