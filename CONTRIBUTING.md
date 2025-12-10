## Quick contributor notes

This small guide explains the repository remotes and how to keep notebooks clean when committing.

1) Remotes

- `origin` should point to your fork (e.g. `https://github.com/rogermt/Paper2Code`).
- `upstream` typically points to the original project `https://github.com/going-doer/Paper2Code.git`.

Commands to verify:

```powershell
git remote -v
git branch -vv
```

2) Working with branches

- Create or track a branch from the remote branch:

```powershell
# create local tracking branch from origin
git fetch origin
git checkout -b feature/litellm-support origin/feature/litellm-support
```

3) Pushing safely

- Push your local branch to your fork (origin):

```powershell
git push -u origin feature/litellm-support
```

- If you accidentally pushed to `upstream`, you can push the branch to your fork using the command above.

4) Notebook output hygiene

- This repo uses `nbstripout` to remove outputs from notebooks during commits. To install hooks locally:

```powershell
python -m pip install -r requirements.txt
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

- Alternatively, you can run `nbstripout --install` in your environment (it will create a git filter hook).

5) Working on Kaggle

- Kaggle kernels are ephemeral. Clone your fork and work there for GPU access:

```bash
git clone -b feature/litellm-support https://github.com/rogermt/Paper2Code.git
```

- If you want to push from Kaggle, you'll need to provide credentials (PAT or SSH). Prefer pushing from your laptop where credentials are safer.

If anything here is unclear or you want me to add a CI check that ensures notebooks contain no outputs, tell me and I can add it.
