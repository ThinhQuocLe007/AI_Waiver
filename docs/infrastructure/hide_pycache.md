# How to Hide and Ignore `__pycache__`

If you are tired of seeing `__pycache__` folders cluttering your project, here are the three professional ways to handle them.

---

## 1. Professional Method: `.vscode/settings.json`

Instead of changing your personal settings, you can add a setting file **directly to your project**. This ensures that everyone who opens the project (and you across different computers) will have `__pycache__` hidden automatically.

1. Create a folder named `.vscode` in your project root (if it doesn't exist).
2. Create or edit a file named `settings.json` inside it.
3. Add the `files.exclude` block:

```json
{
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/*.egg-info": true
    }
}
```

> [!TIP]
> This is the preferred method for teams because the settings "live" with the code.

---

## 2. Ignore in Git (Best for your Repo)

You should **never** commit `__pycache__` to GitHub. It contains compiled bytecode that is specific to your machine.

Add these lines to your `.gitignore` file:

```text
# Python bytecode
__pycache__/
*.pyc
*.pyo
*.pyd
```

> [!NOTE]
> If you already committed `__pycache__` by mistake, you can remove it from Git tracking without deleting it from your computer using:
> `git rm -r --cached .` followed by a new commit.

---

## 3. Prevent Python from Creating Them

If you absolutely do not want Python to create these folders (useful for Docker or thin environments), you can set an environment variable.

### In your terminal:
```bash
export PYTHONDONTWRITEBYTECODE=1
```

### In your Python script:
```python
import sys
sys.dont_write_bytecode = True
```

> [!WARNING]
> Disabling bytecode generation can make your Python app start **slower**, especially in large projects, because Python has to re-compile your code every single time you run it.

---

## Summary

| Goal | Method |
| :--- | :--- |
| **I don't want to see it** | VS Code `Files: Exclude` |
| **I don't want to upload it** | Edit `.gitignore` |
| **I don't want it to exist** | `export PYTHONDONTWRITEBYTECODE=1` |
