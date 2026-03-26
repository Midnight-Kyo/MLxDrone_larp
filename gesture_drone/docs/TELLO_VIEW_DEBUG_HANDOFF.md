# Tello gesture view — debug handoff (Windows / PowerShell)

Send this file + a **full console log** + `git rev-parse HEAD` when asking for help.

## 1. Is `--enhance-stream` actually reaching Python?

After the `====` banner, **`tello_view.py` always prints:**

```text
  [flags] enhance_stream=True|False  (...)
```

- **`enhance_stream=False`** but you typed `tello --enhance-stream` → the **wrapper did not forward arguments** (wrong profile function, shadowed `tello` command, or old script).

**Fix checklist:**

1. **Reload aliases** from the repo (not a stale copy in `$PROFILE`):

   ```powershell
   cd <repo>\gesture_drone\scripts
   . .\Load-PowerShellAliases.ps1
   ```

2. **See what `tello` really is:**

   ```powershell
   Get-Command tello -All
   ```

   If something **other than Function** wins (e.g. another `tello` on `PATH`), rename the function or remove the conflict.

3. **Bypass the alias** (ground truth):

   ```powershell
   cd <repo>\gesture_drone\scripts
   python .\tello_view.py --enhance-stream
   # or short flag:
   python .\tello_view.py -E
   ```

   If this shows `enhance_stream=True` and the cyan **ENHANCED** badge, the problem is only PowerShell forwarding — use the updated `Load-PowerShellAliases.ps1` (`ValueFromRemainingArguments`).

## 2. UTF-8 decode error on first `command`

Example:

```text
'utf-8' codec can't decode byte 0xcc ...
```

Often **noisy UDP / stale packet** before the real `ok`. If the **second** `command` returns `ok`, you can ignore the first error unless it repeats forever.

## 3. Wrong responses after `setresolution` / `setfps` / `setbitrate`

Example: `setfps` returns `unknown command: setresolution`, or `streamon` returns `unknown command: setbitrate`.

The Ryze Tello **command/response channel is UDP**; answers can **get out of order** or echo the **previous** failure, especially when commands are sent back-to-back. The script already tries **fps** and **bitrate** separately after resolution; if firmware rejects some subcommands, treat logs as **best-effort**.

**Practical checks:**

- Power-cycle the drone, connect Wi‑Fi, try again.
- Some firmware accepts **`setresolution high`** but not **`setfps` / `setbitrate`** — stream may still work at default quality.

## 4. ENHANCED badge missing when `enhance_stream=True`

- Must be on **`tello_view.py` revision** that draws the badge **after** `draw_cam_panel` (top-left cyan label).
- If the flag is **True** but no badge: confirm you are running the script from the **same clone** you updated (`python -c "import pathlib; print(pathlib.Path('tello_view.py').resolve())"` from `gesture_drone\scripts`).

## 5. Video still looks noisy with enhancement on

`--enhance-stream` is **intentionally light** (bilateral + unsharp on the **full frame**). It will not turn the Tello feed into a studio camera; it only nudges edges/contrast for the detector stack.

## 6. Gesture model “5 classes” in the log

If the classifier checkpoint was trained with five extra/unknown class, the loader will report five names. That is independent of stream enhancement; verify `gesture_model.pt` matches the intended **4-class** flight set if commands look wrong.

---

## Minimal “status bundle” to paste in chat

```powershell
cd <repo>\gesture_drone\scripts
git rev-parse HEAD
Get-Command tello -All
python --version
python .\tello_view.py --help
python .\tello_view.py -E
```

Copy **all** console output from `tello` / `python ... tello_view.py` including the `[flags] enhance_stream=...` line.
