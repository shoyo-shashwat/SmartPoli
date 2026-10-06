# Deploy the Digital Twin demo on Render (step by step)

This runs the whole project as its own demo site. The twin switches on only through environment variables, so the same code is safe anywhere
the switches are left off.

## 1. Create the service
1. Render dashboard -> **New +** -> **Web Service** -> connect GitHub and pick `shoyo-shashwat/SmartPoli` (branch `main`).
2. **Name:** `smartpoli-demo`. **Region:** any (the existing app uses Oregon). **Runtime / Language:** `Docker`.
3. **Root Directory:** `backend`. **Dockerfile Path:** `./Dockerfile`. **Docker Build Context Directory:** `.` (this is what `render.yaml` uses too).
4. **Instance type:** Free is enough to try; if the service restarts with an "out of memory" message, switch the demo to **Starter**.
5. **Health Check Path:** `/`.

## 2. Environment variables (Environment tab -> Add Environment Variable)
| Key | Value |
|---|---|
| `SMARTPOLI_TWIN_ENABLED` | `1` |
| `SMARTPOLI_DEMO_LOGIN` | `1` |
| `SMARTPOLI_DEMO_SEED` | `1` |
| `SMARTPOLI_JWT_SECRET` | any long random text (use the "Generate" button) |
| `SMARTPOLI_ALLOWED_ORIGINS` | your demo URL, e.g. `https://smartpoli-demo.onrender.com` (add it after the first deploy if you do not know it yet) |

Leave `SMARTPOLI_DATABASE_URL` **unset**. The demo then uses a file database inside the container; every start re-creates the demo accounts and the two
patients automatically (reviews and notes you add are cleared when the service restarts, which is fine for a demo). Never point it at the live app's database.
You do not need the WhatsApp, Twilio, Groq or Gemini keys for the demo.

## 3. Send the private patient (B) to the deployed app
Patient A comes with the repo (ShanghaiT2DM, credited). Patient B (CGMacros) is not in the public repo. Pick ONE way:
- **Secret File (easiest):** service -> **Environment** -> **Secret Files** -> **Add Secret File** -> Filename `demo_b.json`, paste the whole contents of
  `backend/twin_demo_data/demo_b.json` from your computer (open it in a text editor, select all, copy). Render keeps it private and the app reads it from `/etc/secrets/demo_b.json`.
- **If the paste is refused for size,** use an environment variable instead: run `python ml/pack_private_demo.py backend/twin_demo_data/demo_b.json` on your computer, open the
  file `ml/demo_b.packed.txt`, copy its single line, and add it as `SMARTPOLI_TWIN_DEMO_B_GZ_B64`.
Do not commit either of these to GitHub.

## 4. Deploy and check
1. **Create Web Service.** The first build takes several minutes (it installs Tesseract and the Python packages).
2. When it says **Live**, open `https://<your-service>.onrender.com/twin/info`. You should see `{"enabled":true,"demo_login":{...}}`.
3. Open the site's `/login`. You should see the green **Doctor demo** card. Tap **Open the doctor demo**, then **Digital Twin** in the menu: both demo patients appear.
4. If `/twin/info` says "Not found", the `SMARTPOLI_TWIN_ENABLED` variable is missing or not exactly `1`. If only patient A appears, patient B was not delivered (step 3). Check **Logs** for "demo seeding failed".
5. Free plan services sleep after inactivity; the first open can take about a minute.

## 5. Updating
Every push to `main` redeploys automatically. Change a setting in the **Environment** tab and Render restarts the service.

## 6. The live app
Do not add the three `SMARTPOLI_TWIN_*`/`DEMO_*` variables to the live service: without them every twin endpoint answers "not found" and the live site looks exactly as before.
If the live service is deployed from this same repo, the next deploy also installs `xgboost-cpu` and creates four small empty tables in its database; nothing else changes.
