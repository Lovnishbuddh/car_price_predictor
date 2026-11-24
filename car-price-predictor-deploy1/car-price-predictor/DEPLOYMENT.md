# How to put this app ONLINE (free options)

The app is deployment-ready. This file shows 3 ways, from easiest to
most popular. All are free.

> Requirements for every option: a free account on the platform and this
> project folder (or the zip) on your computer.

---

## Option A - Railway (easiest, no GitHub needed, ~5 minutes)

Railway can upload your local folder directly - no GitHub account required.

1. Go to https://railway.app and sign up (free).
2. Click **New Project -> Deploy from local code** and install the
   Railway CLI when prompted, OR use the browser upload.
   - With the CLI:
     ```bash
     npm install -g @railway/cli
     railway login
     cd car-price-predictor
     railway init
     railway up
     ```
3. Railway auto-detects Python, installs `requirements.txt` and runs the app.
   If it asks for a start command, use:
   ```
   python app.py
   ```
4. Open **Settings -> Networking -> Generate Domain** to get your public URL.

Your app is now live at `https://<your-name>.up.railway.app`.

---

## Option B - Render (free, needs a GitHub account, ~10 minutes)

1. Create a GitHub repository called `car-price-predictor` and upload
   ALL files from this folder to it (drag & drop works on github.com:
   "uploading an existing file" - make sure `.joblib` and `.csv` upload too).
2. Go to https://render.com and sign up with GitHub.
3. **New -> Web Service -> connect your repository**.
4. Settings:
   - Environment: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
5. Click **Create Web Service**. Done - free URL like
   `https://car-price-predictor-xxxx.onrender.com`.

Note: free Render apps sleep after 15 min of inactivity; the first visit
after that takes ~30 seconds to wake up. For your demo, open the URL
5 minutes before you present.

---

## Option C - Vercel (needs a GitHub account)

This project already contains `vercel.json` and `api/index.py`, which is
everything Vercel needs.

1. Push the project to a GitHub repository (same as step 1 of Option B).
2. Go to https://vercel.com and sign in with GitHub.
3. **Add New -> Project -> Import** your `car-price-predictor` repo.
   Keep all defaults (Vercel reads `vercel.json` automatically).
4. Click **Deploy**. Live at `https://car-price-predictor.vercel.app`.

Note: the model file is 23 MB; the first request after a cold start may
take a few seconds while the serverless function loads it.

---

## Option D - PythonAnywhere (upload the zip in the browser, no CLI, no GitHub)

1. Sign up free at https://www.pythonanywhere.com.
2. **Files -> Upload files** -> upload this zip, then click it to unzip
   (or upload the unzipped folder).
3. **Web -> Add a new web app -> Manual configuration -> Python 3.10**.
4. Set the **source directory** to your project folder, and the
   **WSGI file** to:
   ```python
   import sys
   sys.path.insert(0, "/home/yourusername/car-price-predictor")
   from app import app as application
   ```
5. In a Bash console:
   ```bash
   pip3.10 install --user -r ~/car-price-predictor/requirements.txt
   ```
6. Reload the web app. Live at `https://yourusername.pythonanywhere.com`.

---

## Which one should I choose?

| Platform | GitHub needed | Effort | Best for |
|---|---|---|---|
| Railway | no | 5 min | fastest way to a public URL |
| Render | yes | 10 min | most reliable free option |
| Vercel | yes | 10 min | if you specifically want Vercel |
| PythonAnywhere | no | 15 min | pure browser upload, zero CLI |

For tomorrow's demo, **Railway** is the quickest path.

## Troubleshooting

- **"ModuleNotFoundError: src"** on deployment -> make sure the whole
  project folder is uploaded, not only `app.py`.
- **Slow first prediction** -> the 23 MB model loads once at startup;
  later predictions are instant.
- **Port errors** -> never hardcode a port in the start command;
  the app already reads the `PORT` environment variable.
