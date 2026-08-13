# Live demo script - MAI201 Phase 2

Runs inside the final presentation, at slide 10. Target length **1:30-2:00**,
which keeps the whole talk inside the 7-10 minute limit.

Driver: **Someyah** (Documentation Lead). Abdulraouf answers model questions,
Mohamed answers deployment questions.

The service is deployed and responding at
**https://mai201-fraud-api.onrender.com**, released automatically by the latest
successful GitHub Actions CI/CD run with all three jobs green. Everything below
is a live call - there are no screenshots standing in for anything.

---

## Before you start

Do all of this **before** the presentation begins, not during it:

1. Open these four tabs, in this order, and leave them loaded:

   | Tab | URL |
   |---|---|
   | 1 | https://mai201-fraud-api.onrender.com/docs |
   | 2 | https://mai201-fraud-api.onrender.com/health |
   | 3 | https://github.com/AZabalawi/MLops-credit-card-fraud/actions |
   | 4 | `monitoring/reports/drift_report.html` (opened from the local repo) |

2. **Wake the service up.** The Render free plan puts an idle service to sleep,
   and the first request afterwards can take 30-60 seconds while the container
   restarts. Load tab 2 about five minutes before you present, and again right
   before you walk up. If `/health` answers instantly, you are warm.

3. Have `presentation/sample_requests/fraud.json` open in an editor so you can
   copy the body quickly, and have the backup curl commands below in a terminal
   you can switch to.

4. Zoom the browser to about 125% so the room can read the JSON.

---

## The demo, step by step

### 1. Swagger UI  (~20 seconds)

Switch to tab 1.

> "This is the public API, running on Render. The docs page is generated
> straight from the Pydantic schema, so it always matches what the service
> actually accepts."

Scroll once so the four routes are visible: `/`, `/health`, `/model-info`,
`/predict`.

### 2. Health  (~15 seconds)

Switch to tab 2.

> "Health returns 200 and `model_loaded: true`. If the model had failed to
> load this would be a 503 - we did not want a health check that says healthy
> while the service cannot actually score anything."

Expected response:

```json
{"status":"healthy","model_loaded":true,"model_path":"/srv/app/models/model.pkl","version":"2.0.0","detail":null}
```

The path is the one inside the container, which is a small but useful detail:
it shows the answer is coming from the deployed image, not from a laptop.

### 3. A real prediction  (~40 seconds)

Back to tab 1. Expand **POST /predict** → **Try it out**. Replace the request
body with the contents of `presentation/sample_requests/fraud.json` → **Execute**.

> "This is a real fraudulent transaction from our own held-out test split -
> all 30 features, no label. The model gives it a fraud probability of 0.9696."

Expected response:

```json
{
  "predicted_class": 1,
  "label": "fraud",
  "fraud_probability": 0.969592,
  "decision_threshold": 0.5,
  "is_fraud": true,
  "model_version": "2.0.0"
}
```

Then paste `legitimate.json` and execute again - it comes back with
`fraud_probability` 0.000235 and label `legitimate`. The contrast lands well,
and this is the exact call we confirmed against the deployed service, so it is
the safer one to lead with if you only have time for a single request.

Both payloads and both expected outputs were exported from the held-out test
split by `src/export_serving_assets.py`. They are the model's own numbers, not
figures written into a slide.

### 4. CI/CD  (~20 seconds)

Switch to tab 3.

> "Every push runs this workflow. Ruff lints the repository, pytest runs 34
> tests against the real model, then Docker builds the image and CI starts the
> container and probes it. Only after all of that passes does the deploy job
> call the Render deploy hook - which is what released the version we just
> called."

Open the **latest successful run on `main`** so the three green jobs are
visible: **Lint and test**, **Build and smoke-test the image**, **Deploy to
Render**. The page shows the commit it released, so read it off the screen rather
than quoting a number from the slides. Do not click into individual step logs -
there is not time.

If anyone asks how the deploy is authenticated: the Render deploy hook is
stored as a GitHub Actions secret and is never printed by the workflow. Do not
open the secret settings page on the projector.

### 5. Drift report  (~25 seconds)

Switch to tab 4.

> "And this is the EvidentlyAI report. We compared a batch with a simulated
> distribution shift against the training distribution, and it flagged 17 of
> the 30 features - a share of 0.567, above the 0.5 threshold, so it reports
> dataset drift. The control run on untouched data flags zero."

Scroll to the drift table so the per-column results are on screen.

Then hand back to the closing slide.

---

## Backup: curl commands

Use these if Swagger is slow or the projector struggles with the browser. Run
them from the repository root.

```bash
# health -> {"status":"healthy","model_loaded":true,...}
curl -s https://mai201-fraud-api.onrender.com/health

# a real fraud transaction
curl -s -X POST https://mai201-fraud-api.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d @presentation/sample_requests/fraud.json

# a real legitimate transaction
curl -s -X POST https://mai201-fraud-api.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d @presentation/sample_requests/legitimate.json

# validation is enforced: this returns 422
curl -s -X POST https://mai201-fraud-api.onrender.com/predict \
  -H 'Content-Type: application/json' -d '{"Time": 0}'
```

On Windows PowerShell, `curl` is an alias for `Invoke-WebRequest` and will not
accept these flags. Use `curl.exe` explicitly, or run the commands from Git Bash
or WSL.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| The service is cold and the first call hangs | Say so - "free plan, the container is waking up" - and keep talking through the CI/CD tab. Come back to it. |
| Render is down or the URL will not load | Run the service locally: `uvicorn app.main:app --port 8000`, then use `http://localhost:8000/docs`. Say clearly that you are showing the local container because the cloud service is unreachable. |
| Swagger rejects the pasted body | You almost certainly missed a bracket. Use the curl backup instead. |
| The drift report will not open | Show `monitoring/reports/drift_summary.json` instead - the headline numbers are in the first few lines. |

Never present a screenshot as a live call. If the live service is unavailable,
say that it is unavailable and show the local run instead.

---

## Timing for the whole presentation

| Slide | Speaker | Target |
|---|---|---|
| 1. Title and team | Abdulraouf | 0:40 |
| 2. Problem and data | Abdulraouf | 0:50 |
| 3. Phase 1 recap | Abdulraouf | 0:55 |
| 4. Phase 2 architecture | Mohamed | 0:45 |
| 5. FastAPI | Abdulraouf | 0:55 |
| 6. Docker and Render | Mohamed | 0:55 |
| 7. GitHub Actions | Mohamed | 0:50 |
| 8. EvidentlyAI monitoring | Someyah | 1:00 |
| 9. Retraining and Model Card | Someyah | 1:00 |
| 10. Live demo and close | Someyah | 0:40 + demo |
| **Total** | | **~8:30 with a 1:30 demo** |

Speaking time works out at roughly 2:25 for Abdulraouf, 2:30 for Mohamed and
3:10 for Someyah including the demo. Full speaker notes are in the notes pane
of every slide in `MAI201_Phase2_Final_Presentation.pptx`.
