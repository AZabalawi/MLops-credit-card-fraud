/*
 * build_deck.js
 * -------------
 * Generates presentation/MAI201_Phase2_Final_Presentation.pptx.
 *
 * The deck is scripted rather than hand-built so it can be regenerated after a
 * number changes. Every figure in it comes from the project's own files:
 * metrics/eval_metrics.json, metrics/data_quality_report.json,
 * monitoring/reports/*_summary.json and monitoring/reports/retraining_summary.json.
 *
 *   cd presentation && npm install pptxgenjs && node build_deck.js
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const readJson = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));

const evalMetrics = readJson("metrics/eval_metrics.json");
const quality = readJson("metrics/data_quality_report.json");
const driftSummary = readJson("monitoring/reports/drift_summary.json");
const refSummary = readJson("monitoring/reports/reference_summary.json");
const retrain = readJson("monitoring/reports/retraining_summary.json");
const deploy = readJson("presentation/deploy_info.json");

const API = deploy.public_api_url;
const REPO = deploy.repo_url;

/* palette: navy operations console, signal red for fraud, teal for infrastructure */
const NAVY = "10233F";
const NAVY_SOFT = "1C3A5E";
const RED = "C81E3C";
const TEAL = "1C7293";
const GREEN = "1E7A50";
const PAPER = "FFFFFF";
const CARD = "F2F5F9";
const INK = "17222E";
const MUTED = "5A6B7D";

const TITLE_FONT = "Cambria";
const BODY_FONT = "Calibri";
const MONO = "Courier New";

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Abdulraouf Zabalawi, Mohamed Roble, Someyah Balashi";
pres.title = "MAI201 MLOps Phase 2 - Credit Card Fraud Detection";

const W = 13.33;

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */
function slideTitle(slide, text, speaker, dark) {
  slide.addText(text, {
    x: 0.6, y: 0.42, w: 9.4, h: 0.8, margin: 0,
    fontFace: TITLE_FONT, fontSize: 34, bold: true,
    color: dark ? PAPER : NAVY, align: "left", valign: "middle",
  });
  if (speaker) {
    slide.addShape(pres.ShapeType.roundRect, {
      x: 10.35, y: 0.5, w: 2.4, h: 0.46, rectRadius: 0.22,
      fill: { color: dark ? NAVY_SOFT : CARD }, line: { color: dark ? NAVY_SOFT : CARD },
    });
    slide.addText(speaker, {
      x: 10.35, y: 0.5, w: 2.4, h: 0.46, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, bold: true,
      color: dark ? "CADCFC" : TEAL, align: "center", valign: "middle",
    });
  }
}

/* the repeated motif: a filled circle holding a number or short glyph */
function numberCircle(slide, x, y, d, label, color) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color }, line: { color },
  });
  slide.addText(label, {
    x, y, w: d, h: d, margin: 0,
    fontFace: BODY_FONT, fontSize: d > 0.55 ? 15 : 12, bold: true,
    color: PAPER, align: "center", valign: "middle",
  });
}

function card(slide, opts) {
  slide.addShape(pres.ShapeType.roundRect, {
    x: opts.x, y: opts.y, w: opts.w, h: opts.h, rectRadius: 0.09,
    fill: { color: opts.fill || CARD },
    line: { color: opts.fill || CARD },
  });
}

function statBlock(slide, x, y, w, value, label, color) {
  slide.addText(value, {
    x, y, w, h: 0.85, margin: 0,
    fontFace: TITLE_FONT, fontSize: 40, bold: true, color,
    align: "left", valign: "bottom",
  });
  slide.addText(label, {
    x, y: y + 0.88, w, h: 0.5, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, color: MUTED, align: "left", valign: "top",
  });
}

/* icon + heading + description row, used on several slides */
function iconRow(slide, x, y, w, glyph, heading, body, color) {
  numberCircle(slide, x, y + 0.04, 0.46, glyph, color);
  slide.addText(heading, {
    x: x + 0.66, y, w: w - 0.66, h: 0.3, margin: 0,
    fontFace: BODY_FONT, fontSize: 14, bold: true, color: INK, valign: "middle",
  });
  slide.addText(body, {
    x: x + 0.66, y: y + 0.29, w: w - 0.66, h: 0.42, margin: 0,
    fontFace: BODY_FONT, fontSize: 11.5, color: MUTED, valign: "top",
  });
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

/* ================================================================== */
/* 1. Title                                                            */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addText("Credit Card Fraud Detection", {
    x: 0.75, y: 1.5, w: 7.4, h: 0.95, margin: 0,
    fontFace: TITLE_FONT, fontSize: 40, bold: true, color: PAPER, valign: "middle",
  });
  s.addText("From a DVC pipeline to a monitored API in the cloud", {
    x: 0.75, y: 2.45, w: 7.4, h: 0.5, margin: 0,
    fontFace: BODY_FONT, fontSize: 17, color: "CADCFC", valign: "middle",
  });
  s.addText("MAI201 MLOps  ·  Phase 2 Final Presentation", {
    x: 0.75, y: 3.05, w: 7.4, h: 0.4, margin: 0,
    fontFace: BODY_FONT, fontSize: 13, italic: true, color: "8FA6C4", valign: "middle",
  });

  const delivered = [
    "Public HTTPS API on Render",
    "Docker image, built and smoke-tested in CI",
    "GitHub Actions: lint, tests, deploy",
    "EvidentlyAI drift reports + retraining",
  ];
  delivered.forEach((d, i) => {
    const y = 3.85 + i * 0.44;
    numberCircle(s, 0.82, y + 0.09, 0.17, "", RED);
    s.addText(d, {
      x: 1.2, y, w: 6.9, h: 0.34, margin: 0,
      fontFace: BODY_FONT, fontSize: 12.5, color: "CADCFC", valign: "middle",
    });
  });

  s.addText("Kaggle / ULB dataset  ·  284,807 transactions  ·  0.17% fraud", {
    x: 0.75, y: 5.85, w: 7.4, h: 0.4, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, color: "8FA6C4",
  });

  const team = [
    ["Abdulraouf Zabalawi", "Project & ML Lead", "Model, FastAPI, Model Card, MLflow", RED],
    ["Mohamed Roble", "Engineering Lead", "Docker, Render deployment, CI/CD", TEAL],
    ["Someyah Balashi", "Documentation Lead", "EvidentlyAI, retraining, slides", GREEN],
  ];
  team.forEach(([name, role, work, color], i) => {
    const y = 1.55 + i * 1.42;
    card(s, { x: 8.5, y, w: 4.2, h: 1.2, fill: NAVY_SOFT });
    numberCircle(s, 8.75, y + 0.37, 0.46, name.charAt(0), color);
    s.addText(name, {
      x: 9.35, y: y + 0.16, w: 3.2, h: 0.32, margin: 0,
      fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: PAPER, valign: "middle",
    });
    s.addText(role, {
      x: 9.35, y: y + 0.45, w: 3.2, h: 0.28, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, bold: true, color: "CADCFC", valign: "middle",
    });
    s.addText(work, {
      x: 9.35, y: y + 0.71, w: 3.2, h: 0.4, margin: 0,
      fontFace: BODY_FONT, fontSize: 10, color: "8FA6C4", valign: "top",
    });
  });

  s.addNotes(
    "ABDULRAOUF (0:40). Good morning. We are presenting Phase 2 of our credit card " +
    "fraud detection project. In Phase 1 we built a reproducible training pipeline with " +
    "DVC and MLflow. In Phase 2 we took that trained model and put it into production: " +
    "a FastAPI service, a Docker image, a live deployment on Render, automated CI/CD, " +
    "and drift monitoring with a retraining path. I am Abdulraouf, project and ML lead. " +
    "Mohamed handled the container, the cloud deployment and CI/CD. Someyah handled " +
    "monitoring, retraining and documentation. We will each take our own part."
  );
}

/* ================================================================== */
/* 2. Problem + dataset                                                */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "The problem, and the data", "Abdulraouf", false);

  statBlock(s, 0.75, 1.45, 3.0, quality.n_rows.toLocaleString(), "transactions over two days", NAVY);
  statBlock(s, 4.55, 1.45, 3.0, quality.fraud_count.toString(), "of them fraudulent", RED);
  statBlock(s, 8.35, 1.45, 4.2, `${quality.fraud_rate_percent}%`, "fraud rate — one in every 580", RED);

  card(s, { x: 0.75, y: 3.35, w: 5.9, h: 3.35 });
  s.addText("What makes this hard", {
    x: 1.05, y: 3.55, w: 5.3, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: INK,
  });
  s.addText(
    [
      { text: "Accuracy is useless here. Answering \"legitimate\" to everything scores 99.83%, so we judge the model on precision, recall, F1 and PR-AUC instead.", options: { bullet: true, breakLine: true } },
      { text: "Only 492 fraud rows exist in total. A handful of transactions moves recall by a full percentage point.", options: { bullet: true, breakLine: true } },
      { text: "V1–V28 are PCA components published in place of the real fields, so no feature has a human-readable meaning.", options: { bullet: true, breakLine: false } },
    ],
    {
      x: 1.05, y: 3.95, w: 5.3, h: 2.5, margin: 0,
      fontFace: BODY_FONT, fontSize: 12, color: MUTED,
      paraSpaceAfter: 8, valign: "top",
    }
  );

  card(s, { x: 7.05, y: 3.35, w: 5.55, h: 3.35 });
  s.addText("The 30 inputs the API accepts", {
    x: 7.35, y: 3.55, w: 5.0, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: INK,
  });
  const fields = [
    ["Time", "seconds since the first transaction"],
    ["V1 – V28", "anonymised PCA components"],
    ["Amount", "transaction value, 0 to 25,691.16"],
    ["Class", "the label — never sent to /predict"],
  ];
  fields.forEach(([f, d], i) => {
    const y = 4.02 + i * 0.62;
    s.addText(f, {
      x: 7.35, y, w: 1.7, h: 0.35, margin: 0,
      fontFace: MONO, fontSize: 12, bold: true,
      color: i === 3 ? RED : TEAL, valign: "middle",
    });
    s.addText(d, {
      x: 9.1, y, w: 3.3, h: 0.35, margin: 0,
      fontFace: BODY_FONT, fontSize: 11.5, color: MUTED, valign: "middle",
    });
  });

  s.addNotes(
    "ABDULRAOUF (0:50). The dataset is real card transactions from European cardholders " +
    "over two days in 2013. 284,807 transactions, 492 of which are fraud — that is a " +
    "fraud rate of 0.1727%, roughly one in 580. That imbalance drives every decision we " +
    "made. Accuracy is meaningless: a model that says legitimate to everything already " +
    "scores 99.83%. So we report precision, recall, F1 and PR-AUC. The inputs are Time, " +
    "the 28 PCA components V1 to V28, and Amount. Class is the label and is never sent " +
    "to the API — the API rejects a request that includes it."
  );
}

/* ================================================================== */
/* 3. Phase 1 recap                                                    */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "Phase 1 recap: DVC pipeline + MLflow", "Abdulraouf", false);

  const stages = [
    ["1", "prepare", "quality report, drop 1,081 duplicates, stratified 80/20 split"],
    ["2", "train", "scale Time and Amount inside the pipeline, then RandomForest"],
    ["3", "evaluate", "imbalance-aware metrics on the held-out 56,746 rows"],
  ];
  stages.forEach(([n, name, desc], i) => {
    const x = 0.75 + i * 4.02;
    card(s, { x, w: 3.7, y: 1.45, h: 1.75 });
    numberCircle(s, x + 0.28, 1.7, 0.5, n, TEAL);
    s.addText(name, {
      x: x + 0.9, y: 1.68, w: 2.6, h: 0.42, margin: 0,
      fontFace: MONO, fontSize: 15, bold: true, color: NAVY, valign: "middle",
    });
    s.addText(desc, {
      x: x + 0.28, y: 2.25, w: 3.15, h: 0.85, margin: 0,
      fontFace: BODY_FONT, fontSize: 11.5, color: MUTED, valign: "top",
    });
    if (i < 2) {
      s.addText("→", {
        x: x + 3.72, y: 2.0, w: 0.3, h: 0.4, margin: 0,
        fontFace: BODY_FONT, fontSize: 20, bold: true, color: TEAL, align: "center",
      });
    }
  });

  card(s, { x: 0.75, y: 3.45, w: 6.2, h: 3.25 });
  s.addText("Baseline on the held-out test set", {
    x: 1.05, y: 3.65, w: 5.6, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: INK,
  });
  const rows = [
    ["Precision", evalMetrics.precision],
    ["Recall", evalMetrics.recall],
    ["F1", evalMetrics.f1_score],
    ["ROC-AUC", evalMetrics.roc_auc],
    ["PR-AUC", evalMetrics.pr_auc],
  ];
  rows.forEach(([k, v], i) => {
    const y = 4.12 + i * 0.47;
    s.addText(k, {
      x: 1.05, y, w: 2.2, h: 0.38, margin: 0,
      fontFace: BODY_FONT, fontSize: 12.5, color: MUTED, valign: "middle",
    });
    s.addText(v.toFixed(4), {
      x: 3.2, y, w: 1.3, h: 0.38, margin: 0,
      fontFace: MONO, fontSize: 13, bold: true, color: NAVY, valign: "middle",
    });
    s.addShape(pres.ShapeType.rect, {
      x: 4.65, y: y + 0.13, w: 1.95 * v, h: 0.13,
      fill: { color: i === 1 ? RED : TEAL }, line: { color: i === 1 ? RED : TEAL },
    });
  });

  card(s, { x: 7.35, y: 3.45, w: 5.25, h: 3.25 });
  s.addText("Tracked, not remembered", {
    x: 7.65, y: 3.65, w: 4.65, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: INK,
  });
  iconRow(s, 7.65, 4.12, 4.65, "D", "DVC",
    "dvc.yaml wires the three stages; dvc.lock pins data, code and model hashes.", TEAL);
  iconRow(s, 7.65, 4.95, 4.65, "M", "MLflow",
    "Baseline plus two experiments, logged to a local SQLite store.", RED);
  iconRow(s, 7.65, 5.78, 4.65, "P", "params.yaml",
    "One config file feeds every stage, so a change re-runs only what it affects.", GREEN);

  s.addNotes(
    "ABDULRAOUF (0:55). A quick recap of Phase 1, because Phase 2 sits directly on top of " +
    "it. The DVC pipeline has three stages. Prepare loads the raw CSV, writes a data " +
    "quality report, removes 1,081 duplicate transactions and does a stratified split — " +
    "stratified because with 0.17% positives an ordinary split can leave the two sets " +
    "with different fraud rates. Train builds a scikit-learn Pipeline: a StandardScaler " +
    "on Time and Amount only, since V1 to V28 are already PCA outputs, followed by a " +
    "Random Forest with class_weight balanced. The scaler lives inside the pipeline, so " +
    "it is fitted on training rows only and there is no leakage. Evaluate scores the " +
    "held-out set. Precision 0.88, recall 0.73, PR-AUC 0.79. Everything is tracked by " +
    "DVC and logged to MLflow, so these numbers are reproducible rather than remembered."
  );
}

/* ================================================================== */
/* 4. Phase 2 architecture                                             */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "What Phase 2 added", "Mohamed", false);
  s.addImage({
    path: path.join(ROOT, "docs/architecture_phase2.png"),
    x: 0.62, y: 1.28, w: 12.1, h: 5.55, sizing: { type: "contain", w: 12.1, h: 5.55 },
  });
  s.addNotes(
    "MOHAMED (0:45). This is the whole system on one slide. On the left is the Phase 1 " +
    "training pipeline Abdulraouf just described, ending in models/model.pkl. The middle " +
    "column is what Phase 2 added on the serving side: that same pickle is loaded by a " +
    "FastAPI application, the application is baked into a Docker image, and the image is " +
    "deployed to Render as a web service with a public HTTPS URL. On the right is CI/CD " +
    "— every push runs lint, tests and a container smoke test in GitHub Actions, and only " +
    "then does it deploy. Along the bottom is the monitoring loop: EvidentlyAI compares " +
    "live-shaped batches against the training distribution, and if it finds drift and we " +
    "have labelled data, retraining produces a candidate that has to earn its promotion. " +
    "I will take the deployment and CI/CD parts."
  );
}

/* ================================================================== */
/* 5. FastAPI                                                          */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "Serving the real model with FastAPI", "Abdulraouf", false);

  const endpoints = [
    ["/", "what this service is"],
    ["/health", "200 only when the model is really loaded, 503 otherwise"],
    ["/model-info", "model type, 30 features, hyperparameters, artifact hash"],
    ["/predict", "one transaction in, class and fraud probability out"],
    ["/docs", "Swagger UI, generated from the Pydantic schema"],
  ];
  card(s, { x: 0.75, y: 1.4, w: 5.85, h: 3.35 });
  endpoints.forEach(([ep, desc], i) => {
    const y = 1.62 + i * 0.62;
    s.addText(ep, {
      x: 1.02, y, w: 1.65, h: 0.4, margin: 0,
      fontFace: MONO, fontSize: 12.5, bold: true, color: TEAL, valign: "middle",
    });
    s.addText(desc, {
      x: 2.72, y, w: 3.65, h: 0.4, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, valign: "middle",
    });
  });

  card(s, { x: 0.75, y: 4.98, w: 5.85, h: 1.72 });
  s.addText("Two details that matter", {
    x: 1.02, y: 5.13, w: 5.3, h: 0.3, margin: 0,
    fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: INK,
  });
  s.addText(
    [
      { text: "The whole sklearn Pipeline is served, not just the forest — so scaling at inference is identical to training.", options: { bullet: true, breakLine: true } },
      { text: "The model is loaded once at startup, not per request.", options: { bullet: true, breakLine: false } },
    ],
    {
      x: 1.02, y: 5.46, w: 5.3, h: 1.1, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, paraSpaceAfter: 6, valign: "top",
    }
  );

  card(s, { x: 7.0, y: 1.4, w: 5.6, h: 5.3, fill: NAVY });
  s.addText("A real request against the deployed service", {
    x: 7.3, y: 1.58, w: 5.0, h: 0.32, margin: 0,
    fontFace: BODY_FONT, fontSize: 12.5, bold: true, color: "CADCFC",
  });
  s.addText(
    "POST /predict\n" +
    "{\n" +
    '  "Time": 406.0,\n' +
    '  "V1": -2.3122, "V2": 1.9520,\n' +
    '  ...  "V28": -0.0210,\n' +
    '  "Amount": 0.0\n' +
    "}",
    {
      x: 7.3, y: 1.98, w: 5.0, h: 1.75, margin: 0,
      fontFace: MONO, fontSize: 11, color: "9FE8D6", valign: "top", lineSpacing: 15,
    }
  );
  s.addText("response", {
    x: 7.3, y: 3.82, w: 5.0, h: 0.28, margin: 0,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: "8FA6C4",
  });
  s.addText(
    "{\n" +
    '  "predicted_class": 1,\n' +
    '  "label": "fraud",\n' +
    '  "fraud_probability": 0.969592,\n' +
    '  "decision_threshold": 0.5,\n' +
    '  "is_fraud": true\n' +
    "}",
    {
      x: 7.3, y: 4.12, w: 5.0, h: 1.75, margin: 0,
      fontFace: MONO, fontSize: 11, color: "FFC7CF", valign: "top", lineSpacing: 15,
    }
  );
  s.addText("A genuine fraud row from our own test split.", {
    x: 7.3, y: 6.0, w: 5.0, h: 0.5, margin: 0,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: "8FA6C4", valign: "top",
  });

  s.addNotes(
    "ABDULRAOUF (0:55). The API exposes five routes. Health is the one worth calling out: " +
    "it returns 200 only when the model actually loaded, and 503 otherwise. A health " +
    "check that says healthy while the service cannot score anything is worse than no " +
    "health check at all. Predict takes exactly the 30 training features, validated by " +
    "Pydantic — a missing field, a wrong type, or an extra field like Class all come back " +
    "as 422. Two implementation details. First, we serve the entire scikit-learn " +
    "Pipeline, not just the classifier, so the Time and Amount scaling at inference is " +
    "byte-for-byte what it was at training. Second, the model is loaded once at startup " +
    "and reused. On the right is a real request and the real response: a fraud row from " +
    "our test split, scored at 0.9696."
  );
}

/* ================================================================== */
/* 6. Docker + Render                                                  */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "Containerised and deployed", "Mohamed", false);

  const items = [
    ["1", "python:3.11-slim", "Small base. Only the serving dependencies go in — no DVC, MLflow or Evidently."],
    ["2", "Pinned requirements", "requirements-api.txt is fully pinned, so the container loads model.pkl with the scikit-learn that fitted it."],
    ["3", "Uvicorn on $PORT", "Render injects the port; the container honours it and falls back to 8000 locally."],
    ["4", "No secrets, non-root", "The image holds code and a 1.9 MB model. The deploy hook lives in GitHub secrets."],
  ];
  items.forEach(([n, head, body], i) => {
    const x = 0.75 + (i % 2) * 6.15;
    const y = 1.4 + Math.floor(i / 2) * 1.62;
    card(s, { x, y, w: 5.8, h: 1.4 });
    numberCircle(s, x + 0.3, y + 0.28, 0.5, n, i % 2 === 0 ? TEAL : NAVY_SOFT);
    s.addText(head, {
      x: x + 0.95, y: y + 0.22, w: 4.6, h: 0.35, margin: 0,
      fontFace: BODY_FONT, fontSize: 14, bold: true, color: INK, valign: "middle",
    });
    s.addText(body, {
      x: x + 0.95, y: y + 0.6, w: 4.6, h: 0.68, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, valign: "top",
    });
  });

  card(s, { x: 0.75, y: 4.72, w: 11.85, h: 1.98, fill: NAVY });
  numberCircle(s, 1.1, 4.93, 0.3, "", GREEN);
  s.addText("Live now", {
    x: 1.52, y: 4.9, w: 4.0, h: 0.34, margin: 0,
    fontFace: BODY_FONT, fontSize: 12, bold: true, color: "8FA6C4", valign: "middle",
  });
  s.addText(API, {
    x: 1.1, y: 5.26, w: 11.1, h: 0.5, margin: 0,
    fontFace: MONO, fontSize: 19, bold: true, color: "9FE8D6", valign: "middle",
  });
  s.addText(
    '/health  ->  {"status":"healthy","model_loaded":true,"model_path":"/srv/app/models/model.pkl"}',
    {
      x: 1.1, y: 5.82, w: 11.1, h: 0.34, margin: 0,
      fontFace: MONO, fontSize: 10.5, color: "CADCFC", valign: "middle",
    }
  );
  s.addText(`Swagger UI at ${API}/docs   ·   released automatically by the latest successful CI/CD run`, {
    x: 1.1, y: 6.18, w: 11.1, h: 0.34, margin: 0,
    fontFace: BODY_FONT, fontSize: 11, color: "8FA6C4", valign: "middle",
  });

  s.addNotes(
    "MOHAMED (0:55). The container is deliberately small. It starts from python 3.11-slim " +
    "and installs only what serving needs — FastAPI, Uvicorn, scikit-learn, pandas, " +
    "numpy. DVC, MLflow and Evidently are training and monitoring tools, so they stay " +
    "out of the image. requirements-api.txt is fully pinned, and that matters more than " +
    "it sounds: model.pkl is a pickled scikit-learn object, so the container has to " +
    "install the same version that fitted it, otherwise loading it is a gamble. The " +
    "container runs as a non-root user, holds no secrets, and listens on the port Render " +
    "injects. It is deployed on Render as a Docker web service with the health check " +
    "pointed at /health, and that is the public URL on the screen. It is live right " +
    "now: /health comes back healthy, with model_loaded true and the model path " +
    "inside the container. This version was released by the CI/CD pipeline, not by " +
    "anyone clicking deploy. We will hit it live in a moment."
  );
}

/* ================================================================== */
/* 7. CI/CD                                                            */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "CI/CD with GitHub Actions", "Mohamed", false);

  const steps = [
    ["1", "Ruff lint", "the whole repo, clean"],
    ["2", "pytest", "34 tests against the real model"],
    ["3", "docker build", "then the container is started and probed"],
    ["4", "deploy", "Render deploy hook, main branch only"],
  ];
  steps.forEach(([n, head, body], i) => {
    const x = 0.75 + i * 3.06;
    card(s, { x, y: 1.5, w: 2.75, h: 2.05 });
    numberCircle(s, x + 1.1, 1.75, 0.55, n, i === 3 ? RED : TEAL);
    s.addText(head, {
      x: x + 0.15, y: 2.42, w: 2.45, h: 0.35, margin: 0,
      fontFace: BODY_FONT, fontSize: 14, bold: true, color: INK, align: "center",
    });
    s.addText(body, {
      x: x + 0.15, y: 2.78, w: 2.45, h: 0.65, margin: 0,
      fontFace: BODY_FONT, fontSize: 10.5, color: MUTED, align: "center", valign: "top",
    });
    if (i < 3) {
      s.addText("→", {
        x: x + 2.78, y: 2.35, w: 0.26, h: 0.4, margin: 0,
        fontFace: BODY_FONT, fontSize: 18, bold: true, color: TEAL, align: "center",
      });
    }
  });

  card(s, { x: 0.75, y: 3.8, w: 5.85, h: 2.72 });
  s.addText("The container is tested, not just built", {
    x: 1.05, y: 4.0, w: 5.3, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 14.5, bold: true, color: INK,
  });
  s.addText(
    [
      { text: "CI starts the image it just built, waits for it to come up, and checks /health reports the model as loaded.", options: { bullet: true, breakLine: true } },
      { text: "It posts a real transaction to /predict and asserts the probability is between 0 and 1.", options: { bullet: true, breakLine: true } },
      { text: "It sends a malformed body and asserts a 422 comes back.", options: { bullet: true, breakLine: false } },
    ],
    {
      x: 1.05, y: 4.42, w: 5.3, h: 2.1, margin: 0,
      fontFace: BODY_FONT, fontSize: 11.5, color: MUTED, paraSpaceAfter: 8, valign: "top",
    }
  );

  card(s, { x: 7.0, y: 3.8, w: 5.6, h: 2.72 });
  s.addText("Pull request vs. main", {
    x: 7.3, y: 4.0, w: 5.0, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 14.5, bold: true, color: INK,
  });
  iconRow(s, 7.3, 4.5, 5.0, "PR", "Pull request",
    "Lint, tests and the container smoke test run. Deployment is skipped.", TEAL);
  iconRow(s, 7.3, 5.35, 5.0, "M", "Push to main",
    "The same checks run first; only if they all pass does the deploy job fire.", RED);
  s.addText("Render auto-deploy is switched off on purpose — CI is the only route to production.", {
    x: 7.3, y: 6.12, w: 5.0, h: 0.45, margin: 0,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED, valign: "top",
  });
  card(s, { x: 0.75, y: 6.76, w: 11.85, h: 0.42, fill: "E7F3EC" });
  s.addText(
    "Latest main-branch CI/CD run — all three jobs green: lint and test, build and smoke-test the image, deploy to Render.",
    {
      x: 1.05, y: 6.76, w: 11.25, h: 0.42, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, bold: true, color: GREEN, valign: "middle",
    }
  );

  s.addNotes(
    "MOHAMED (0:50). The workflow is one file, .github/workflows/ci-cd.yml, and it runs " +
    "on every push and every pull request. Four things happen in order. Ruff lints the " +
    "whole repository. Pytest runs our 34 tests against the real model artifact, not a " +
    "stub. Then Docker builds the image — and this is the part I want to highlight — CI " +
    "does not just build it, it starts the container, waits for it, checks that health " +
    "reports the model as loaded, posts a real transaction to predict and asserts the " +
    "probability is in range, and sends a broken payload to confirm it gets a 422. Only " +
    "if all of that passes, and only on the main branch, does the deploy job fire the " +
    "Render deploy hook, which is stored as a GitHub secret and never appears in the " +
    "code. We deliberately turned Render's own auto-deploy off, so nothing reaches " +
    "production without passing CI first. The run on screen is the latest successful " +
    "one on main, where all three jobs passed and the deploy job released the version " +
    "that is serving right now."
  );
}

/* ================================================================== */
/* 8. Evidently monitoring                                             */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "Drift monitoring with EvidentlyAI", "Someyah", false);

  s.addChart(
    pres.ChartType.bar,
    [{
      name: "Drifted columns",
      labels: ["Healthy batch", "Simulated drift"],
      values: [refSummary.drifted_columns_count, driftSummary.drifted_columns_count],
    }],
    {
      x: 0.75, y: 1.45, w: 5.85, h: 3.05,
      barDir: "col", chartColors: [TEAL, RED],
      varyColors: true,
      showTitle: true, title: "Columns flagged as drifted (out of 30)",
      titleFontSize: 13, titleColor: INK, titleFontFace: BODY_FONT,
      showValue: true, dataLabelPosition: "outEnd",
      dataLabelFontSize: 14, dataLabelFontBold: true, dataLabelColor: INK,
      dataLabelFontFace: BODY_FONT,
      showLegend: false,
      valAxisMaxVal: 30, valAxisMajorUnit: 10,
      catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
      catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
      catAxisLabelFontFace: BODY_FONT, valAxisLabelFontFace: BODY_FONT,
      valGridLine: { color: "E2E8F0", size: 1 },
      catGridLine: { style: "none" },
      barGapWidthPct: 120,
    }
  );

  card(s, { x: 0.75, y: 4.72, w: 5.85, h: 1.98 });
  s.addText("How the comparison is set up", {
    x: 1.05, y: 4.9, w: 5.3, h: 0.32, margin: 0,
    fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: INK,
  });
  s.addText(
    "Reference: a fixed 10,000-row sample of the training split.\n" +
    "Current: 5,000 production-shaped rows.\n" +
    "30 numeric features compared with Evidently's DataDriftPreset. In the generated " +
    "reports Evidently selected normalized Wasserstein distance, threshold 0.1. " +
    "Class is excluded because the API never receives the label.",
    {
      x: 1.05, y: 5.22, w: 5.3, h: 1.42, margin: 0,
      fontFace: BODY_FONT, fontSize: 10.2, color: MUTED, valign: "top", lineSpacing: 13,
    }
  );

  card(s, { x: 7.0, y: 1.45, w: 5.6, h: 2.18 });
  numberCircle(s, 7.3, 1.68, 0.46, "OK", GREEN);
  s.addText("Healthy batch", {
    x: 7.96, y: 1.66, w: 4.4, h: 0.34, margin: 0,
    fontFace: BODY_FONT, fontSize: 14.5, bold: true, color: INK, valign: "middle",
  });
  s.addText(
    `Untouched rows from the test split. Evidently flags ${refSummary.drifted_columns_count} of 30 columns, ` +
    `share ${refSummary.drifted_columns_share.toFixed(3)} — no dataset drift. This is the control: ` +
    "it shows the monitor is not simply alarmed by everything.",
    {
      x: 7.3, y: 2.22, w: 5.0, h: 1.5, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, valign: "top",
    }
  );

  card(s, { x: 7.0, y: 3.82, w: 5.6, h: 2.88, fill: "FBE9EC" });
  numberCircle(s, 7.3, 4.05, 0.46, "!", RED);
  s.addText("Simulated drift", {
    x: 7.96, y: 4.03, w: 4.4, h: 0.34, margin: 0,
    fontFace: BODY_FONT, fontSize: 14.5, bold: true, color: INK, valign: "middle",
  });
  s.addText(
    `A seeded shift applied to 17 columns — larger amounts, later timestamps, ` +
    `moved PCA components. Evidently flags ${driftSummary.drifted_columns_count} of 30, ` +
    `share ${driftSummary.drifted_columns_share.toFixed(3)}, above the 0.5 threshold, ` +
    "so dataset drift is reported. Clearly labelled as simulated — it is not real customer traffic.",
    {
      x: 7.3, y: 4.6, w: 5.0, h: 1.95, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, valign: "top",
    }
  );

  s.addNotes(
    "SOMEYAH (1:00). Monitoring answers one question: is the traffic still shaped like " +
    "what the model learned from? We use EvidentlyAI with the DataDriftPreset over the " +
    "30 input features. The reference is a fixed 10,000-row sample of the training split. " +
    "We exclude Class, because the API never receives a label at prediction time. " +
    "Evidently picks the test itself from the data - at these batch sizes it chose " +
    "normalized Wasserstein distance with a threshold of 0.1, and that choice is " +
    "recorded in our JSON summary next to every score. We ran " +
    "two scenarios. The first is a control: untouched rows from the test split. Evidently " +
    "flags zero of 30 columns — no drift, which tells us the monitor is not just alarmed " +
    "by everything. The second is a deliberately drifted batch: we applied a seeded shift " +
    "to 17 columns, simulating larger amounts, later timestamps and moved components. " +
    "Evidently caught exactly those 17, a share of 0.567, which is above the 0.5 " +
    "threshold, so it reports dataset drift. I want to be clear that this second batch " +
    "is simulated for demonstration and labelled that way in the repository — we do not " +
    "have real production traffic. Both reports are committed as HTML plus a JSON summary."
  );
}

/* ================================================================== */
/* 9. Retraining + Model Card                                          */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: PAPER };
  slideTitle(s, "Retraining, and the Model Card", "Someyah", false);

  const cm = retrain.current_metrics;
  const cd = retrain.candidate_metrics;
  s.addChart(
    pres.ChartType.bar,
    [
      { name: "Deployed model", labels: ["Recall", "F1", "PR-AUC", "ROC-AUC"], values: [cm.recall, cm.f1_score, cm.pr_auc, cm.roc_auc] },
      { name: "Retrained candidate", labels: ["Recall", "F1", "PR-AUC", "ROC-AUC"], values: [cd.recall, cd.f1_score, cd.pr_auc, cd.roc_auc] },
    ],
    {
      x: 0.75, y: 1.45, w: 6.4, h: 3.15,
      barDir: "col", chartColors: [NAVY_SOFT, GREEN],
      showTitle: true, title: "Candidate vs deployed, on the untouched test split",
      titleFontSize: 12.5, titleColor: INK, titleFontFace: BODY_FONT,
      showValue: false,
      showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: MUTED,
      valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
      catAxisLabelColor: MUTED, valAxisLabelColor: MUTED,
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
      catAxisLabelFontFace: BODY_FONT, valAxisLabelFontFace: BODY_FONT,
      valGridLine: { color: "E2E8F0", size: 1 },
      catGridLine: { style: "none" },
      barGapWidthPct: 70,
    }
  );

  card(s, { x: 0.75, y: 4.78, w: 6.4, h: 1.92 });
  s.addText("Promotion is earned, not automatic", {
    x: 1.05, y: 4.95, w: 5.8, h: 0.32, margin: 0,
    fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: INK,
  });
  s.addText(
    "A candidate replaces the deployed model only if PR-AUC is at least as good and recall " +
    "has not dropped by more than 0.02. This one earned it: PR-AUC 0.7854 → 0.7865, " +
    "recall 0.7263 → 0.7368. If it had failed, the old model would have stayed.",
    {
      x: 1.05, y: 5.3, w: 5.8, h: 1.25, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, color: MUTED, valign: "top",
    }
  );

  card(s, { x: 7.55, y: 1.45, w: 5.05, h: 2.55 });
  s.addText("Retraining needs labels", {
    x: 7.85, y: 1.63, w: 4.45, h: 0.32, margin: 0,
    fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: INK,
  });
  s.addText(
    [
      { text: "Production requests arrive unlabelled — training on our own predictions would teach the model its mistakes.", options: { bullet: true, breakLine: true } },
      { text: "The script refuses a batch with no Class column, or with only one class in it.", options: { bullet: true, breakLine: true } },
      { text: "The test split is never used as training data.", options: { bullet: true, breakLine: false } },
    ],
    {
      x: 7.85, y: 2.0, w: 4.45, h: 1.9, margin: 0,
      fontFace: BODY_FONT, fontSize: 10.5, color: MUTED, paraSpaceAfter: 6, valign: "top",
    }
  );

  card(s, { x: 7.55, y: 4.18, w: 5.05, h: 2.52, fill: NAVY });
  s.addText("MODEL_CARD.md", {
    x: 7.85, y: 4.38, w: 4.45, h: 0.34, margin: 0,
    fontFace: MONO, fontSize: 14, bold: true, color: "9FE8D6", valign: "middle",
  });
  s.addText(
    "Says plainly what the model is for and what it is not for: it misses 26 of 95 " +
    "frauds, its probabilities are not calibrated, the data has no demographic fields so " +
    "we cannot make a fairness claim, and none of it should touch a real payment " +
    "decision without proper validation and governance.",
    {
      x: 7.85, y: 4.78, w: 4.45, h: 1.8, margin: 0,
      fontFace: BODY_FONT, fontSize: 10.5, color: "CADCFC", valign: "top",
    }
  );

  s.addNotes(
    "SOMEYAH (1:00). When drift is detected, retraining does not just run. It needs " +
    "labelled data, and that is a deliberate design decision: the API only ever sees the " +
    "30 features, and whether a transaction turned out to be fraud is confirmed days " +
    "later by a chargeback. If we retrained on our own predictions we would just be " +
    "teaching the model its own mistakes. So the script refuses a batch without a Class " +
    "column, and refuses one that contains only a single class. It trains a candidate on " +
    "the original training split plus the new batch, using the same Random Forest and the " +
    "same parameters — the model family never changes — and then scores both the " +
    "candidate and the currently deployed model on the test split, which is never used " +
    "for training. The candidate is promoted only if PR-AUC holds up and recall has not " +
    "dropped by more than 0.02. In our run it did earn promotion, and the whole decision " +
    "is logged to MLflow. On the right is the Model Card, which states the limits " +
    "honestly: this model misses 26 of 95 frauds, its probabilities are not calibrated, " +
    "and the dataset has no demographic fields, so we cannot claim it is fair."
  );
}

/* ================================================================== */
/* 10. Live demo + close                                               */
/* ================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  slideTitle(s, "Live demo", "Someyah", true);

  const steps = [
    ["1", "Open the public Swagger UI", `${API}/docs`],
    ["2", "Call /health", "200, model_loaded: true"],
    ["3", "POST a real fraud transaction", "expect 0.9696 and label fraud"],
    ["4", "Show the GitHub Actions run", "latest main-branch run — all three jobs green"],
    ["5", "Open the Evidently drift report", "17 of 30 columns flagged"],
  ];
  steps.forEach(([n, head, detail], i) => {
    const y = 1.42 + i * 0.92;
    card(s, { x: 0.75, y, w: 7.5, h: 0.78, fill: NAVY_SOFT });
    numberCircle(s, 1.0, y + 0.16, 0.46, n, i === 3 ? TEAL : RED);
    s.addText(head, {
      x: 1.62, y: y + 0.06, w: 4.0, h: 0.34, margin: 0,
      fontFace: BODY_FONT, fontSize: 13, bold: true, color: PAPER, valign: "middle",
    });
    s.addText(detail, {
      x: 1.62, y: y + 0.38, w: 6.4, h: 0.3, margin: 0,
      fontFace: MONO, fontSize: 10, color: "9FE8D6", valign: "middle",
    });
  });

  card(s, { x: 8.6, y: 1.42, w: 4.05, h: 2.6, fill: NAVY_SOFT });
  s.addText("What we would do next", {
    x: 8.9, y: 1.6, w: 3.45, h: 0.34, margin: 0,
    fontFace: BODY_FONT, fontSize: 13.5, bold: true, color: PAPER,
  });
  s.addText(
    [
      { text: "Set the threshold from the cost of a missed fraud versus a false alarm.", options: { bullet: true, breakLine: true } },
      { text: "Log live requests so drift runs on real traffic, not a simulation.", options: { bullet: true, breakLine: true } },
      { text: "Schedule the drift check instead of running it by hand.", options: { bullet: true, breakLine: false } },
    ],
    {
      x: 8.9, y: 1.98, w: 3.45, h: 1.9, margin: 0,
      fontFace: BODY_FONT, fontSize: 10.5, color: "CADCFC", paraSpaceAfter: 6, valign: "top",
    }
  );

  card(s, { x: 8.6, y: 4.18, w: 4.05, h: 1.82, fill: NAVY_SOFT });
  s.addText("Everything is in the repo", {
    x: 8.9, y: 4.36, w: 3.45, h: 0.32, margin: 0,
    fontFace: BODY_FONT, fontSize: 13, bold: true, color: PAPER,
  });
  s.addText(REPO, {
    x: 8.9, y: 4.72, w: 3.45, h: 1.1, margin: 0,
    fontFace: MONO, fontSize: 9.5, color: "9FE8D6", valign: "top",
  });

  s.addText(
    "Phase 1 trained a model we could reproduce. Phase 2 made it something we could run, " +
    "test, deploy and watch.",
    {
      x: 0.75, y: 6.28, w: 11.9, h: 0.62, margin: 0,
      fontFace: TITLE_FONT, fontSize: 15, italic: true, color: "CADCFC", valign: "middle",
    }
  );

  s.addNotes(
    "SOMEYAH (0:40 plus about 1:30 of demo). Let me show it working. [Demo: open the " +
    "public Swagger page, call health and show model_loaded true, post the fraud " +
    "transaction and show the 0.9696 probability, then the legitimate one at 0.000235, " +
    "then the latest successful GitHub Actions run with all three jobs green, and " +
    "finally the Evidently report with 17 of 30 columns flagged.] To close: Phase 1 gave us a model we could " +
    "reproduce. Phase 2 turned it into a service we can run, test, deploy and watch. If " +
    "we carried this further, the three things we would do are set the decision threshold " +
    "from the actual cost of a missed fraud versus a false alarm, log real requests so " +
    "drift is measured on live traffic instead of a simulation, and put the drift check " +
    "on a schedule. Thank you — we are happy to take questions."
  );
}

const out = path.join(ROOT, "presentation", "MAI201_Phase2_Final_Presentation.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote " + out));
