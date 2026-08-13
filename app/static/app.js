/* Credit Card Fraud Detection - demo dashboard front end.
 *
 * Every prediction on this page comes from POST /predict on this same origin.
 * Nothing here computes a score locally.
 *
 * EXAMPLES below are input values only: two real transactions taken from the
 * project's held-out test split (presentation/sample_requests/). They pre-fill
 * the form so a demo does not need 30 numbers typed by hand.
 */

const FEATURES = ["Time"].concat(
  Array.from({ length: 28 }, (_, i) => "V" + (i + 1)),
  ["Amount"]
);

const EXAMPLES = {
    "legitimate": {
      "Time": 61290.0,
      "V1": 1.2288211502379,
      "V2": -0.0634077165201056,
      "V3": 0.274145142235826,
      "V4": 0.647465021810117,
      "V5": -0.0481345611508765,
      "V6": 0.372073028593297,
      "V7": -0.22423058741343,
      "V8": 0.0799390492455152,
      "V9": 0.640758817066441,
      "V10": -0.273053702248503,
      "V11": -1.25272793883718,
      "V12": 0.465078770741453,
      "V13": 0.400502115321077,
      "V14": -0.292841860600363,
      "V15": -0.10177401599731,
      "V16": -0.399835897844616,
      "V17": 0.0343356567914817,
      "V18": -0.783550254934187,
      "V19": 0.141344900433949,
      "V20": -0.0965659023514416,
      "V21": -0.129554448055005,
      "V22": -0.0837793282428063,
      "V23": -0.151661473916324,
      "V24": -0.700371597289218,
      "V25": 0.598550164523483,
      "V26": 0.491409070563651,
      "V27": 0.0029892597250263,
      "V28": 0.0017822861144491,
      "Amount": 11.5
    },
    "fraud": {
      "Time": 74159.0,
      "V1": -1.54878809850026,
      "V2": 1.80869795041448,
      "V3": -0.953509033832342,
      "V4": 2.21308539346999,
      "V5": -2.01572779170327,
      "V6": -0.913456844516923,
      "V7": -2.35601298316433,
      "V8": 1.19716896702387,
      "V9": -1.67837405659509,
      "V10": -3.53865023182429,
      "V11": 3.1020899271543,
      "V12": -3.99337305447702,
      "V13": -1.93741062327519,
      "V14": -3.82289410599595,
      "V15": 0.830970110708369,
      "V16": -2.47535885382925,
      "V17": -5.21187516766885,
      "V18": -0.413871678166879,
      "V19": 0.933262164554872,
      "V20": 0.390785963777347,
      "V21": 0.855138263312025,
      "V22": 0.77474482148342,
      "V23": 0.0590371520063436,
      "V24": 0.343199807900813,
      "V25": -0.468937928609185,
      "V26": -0.278337986906642,
      "V27": 0.625922215184372,
      "V28": 0.395573378256676,
      "Amount": 76.94
    }
  };

const HEALTH_TIMEOUT_MS = 20000;
const PREDICT_TIMEOUT_MS = 90000;
const HEALTH_RETRIES = 6;

const $ = (id) => document.getElementById(id);

const els = {
  vGrid: $("v-grid"),
  advanced: $("advanced"),
  btnLegit: $("btn-legit"),
  btnFraud: $("btn-fraud"),
  btnClear: $("btn-clear"),
  btnAnalyze: $("btn-analyze"),
  btnAnalyzeText: $("btn-analyze-text"),
  formError: $("form-error"),
  steps: $("steps"),
  resultEmpty: $("result-empty"),
  resultBody: $("result-body"),
  resultError: $("result-error"),
  errorMessage: $("error-message"),
  verdict: $("verdict"),
  verdictIcon: $("verdict-icon"),
  verdictText: $("verdict-text"),
  probValue: $("prob-value"),
  probFill: $("prob-fill"),
  probThreshold: $("prob-threshold"),
  probNote: $("prob-note"),
  kvClass: $("kv-class"),
  kvLabel: $("kv-label"),
  kvProb: $("kv-prob"),
  kvThreshold: $("kv-threshold"),
  kvIsFraud: $("kv-isfraud"),
  kvVersion: $("kv-version"),
  raw: $("raw"),
  rawJson: $("raw-json"),
  pillApi: $("pill-api"),
  pillApiText: $("pill-api-text"),
  pillModel: $("pill-model"),
  pillModelText: $("pill-model-text"),
  pillVersion: $("pill-version"),
  pillVersionText: $("pill-version-text"),
  pipeModelName: $("pipe-model-name"),
  pipeFeatures: $("pipe-features"),
  pipeThreshold: $("pipe-threshold"),
};

let busy = false;

/* ------------------------------------------------------------ helpers */

function fetchJson(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, Object.assign({ signal: controller.signal }, options || {}))
    .then(async (response) => {
      const text = await response.text();
      let body = null;
      try {
        body = text ? JSON.parse(text) : null;
      } catch (err) {
        body = null;
      }
      return { ok: response.ok, status: response.status, body: body, raw: text };
    })
    .finally(() => clearTimeout(timer));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setPill(el, textEl, state, text) {
  el.classList.remove("pill-ok", "pill-bad", "pill-wait", "pill-idle");
  el.classList.add("pill-" + state);
  textEl.textContent = text;
}

/* ------------------------------------------------------- build inputs */

FEATURES.filter((name) => name !== "Time" && name !== "Amount").forEach((name) => {
  const cell = document.createElement("div");
  cell.className = "v-cell";

  const label = document.createElement("label");
  label.setAttribute("for", name);
  label.textContent = name;

  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.id = name;
  input.autocomplete = "off";
  input.placeholder = "0.0";

  cell.appendChild(label);
  cell.appendChild(input);
  els.vGrid.appendChild(cell);
});

/* ------------------------------------------------------- form actions */

function fillForm(values) {
  FEATURES.forEach((name) => {
    const input = $(name);
    if (input) {
      input.value = values[name];
      input.classList.remove("invalid");
    }
  });
  hideFormError();
}

function clearForm() {
  FEATURES.forEach((name) => {
    const input = $(name);
    if (input) {
      input.value = "";
      input.classList.remove("invalid");
    }
  });
  hideFormError();
  resetSteps();
  els.resultBody.hidden = true;
  els.resultError.hidden = true;
  els.resultEmpty.hidden = false;
  els.rawJson.textContent = "No request has been sent yet.";
}

function showFormError(message) {
  els.formError.textContent = message;
  els.formError.hidden = false;
}

function hideFormError() {
  els.formError.hidden = true;
  els.formError.textContent = "";
}

function readForm() {
  const payload = {};
  const missing = [];
  FEATURES.forEach((name) => {
    const input = $(name);
    const value = input.value.trim();
    if (value === "" || Number.isNaN(Number(value))) {
      missing.push(name);
      input.classList.add("invalid");
    } else {
      input.classList.remove("invalid");
      payload[name] = Number(value);
    }
  });
  return { payload: payload, missing: missing };
}

/* ------------------------------------------------------------- steps */

function resetSteps() {
  els.steps.querySelectorAll(".step").forEach((step) => {
    step.classList.remove("active", "done", "failed");
  });
}

function markStep(number, state) {
  const step = els.steps.querySelector('[data-step="' + number + '"]');
  if (!step) return;
  step.classList.remove("active", "done", "failed");
  if (state) step.classList.add(state);
}

function markRange(from, to, state) {
  for (let i = from; i <= to; i += 1) markStep(i, state);
}

/* ------------------------------------------------------------ health */

async function refreshHealth(attempt) {
  attempt = attempt || 1;
  try {
    const result = await fetchJson("/health", { headers: { Accept: "application/json" } }, HEALTH_TIMEOUT_MS);
    const body = result.body || {};

    if (result.ok && body.model_loaded) {
      setPill(els.pillApi, els.pillApiText, "ok", "API online");
      setPill(els.pillModel, els.pillModelText, "ok", "Model loaded");
    } else {
      setPill(els.pillApi, els.pillApiText, result.status ? "ok" : "bad", result.status ? "API online" : "API unreachable");
      setPill(els.pillModel, els.pillModelText, "bad", "Model not loaded");
    }
    if (body.version) {
      setPill(els.pillVersion, els.pillVersionText, "idle", "Version " + body.version);
    }
    return;
  } catch (err) {
    if (attempt < HEALTH_RETRIES) {
      setPill(els.pillApi, els.pillApiText, "wait", "Connecting to model service...");
      setPill(els.pillModel, els.pillModelText, "wait", "Waiting for model");
      await sleep(3000);
      return refreshHealth(attempt + 1);
    }
    setPill(els.pillApi, els.pillApiText, "bad", "API unreachable");
    setPill(els.pillModel, els.pillModelText, "bad", "Model status unknown");
  }
}

async function refreshModelInfo() {
  try {
    const result = await fetchJson("/model-info", { headers: { Accept: "application/json" } }, HEALTH_TIMEOUT_MS);
    if (!result.ok || !result.body) return;
    const info = result.body;
    if (info.model_type) els.pipeModelName.textContent = info.model_type;
    if (info.expected_feature_count) {
      els.pipeFeatures.textContent = info.expected_feature_count + " features";
    }
    if (typeof info.decision_threshold === "number") {
      els.pipeThreshold.textContent = info.decision_threshold.toFixed(2);
    }
  } catch (err) {
    /* the pipeline diagram keeps its neutral defaults */
  }
}

/* ----------------------------------------------------------- predict */

function describeFailure(result) {
  if (result.status === 422 && result.body && Array.isArray(result.body.detail)) {
    const fields = result.body.detail
      .map((item) => (Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null))
      .filter(Boolean);
    const unique = Array.from(new Set(fields));
    return (
      "The API rejected the transaction (HTTP 422). Check these fields: " +
      (unique.length ? unique.join(", ") : "request body") +
      "."
    );
  }
  if (result.status === 503) {
    return "The service is running but the model is not loaded, so it cannot score (HTTP 503).";
  }
  if (result.body && typeof result.body.detail === "string") {
    return "HTTP " + result.status + ": " + result.body.detail;
  }
  return "The API returned HTTP " + result.status + ".";
}

function renderResult(data) {
  const isFraud = Boolean(data.is_fraud);
  const probability = Number(data.fraud_probability);
  const threshold = Number(data.decision_threshold);

  els.resultEmpty.hidden = true;
  els.resultError.hidden = true;
  els.resultBody.hidden = false;

  els.verdict.classList.remove("verdict-legit", "verdict-fraud");
  els.verdict.classList.add(isFraud ? "verdict-fraud" : "verdict-legit");
  els.verdictIcon.textContent = isFraud ? "!" : "\u2713";
  els.verdictText.textContent = isFraud ? "FRAUD DETECTED" : "LEGITIMATE TRANSACTION";

  const percent = probability * 100;
  els.probValue.textContent = percent.toFixed(2) + "%";
  els.probFill.classList.toggle("is-fraud", isFraud);
  els.probFill.style.width = Math.min(100, Math.max(0, percent)) + "%";
  els.probThreshold.style.left = Math.min(100, Math.max(0, threshold * 100)) + "%";

  const thresholdPercent = (threshold * 100).toFixed(0);
  els.probNote.innerHTML = isFraud
    ? "Probability <strong>" + percent.toFixed(2) + "%</strong> is above the decision threshold of <strong>" +
      thresholdPercent + "%</strong>, so the transaction is flagged as fraud."
    : "Probability <strong>" + percent.toFixed(2) + "%</strong> stays below the decision threshold of <strong>" +
      thresholdPercent + "%</strong>, so the transaction is accepted.";

  els.kvClass.textContent = data.predicted_class;
  els.kvLabel.textContent = data.label;
  els.kvProb.textContent = data.fraud_probability;
  els.kvThreshold.textContent = data.decision_threshold;
  els.kvIsFraud.textContent = String(data.is_fraud);
  els.kvVersion.textContent = data.model_version;
}

function renderError(message) {
  els.resultEmpty.hidden = true;
  els.resultBody.hidden = true;
  els.resultError.hidden = false;
  els.errorMessage.textContent = message;
}

function setBusy(state) {
  busy = state;
  els.btnAnalyze.disabled = state;
  els.btnLegit.disabled = state;
  els.btnFraud.disabled = state;
  els.btnClear.disabled = state;
  els.btnAnalyzeText.textContent = state ? "Analyzing..." : "Analyze transaction";
}

async function analyze() {
  if (busy) return;

  resetSteps();
  hideFormError();

  markStep(1, "active");
  const form = readForm();
  await sleep(180);

  if (form.missing.length) {
    markStep(1, "failed");
    els.advanced.open = true;
    showFormError(
      "Missing or invalid value for " +
        form.missing.slice(0, 6).join(", ") +
        (form.missing.length > 6 ? " and " + (form.missing.length - 6) + " more" : "") +
        ". Load an example to fill every field."
    );
    return;
  }
  markStep(1, "done");

  setBusy(true);
  markStep(2, "active");
  await sleep(180);
  markStep(2, "done");

  markStep(3, "active");
  await sleep(140);
  markStep(3, "done");
  markStep(4, "active");

  try {
    const result = await fetchJson(
      "/predict",
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(form.payload),
      },
      PREDICT_TIMEOUT_MS
    );

    els.rawJson.textContent = result.body
      ? JSON.stringify(result.body, null, 2)
      : result.raw || "(empty response)";

    if (!result.ok) {
      markStep(4, "failed");
      renderError(describeFailure(result));
      els.raw.open = true;
      return;
    }

    markRange(4, 7, "done");
    renderResult(result.body);
    refreshHealth();
  } catch (err) {
    markStep(4, "failed");
    const aborted = err && err.name === "AbortError";
    renderError(
      aborted
        ? "The request timed out. The free hosting plan puts the service to sleep when it is idle, so the first call after a quiet period can take a while. Please try again."
        : "Could not reach the prediction service. Check the network connection and try again."
    );
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------- wiring */

els.btnLegit.addEventListener("click", () => fillForm(EXAMPLES.legitimate));
els.btnFraud.addEventListener("click", () => fillForm(EXAMPLES.fraud));
els.btnClear.addEventListener("click", clearForm);
els.btnAnalyze.addEventListener("click", analyze);

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) analyze();
});

refreshHealth();
refreshModelInfo();
