/* ==========================================================================
   main.js - front-end logic for the Car Price Predictor
   1. Filters the "Car Model" dropdown by the chosen company
   2. Validates the form before sending
   3. Calls POST /predict and shows the result without reloading the page
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {

  // ---- elements --------------------------------------------------------
  const companySel = document.getElementById("company");
  const nameSel = document.getElementById("name");
  const yearSel = document.getElementById("year");
  const kmsInput = document.getElementById("kms_driven");
  const fuelSel = document.getElementById("fuel_type");
  const form = document.getElementById("predict-form");
  const btn = document.getElementById("predict-btn");
  const btnText = btn.querySelector(".btn-text");
  const btnSpinner = btn.querySelector(".btn-spinner");
  const resultBox = document.getElementById("result-box");
  const resultPrice = document.getElementById("result-price");
  const resultNote = document.getElementById("result-note");
  const errorBox = document.getElementById("error-box");
  const errorText = document.getElementById("error-text");

  // all 242 model names, injected by Flask as JSON
  const ALL_MODELS = JSON.parse(document.getElementById("meta-models").textContent);

  // ---- 1. filter model dropdown by company ------------------------------
  companySel.addEventListener("change", () => {
    nameSel.innerHTML = '<option value="" disabled selected>Loading models…</option>';
    nameSel.disabled = true;
    clearError();

    // small timeout so the "Loading…" state is visible for a moment
    setTimeout(() => {
      // case-insensitive: company stored as "Bmw" but models as "BMW 3 Series"
      const prefix = companySel.value.toLowerCase() + " ";
      const models = ALL_MODELS.filter(m => m.toLowerCase().startsWith(prefix));
      nameSel.innerHTML =
        '<option value="" disabled selected>Select model</option>' +
        models.map(m => `<option value="${m}">${m}</option>`).join("");
      // if no model of that brand, allow any model name (edge case)
      if (models.length === 0) {
        nameSel.innerHTML =
          '<option value="" disabled selected>No models found</option>';
      }
      nameSel.disabled = false;
    }, 150);
  });

  // ---- helpers -----------------------------------------------------------
  function setFieldError(field, message) {
    const group = field.closest(".form-group");
    group.classList.add("invalid");
    group.querySelector(".error-msg").textContent = message;
  }

  function clearFieldErrors() {
    document.querySelectorAll(".form-group").forEach(g =>
      g.classList.remove("invalid"));
  }

  function showError(message) {
    errorText.textContent = message;
    errorBox.classList.remove("hidden");
    resultBox.classList.add("hidden");
  }

  function clearError() {
    errorBox.classList.add("hidden");
  }

  function formatIndian(amount) {
    // 425000 -> "Rs 4,25,000" (Indian numbering)
    const s = String(amount);
    let last3 = s.slice(-3);
    let rest = s.slice(0, -3);
    if (rest) last3 = "," + last3;
    return "Rs " + rest.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + last3;
  }

  // ---- 2. validate + submit ---------------------------------------------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFieldErrors();
    clearError();

    // client-side validation (server validates again - never trust only JS)
    let ok = true;
    if (!companySel.value) { setFieldError(companySel, "Please select a company."); ok = false; }
    if (!nameSel.value) { setFieldError(nameSel, "Please select a car model."); ok = false; }
    if (!yearSel.value) { setFieldError(yearSel, "Please select the year."); ok = false; }
    const kms = Number(kmsInput.value);
    if (!kmsInput.value || isNaN(kms)) {
      setFieldError(kmsInput, "Please enter kilometres driven."); ok = false;
    } else if (kms < 100 || kms > 250000) {
      setFieldError(kmsInput, "Kilometres must be between 100 and 2,50,000."); ok = false;
    }
    if (!fuelSel.value) { setFieldError(fuelSel, "Please select fuel type."); ok = false; }
    if (!ok) return;

    // send to the Flask backend
    btn.disabled = true;
    btnText.textContent = "Predicting…";
    btnSpinner.classList.remove("hidden");

    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          company: companySel.value,
          name: nameSel.value,
          year: yearSel.value,
          kms_driven: kmsInput.value,
          fuel_type: fuelSel.value,
        }),
      });
      const data = await resp.json();

      if (data.success) {
        resultPrice.textContent = formatIndian(Math.round(data.price));
        resultNote.textContent =
          `${nameSel.value} · ${yearSel.value} · ${Number(kms).toLocaleString("en-IN")} km · ${fuelSel.value}`;
        resultBox.classList.remove("hidden");
        resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        showError(data.error || "Prediction failed. Please check your inputs.");
      }
    } catch (err) {
      showError("Could not reach the server. Is the Flask app still running?");
    } finally {
      btn.disabled = false;
      btnText.textContent = "Predict Price";
      btnSpinner.classList.add("hidden");
    }
  });
});
