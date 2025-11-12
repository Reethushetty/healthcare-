let lastRequestPayload = null;
let riskChart = null;
let staffChart = null;

document.addEventListener("DOMContentLoaded", () => {
  const problemSelect = document.getElementById("problem_type");
  const diabetesFields = document.getElementById("diabetes-fields");
  const heartFields = document.getElementById("heart-fields");
  const form = document.getElementById("prediction-form");
  const resultCard = document.getElementById("result-card");
  const pdfBtn = document.getElementById("download-pdf");
  const simButton = document.getElementById("run-simulation");
  const simOutput = document.getElementById("simulation-output");

  // Toggle Diabetes/Heart specific fields
  problemSelect.addEventListener("change", () => {
    if (problemSelect.value === "Diabetes") {
      diabetesFields.classList.remove("hidden");
      heartFields.classList.add("hidden");
    } else {
      heartFields.classList.remove("hidden");
      diabetesFields.classList.add("hidden");
    }
  });

  // === Predict Readmission ===
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = buildPayload();
    lastRequestPayload = payload;

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "Prediction failed");
        return;
      }

      renderPredictionResult(data);
      resultCard.classList.remove("hidden");
      simOutput.classList.add("hidden"); // Hide old simulation result
    } catch (err) {
      console.error(err);
      alert("Server error while predicting.");
    }
  });

  // === Run Staffing Simulation ===
  simButton.addEventListener("click", async () => {
    if (!lastRequestPayload) {
      alert("Run a prediction first.");
      return;
    }

    const simDate = document.getElementById("simulation_date").value;
    const unit = document.getElementById("hospital_unit").value;

    if (!simDate || !unit) {
      alert("Please select both Simulation Date and Hospital Unit.");
      return;
    }

    lastRequestPayload["Simulation Date"] = simDate;
    lastRequestPayload["Hospital Unit"] = unit;

    try {
      const res = await fetch("/api/simulate_staffing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lastRequestPayload),
      });

      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "Simulation failed");
        return;
      }

      const s = data.staffing;
      simOutput.classList.remove("hidden");
      simOutput.textContent =
        `Expected readmissions (on ${data.simulation_date}, ${data.hospital_unit}): ` +
        `${s.expected_readmissions}. Beds: ${s.suggested_beds}, Nurses: ${s.suggested_nurses}, Doctors: ${s.suggested_doctors}.`;

      drawStaffChart(s);
    } catch (err) {
      console.error(err);
      alert("Error running staffing simulation.");
    }
  });

  // === PDF Download ===
  pdfBtn.addEventListener("click", async () => {
    if (!lastRequestPayload) {
      alert("Run a prediction first.");
      return;
    }

    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lastRequestPayload),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(err.error || "Failed to generate report");
        return;
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "readmission_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert("Error generating PDF.");
    }
  });
});

// ===== Build Payload =====
function buildPayload() {
  const problem_type = document.getElementById("problem_type").value;

  const payload = {
    "Patient ID": document.getElementById("patient_id").value || "N/A",
    "Patient Name": document.getElementById("patient_name").value || "N/A",
    "Admission Date": document.getElementById("admission_date").value || "N/A",
    "Discharge Date": document.getElementById("discharge_date").value || "N/A",
    "Problem Type": problem_type,
    "Age": Number(document.getElementById("age").value),
    "Sex": document.getElementById("sex").value,
    "Weight": Number(document.getElementById("weight").value),
    "Blood Pressure": document.getElementById("bp").value,
    "Cholesterol": Number(document.getElementById("cholesterol").value),
    "Insulin": document.getElementById("insulin").value,
    "Diabetics": document.getElementById("diabetics").value,
    "air_quality_index": Number(document.getElementById("aqi").value),
    "social_event_count": Number(document.getElementById("events").value),
  };

  if (problem_type === "Diabetes") {
    payload["Hemoglobin (g/dL)"] = Number(document.getElementById("hb").value || 13.5);
    payload["WBC Count (10^9/L)"] = Number(document.getElementById("wbc").value || 7.0);
    payload["Platelet Count (10^9/L)"] = Number(document.getElementById("plt_count").value || 250);
    payload["Urine Protein (mg/dL)"] = Number(document.getElementById("urine_protein").value || 10);
    payload["Urine Glucose (mg/dL)"] = Number(document.getElementById("urine_glucose").value || 5);
  } else {
    payload["ECG Result"] = document.getElementById("ecg").value;
    payload["Pulse Rate (bpm)"] = Number(document.getElementById("pulse").value || 72);
  }

  return payload;
}

// ===== Render Prediction =====
function renderPredictionResult(data) {
  document.getElementById("out-disease").textContent = data.disease_type || "-";
  document.getElementById("out-prediction").textContent = data.prediction || "-";
  document.getElementById("out-score").textContent =
    data.readmission_probability !== undefined
      ? data.readmission_probability.toFixed(4)
      : "-";
  document.getElementById("out-risk-label").textContent = data.risk_label || "-";

  if (data.followup_plan) {
    const f = data.followup_plan;
    document.getElementById("out-followup").textContent =
      `Channel: ${f.channel}. Schedule: ${f.schedule.join(", ")}. ${f.note}`;
  }

  drawRiskChart(data.readmission_probability || 0);
}

// ===== Charts =====
function drawRiskChart(prob) {
  const ctx = document.getElementById("riskChart").getContext("2d");
  if (riskChart) riskChart.destroy();

  riskChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Readmission Risk"],
      datasets: [{
        label: "Probability",
        data: [prob],
        backgroundColor: "rgba(0, 123, 255, 0.5)"
      }]
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1 } }
    }
  });
}

function drawStaffChart(staffing) {
  const ctx = document.getElementById("staffChart").getContext("2d");
  if (staffChart) staffChart.destroy();
  if (!staffing) return;

  staffChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Beds", "Nurses", "Doctors"],
      datasets: [{
        label: "Suggested",
        data: [
          staffing.suggested_beds,
          staffing.suggested_nurses,
          staffing.suggested_doctors
        ],
        backgroundColor: "rgba(0, 123, 255, 0.5)"
      }]
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true } }
    }
  });
}
