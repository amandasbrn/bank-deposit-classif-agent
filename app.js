const form = document.getElementById("prediction-form");
const jobSelect = document.getElementById("job");
const educationSelect = document.getElementById("education");
const statusNode = document.getElementById("status");
const resultsNode = document.getElementById("results");
const submitButton = document.getElementById("submit-button");
const apiBase = document.body.dataset.apiBase || "";

let metadata = null;

function setStatus(message, type = "info") {
  statusNode.textContent = message;
  statusNode.dataset.type = type;
}

function setRangeText(id, range) {
  document.getElementById(id).textContent = `Observed range: ${range.min} to ${range.max}`;
}

function populateJobs() {
  jobSelect.innerHTML = "";
  metadata.jobs.forEach((job) => {
    const option = document.createElement("option");
    option.value = job;
    option.textContent = job;
    jobSelect.appendChild(option);
  });
  populateEducation();
}

function populateEducation() {
  const selectedJob = jobSelect.value;
  const educationOptions = metadata.educationByJob[selectedJob] || [];
  educationSelect.innerHTML = "";
  educationOptions.forEach((education) => {
    const option = document.createElement("option");
    option.value = education;
    option.textContent = education;
    educationSelect.appendChild(option);
  });
}

async function loadMetadata() {
  setStatus("Loading model metadata...");
  const response = await fetch(`${apiBase}/api/metadata`);
  if (!response.ok) {
    throw new Error("Failed to load metadata");
  }
  metadata = await response.json();
  populateJobs();
  setRangeText("duration-range", metadata.ranges.duration);
  setRangeText("balance-range", metadata.ranges.balance);
  setRangeText("previous-range", metadata.ranges.previous);
  setStatus("Ready");
}

function renderResults(data) {
  document.getElementById("prediction-label").textContent = data.predictionLabel;
  document.getElementById("prediction-probability").textContent = `Probability: ${data.probability.toFixed(2)}`;
  document.getElementById("decision-action").textContent = data.agentDecision.action;
  document.getElementById("decision-priority").textContent = data.agentDecision.priority;
  document.getElementById("decision-window").textContent = data.agentDecision.follow_up_window || "None";
  document.getElementById("recommendations").textContent = data.recommendations;
  resultsNode.classList.remove("hidden");
}

jobSelect.addEventListener("change", populateEducation);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  setStatus("Running prediction...");

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    const response = await fetch(`${apiBase}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }
    renderResults(data);
    setStatus("Prediction complete", "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

loadMetadata().catch((error) => {
  setStatus(error.message, "error");
});
