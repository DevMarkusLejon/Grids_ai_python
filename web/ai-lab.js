(function () {
  const fmtPercent = (value) => {
    if (typeof value !== "number") {
      return "-";
    }
    return `${Math.round(value * 100)}%`;
  };

  const basename = (path) => {
    if (!path) {
      return "-";
    }
    return path.split(/[\\/]/).pop();
  };

  const shortModel = (path) => basename(path).replace(/^value_model_/, "").replace(/\.json$/, "");

  const setText = (id, text) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = text;
    }
  };

  function renderLeaderboard(models) {
    const body = document.getElementById("leaderboard-body");
    body.innerHTML = "";
    models.slice(0, 16).forEach((model) => {
      const row = document.createElement("tr");
      const status = model.is_current_champion ? "Champion" : model.promoted ? "Promoted once" : "Challenger";
      row.innerHTML = `
        <td>${model.rank}</td>
        <td><span class="lab-model-name" title="${model.model}">${shortModel(model.model)}</span></td>
        <td>${model.rating}</td>
        <td>${fmtPercent(model.best_overall_score)}</td>
        <td>${fmtPercent(model.head_to_head_vs_champion)}</td>
        <td>${model.games || 0}</td>
        <td><span class="pill">${status}</span></td>
      `;
      body.appendChild(row);
    });
  }

  function renderReports(reports) {
    const list = document.getElementById("report-list");
    list.innerHTML = "";
    reports.slice(0, 10).forEach((report) => {
      const item = document.createElement("article");
      item.className = "lab-report";
      const promoted = report.promoted ? "Promoted" : "Held";
      item.innerHTML = `
        <div>
          <strong>${shortModel(report.model)}</strong>
          <span>${basename(report.report)}</span>
        </div>
        <div class="lab-report-stats">
          <span>${fmtPercent(report.overall_score)} overall</span>
          <span>${fmtPercent(report.head_to_head_score)} vs champion</span>
          <span>${promoted}</span>
        </div>
      `;
      list.appendChild(item);
    });
  }

  async function init() {
    try {
      const response = await fetch("./assets/model-registry.json", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const registry = await response.json();
      setText("champion-model", shortModel(registry.champion_model));
      setText("model-count", String(registry.summary?.models || 0));
      setText("report-count", String(registry.summary?.reports || 0));
      setText("generated-at", registry.generated_at ? new Date(registry.generated_at).toLocaleString() : "-");
      renderLeaderboard(registry.models || []);
      renderReports(registry.reports || []);
    } catch (error) {
      setText("champion-model", "Registry missing");
      const list = document.getElementById("report-list");
      list.innerHTML = `<article class="lab-report"><strong>Run python -m grids_ai.model_registry</strong><span>${error.message}</span></article>`;
    }
  }

  init();
})();
