/**
 * SIGNAL Model Comparison Matrix
 * High-density frontier model matrix with sorting and capability categorization.
 */

class ModelComparisonView {
  constructor(tableId) {
    this.table = document.getElementById(tableId);
    this.body = document.getElementById("matrixBody");
    this.currentData = [];
    this.filteredData = [];
    this.sortCol = "swe_bench_verified";
    this.sortAsc = false;
  }

  render(data) {
    if (!data || !data.models) return;
    this.currentData = data.models;
    this.filteredData = [...this.currentData];

    // Dynamically update filter button counts based on queried live models
    const totalCount = this.currentData.length;
    const reasoningCount = this.currentData.filter(m => (m.architecture && (m.architecture.includes("Reasoning") || m.architecture.includes("Search") || m.architecture.includes("Hybrid")))).length;
    const openCount = this.currentData.filter(m => m.is_open_weights === true).length;
    const codeCount = this.currentData.filter(m => (m.model_name && (m.model_name.includes("Coder") || m.model_name.includes("Sonnet") || m.model_name.includes("Codestral")))).length;

    const btnAll = document.getElementById("btnFilterAll");
    if (btnAll) btnAll.innerText = `All (${totalCount})`;
    const btnReasoning = document.getElementById("btnFilterReasoning");
    if (btnReasoning) btnReasoning.innerText = `Reasoning (${reasoningCount})`;
    const btnOpen = document.getElementById("btnFilterOpen");
    if (btnOpen) btnOpen.innerText = `Open Weights (${openCount})`;
    const btnCode = document.getElementById("btnFilterCode");
    if (btnCode) btnCode.innerText = `Coding (${codeCount})`;

    this.applySortAndRender();
  }

  filter(category) {
    if (category === "all") {
      this.filteredData = [...this.currentData];
    } else if (category === "reasoning") {
      this.filteredData = this.currentData.filter(m => (m.architecture && (m.architecture.includes("Reasoning") || m.architecture.includes("Search") || m.architecture.includes("Hybrid"))));
    } else if (category === "open") {
      this.filteredData = this.currentData.filter(m => m.is_open_weights === true);
    } else if (category === "code") {
      this.filteredData = this.currentData.filter(m => (m.model_name && (m.model_name.includes("Coder") || m.model_name.includes("Sonnet") || m.model_name.includes("Codestral"))));
    }
    this.applySortAndRender();
  }

  sort(column) {
    if (this.sortCol === column) {
      this.sortAsc = !this.sortAsc;
    } else {
      this.sortCol = column;
      this.sortAsc = false;
    }
    this.applySortAndRender();
  }

  applySortAndRender() {
    this.filteredData.sort((a, b) => {
      let valA = a[this.sortCol];
      let valB = b[this.sortCol];
      if (valA === null || valA === undefined) valA = this.sortAsc ? 999999 : -999999;
      if (valB === null || valB === undefined) valB = this.sortAsc ? 999999 : -999999;

      if (typeof valA === "string") {
        return this.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return this.sortAsc ? valA - valB : valB - valA;
    });

    if (!this.body) return;

    this.body.innerHTML = this.filteredData.map(m => {
      const openTag = m.is_open_weights
        ? `<span style="background:rgba(16,185,129,0.15); color:var(--accent-emerald); font-size:9px; padding:1px 4px; border-radius:3px; font-weight:700;">OPEN</span>`
        : `<span style="background:rgba(148,163,184,0.15); color:var(--text-muted); font-size:9px; padding:1px 4px; border-radius:3px;">PROPRIETARY</span>`;

      const sweVal = m.swe_bench_verified ? `<strong style="color:var(--accent-cyan);">${m.swe_bench_verified.toFixed(1)}%</strong>` : `<span style="color:var(--text-muted);">--</span>`;
      const gpqaVal = m.gpqa_diamond ? `${m.gpqa_diamond.toFixed(1)}%` : `<span style="color:var(--text-muted);">--</span>`;
      const mathVal = m.math_500 ? `<strong style="color:var(--accent-emerald);">${m.math_500.toFixed(1)}%</strong>` : `<span style="color:var(--text-muted);">--</span>`;
      
      const costVal = m.pricing_output_per_m !== undefined && m.pricing_output_per_m !== null
        ? `<span style="font-family:var(--font-mono); color:${m.pricing_output_per_m < 3.0 ? 'var(--accent-emerald)' : 'var(--accent-amber)'}; font-weight:700;">$${m.pricing_output_per_m.toFixed(2)}</span>`
        : `--`;

      return `
        <tr>
          <td>
            <div style="font-weight:700; color:#fff; display:flex; align-items:center; gap:6px;">
              ${m.model_name || m.name}
              ${openTag}
            </div>
            <div style="font-size:10px; color:var(--text-muted);">${m.provider}</div>
          </td>
          <td>
            <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-secondary);">${m.architecture || "Dense Transformer"}</div>
          </td>
          <td style="font-family:var(--font-mono); font-size:12px;">${sweVal}</td>
          <td style="font-family:var(--font-mono); font-size:12px;">${gpqaVal}</td>
          <td style="font-family:var(--font-mono); font-size:12px;">${mathVal}</td>
          <td style="font-family:var(--font-mono); font-size:12px;">${costVal}</td>
        </tr>
      `;
    }).join("");
  }
}

window.ModelComparisonView = ModelComparisonView;
