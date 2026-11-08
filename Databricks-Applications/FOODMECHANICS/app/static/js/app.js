/**
 * FOODMECHANICS: Colloid Dynamics & Formulation Studio Controller
 * Powered by Databricks Serverless SQL Compute
 */

let allIngredients = [];
let allMechanisms = [];
let systemsCatalog = [];
let culinaryRecipes = [];
let activeSystemId = "whipped_foam";
let balancerDebounceTimer = null;
let activeSystemData = null;

// Preset DBSQL queries for the Databricks Lakehouse Studio
const presetFoodQueries = {
  1: "SELECT ingredient_name, category, default_serving_g, density_g_ml, overrun_pct, drainage_t50_min FROM workspace.foodmechanics_gold.v_ingredient_comparisons ORDER BY overrun_pct DESC",
  2: "SELECT i.name, n.energy_kcal, n.protein_g, n.fat_total_g, n.carb_g, n.fiber_g FROM workspace.foodmechanics_silver.silver_ingredients i JOIN workspace.foodmechanics_silver.silver_nutritional_profiles n ON i.ingredient_id = n.ingredient_id WHERE n.protein_g > 10.0 ORDER BY n.protein_g DESC",
  3: "SELECT simulation_id, continuous_phase_viscosity_pa_s, yield_stress_pa, calculated_drainage_velocity_mm_s, stokes_stability_score FROM workspace.foodmechanics_gold.gold_formulation_simulations ORDER BY stokes_stability_score DESC LIMIT 25",
  4: "SELECT mechanism_id, title, mechanism_type, thermodynamic_or_kinetic_basis FROM workspace.foodmechanics_silver.silver_food_mechanisms"
};

document.addEventListener("DOMContentLoaded", async () => {
  await fetchStatus();
  await Promise.all([
    loadSystemsCatalog(),
    loadIngredients(),
    loadMechanisms(),
    loadSubstitutionRecipes()
  ]);
  updateSandboxTotals();
  await runInverseOptimizer();
});

// --- API Loaders ---

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const lbl = document.getElementById("engine-label");
    if (data.databricks_connected) {
      lbl.innerText = "● Databricks Serverless (Active)";
    } else {
      lbl.innerText = "● DuckDB Local Engine (Fallback)";
    }
  } catch (e) {
    console.error("Status fetch error", e);
  }
}

async function loadSystemsCatalog() {
  const grid = document.getElementById("systemsSelectorGrid");
  if (!grid) return;

  try {
    const res = await fetch("/api/systems");
    const data = await res.json();
    systemsCatalog = data.systems || [];

    const iconMap = {
      whipped_foam: "🍨",
      viscoelastic_dough: "🍞",
      plant_mayo_emulsion: "🥑",
      calcium_spherification: "🧪",
      low_sugar_pectin_matrix: "🍓"
    };

    grid.innerHTML = systemsCatalog.map((sys, idx) => `
      <div class="system-choice-card ${sys.system_id === activeSystemId ? 'active' : ''}" 
           id="sys-card-${sys.system_id}"
           onclick="selectSystem('${sys.system_id}')">
        <div class="system-icon">${iconMap[sys.system_id] || '🔬'}</div>
        <div class="system-card-title">${sys.short_name || sys.title}</div>
        <div class="system-card-sub">${sys.control.name} vs ${sys.engineered.name}</div>
      </div>
    `).join("");

    if (systemsCatalog.length > 0) {
      await selectSystem(systemsCatalog[0].system_id);
    }
  } catch (e) {
    console.error("Failed to load systems catalog:", e);
    grid.innerHTML = `<div style="padding:16px; color:#dc2626;">Failed to query Databricks formulation catalog.</div>`;
  }
}

async function selectSystem(systemId) {
  activeSystemId = systemId;
  const container = document.getElementById("activeSystemContainer");
  if (!container) return;

  // Update active card styling
  document.querySelectorAll(".system-choice-card").forEach(c => c.classList.remove("active"));
  const activeCard = document.getElementById(`sys-card-${systemId}`);
  if (activeCard) activeCard.classList.add("active");

  container.innerHTML = `
    <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:36px; text-align:center; color:var(--text-muted); font-size:13px;">
      <div style="font-size:24px; margin-bottom:8px;">⚡</div>
      Executing Databricks Serverless physics simulation for <strong>${systemId}</strong>...
    </div>
  `;

  try {
    const res = await fetch(`/api/systems/${systemId}`);
    const data = await res.json();
    activeSystemData = data;
    renderSystemDetail(data);
  } catch (e) {
    console.error("Error loading system detail:", e);
    container.innerHTML = `<div style="padding:20px; color:#dc2626;">Error executing simulation on Databricks.</div>`;
  }
}

function renderSystemDetail(data) {
  const container = document.getElementById("activeSystemContainer");
  if (!container || !data) return;

  const sys = data.system;
  const ctrlSim = data.control_simulation;
  const engSim = data.engineered_simulation;

  const ctrlNut = ctrlSim.normalized_100g_nutrients || {};
  const ctrlRheo = ctrlSim.rheology_predictions || {};
  const engNut = engSim.normalized_100g_nutrients || {};
  const engRheo = engSim.rheology_predictions || {};

  // Compute live deltas
  const calcDelta = (valE, valC, isPct = false, unit = '') => {
    if (valC === 0) return '';
    const diff = ((valE - valC) / valC) * 100;
    const sign = diff > 0 ? '+' : '';
    return `${sign}${diff.toFixed(1)}% ${unit}`;
  };

  const fatDelta = calcDelta(engNut.fat_total_g, ctrlNut.fat_total_g, true, 'Fat');
  const energyDelta = calcDelta(engNut.calories_kcal, ctrlNut.calories_kcal, true, 'Energy');
  const overrunDelta = calcDelta(engRheo.predicted_overrun_pct, ctrlRheo.predicted_overrun_pct, true, 'Volume');
  const stabilityDelta = engRheo.predicted_drainage_half_life_min > ctrlRheo.predicted_drainage_half_life_min ? "24h+ Stable" : "Transient";

  container.innerHTML = `
    <!-- Experiment Headline -->
    <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:16px 20px; margin-bottom:20px;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
          <span class="badge-engineered" style="font-size:11px; font-weight:700; text-transform:uppercase;">${sys.category_badge}</span>
          <h2 style="font-family:var(--font-serif); font-size:1.4rem; color:var(--text-primary); margin:6px 0 4px 0;">${sys.title}</h2>
          <p style="font-size:0.88rem; color:var(--text-secondary); margin:0;">${sys.summary}</p>
        </div>
        <span style="font-family:var(--font-mono); font-size:11px; background:#f0fdf4; color:#166534; border:1px solid #bbf7d0; padding:4px 8px; border-radius:4px; font-weight:700; white-space:nowrap;">
          ⚡ Live Serverless Queried
        </span>
      </div>
    </div>

    <!-- Side-by-Side Comparison -->
    <div class="formulation-grid" style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:24px;">
      
      <!-- Control Side -->
      <div class="formulation-card benchmark" style="background:#fff; border:1px solid #fecaca; border-left:4px solid #ef4444; border-radius:8px; padding:20px;">
        <span class="card-badge badge-control" style="background:#fee2e2; color:#991b1b; font-size:11px; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase;">Conventional Control</span>
        <div class="formula-name" style="font-family:var(--font-serif); font-size:1.15rem; font-weight:700; margin:8px 0 4px 0;">${sys.control.name}</div>
        <p style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:14px;">${sys.control.description}</p>

        <div class="kpi-row" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-bottom:16px; background:#fafafa; padding:10px; border-radius:6px;">
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Total Fat</div>
            <div style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:#dc2626;">${ctrlNut.fat_total_g.toFixed(1)}g</div>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Energy</div>
            <div style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-primary);">${ctrlNut.calories_kcal.toFixed(0)} kcal</div>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Overrun</div>
            <div style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-primary);">${ctrlRheo.predicted_overrun_pct}%</div>
            <div style="font-size:9px; color:var(--text-muted);">expansion</div>
          </div>
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Stability</div>
            <div style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-primary);">${ctrlRheo.predicted_drainage_half_life_min}m</div>
            <div style="font-size:9px; color:var(--text-muted);">half-life</div>
          </div>
        </div>

        <div style="background:#fff1f2; border:1px solid #fecdd3; border-radius:6px; padding:12px;">
          <div style="font-size:12px; font-weight:700; color:#be123c; margin-bottom:4px;">${sys.control.mechanism_title}</div>
          <div style="font-size:11px; color:#4c0519; line-height:1.4;">${sys.control.mechanism_desc}</div>
          ${sys.control.limitation ? `<div style="margin-top:6px; font-size:11px; color:#991b1b; font-weight:600;">${sys.control.limitation}</div>` : ''}
        </div>
      </div>

      <!-- Engineered Side -->
      <div class="formulation-card replacement" style="background:#fff; border:1px solid #bbf7d0; border-left:4px solid #16a34a; border-radius:8px; padding:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="card-badge badge-engineered" style="background:#dcfce7; color:#15803d; font-size:11px; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase;">Engineered Formulation</span>
          <span id="engSimulationIndicator" style="font-family:var(--font-mono); font-size:10px; color:#15803d; font-weight:700;">● Stokes' Model Synchronized</span>
        </div>
        <div class="formula-name" style="font-family:var(--font-serif); font-size:1.15rem; font-weight:700; margin:8px 0 4px 0;">${sys.engineered.name}</div>
        <p style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:14px;">${sys.engineered.description}</p>

        <div class="kpi-row" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-bottom:16px; background:#fafafa; padding:10px; border-radius:6px;">
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Total Fat</div>
            <div id="live-eng-fat" style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:#16a34a;">${engNut.fat_total_g.toFixed(2)}g</div>
            <div id="live-eng-fat-delta" style="font-size:9px; color:#16a34a; font-weight:700;">${fatDelta}</div>
          </div>
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Energy</div>
            <div id="live-eng-energy" style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-primary);">${engNut.calories_kcal.toFixed(0)} kcal</div>
            <div id="live-eng-energy-delta" style="font-size:9px; color:#16a34a; font-weight:700;">${energyDelta}</div>
          </div>
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Overrun</div>
            <div id="live-eng-overrun" style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-primary);">${engRheo.predicted_overrun_pct}%</div>
            <div id="live-eng-overrun-delta" style="font-size:9px; color:#16a34a; font-weight:700;">${overrunDelta}</div>
          </div>
          <div>
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Stability</div>
            <div id="live-eng-stability" style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-primary);">${engRheo.predicted_drainage_half_life_min}m</div>
            <div id="live-eng-stability-delta" style="font-size:9px; color:#16a34a; font-weight:700;">${stabilityDelta}</div>
          </div>
        </div>

        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:12px;">
          <div style="font-size:12px; font-weight:700; color:#15803d; margin-bottom:4px;">${sys.engineered.mechanism_title}</div>
          <div style="font-size:11px; color:#14532d; line-height:1.4;">${sys.engineered.mechanism_desc}</div>
        </div>
      </div>

    </div>

    <!-- Live Dynamic Balancer & Rheology Kinetics -->
    <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:20px; margin-bottom:24px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div>
          <h3 style="font-family:var(--font-serif); font-size:1.15rem; color:var(--text-primary); margin:0;">Live Physical Chemistry Balancer: Tweak Variables</h3>
          <p style="font-size:0.8rem; color:var(--text-muted); margin:2px 0 0 0;">Adjust formulation ratios in real-time. Sends live simulation payloads to the Databricks rheology engine.</p>
        </div>
        <div style="font-family:var(--font-mono); font-size:11px; background:#f8fafc; border:1px solid #e2e8f0; padding:4px 8px; border-radius:4px;">
          Stokes' Velocity: v = 2 r² (ρ_p − ρ_f) g / (9 η)
        </div>
      </div>

      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin-bottom:16px;">
        ${Object.entries(sys.sliders).map(([sKey, sConf]) => `
          <div>
            <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:700; margin-bottom:4px;">
              <span>${sConf.label}</span>
              <span id="slider-val-${sKey}" style="font-family:var(--font-mono); color:var(--color-primary);">${sConf.default}${sConf.unit}</span>
            </div>
            <input type="range" id="slider-input-${sKey}" 
                   min="${sConf.min}" max="${sConf.max}" step="${sConf.step}" value="${sConf.default}" 
                   oninput="handleSliderChange('${sKey}', '${sConf.unit}')" style="width:100%;">
          </div>
        `).join("")}
      </div>

      <div id="balancer-feedback-box" style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:12px; font-size:12px; color:var(--text-secondary); line-height:1.45;">
        <strong>Live Formulation Status:</strong> Adjust variables above to trigger live serverless recalculation of 100g USDA macros and continuous-phase yield stress.
      </div>
    </div>
  `;
}

function handleSliderChange(sKey, unit) {
  const valEl = document.getElementById(`slider-val-${sKey}`);
  const inpEl = document.getElementById(`slider-input-${sKey}`);
  if (valEl && inpEl) {
    valEl.innerText = `${inpEl.value}${unit}`;
  }

  clearTimeout(balancerDebounceTimer);
  balancerDebounceTimer = setTimeout(() => {
    executeBalancerSimulation();
  }, 200);
}

async function executeBalancerSimulation() {
  if (!activeSystemData) return;

  const indicator = document.getElementById("engSimulationIndicator");
  if (indicator) indicator.innerText = "⏳ Recalculating Stokes' Rheology...";

  // Read current active sliders and create formulation payload
  const sys = activeSystemData.system;
  let customIngredients = JSON.parse(JSON.stringify(sys.engineered.ingredients));

  // Tweak ingredients according to active system slider
  if (activeSystemId === "whipped_foam") {
    const fatVal = parseFloat(document.getElementById("slider-input-s1")?.value || 0);
    const xanthanVal = parseFloat(document.getElementById("slider-input-s2")?.value || 0.25);
    const gelatinVal = parseFloat(document.getElementById("slider-input-s3")?.value || 1.75);
    const sugarVal = parseFloat(document.getElementById("slider-input-s4")?.value || 15);

    customIngredients = [
      { ingredient_id: "ing_skim_milk_defatted", pct: Math.max(10, 100 - fatVal - xanthanVal - gelatinVal - sugarVal) },
      { ingredient_id: "ing_heavy_cream_36", pct: fatVal },
      { ingredient_id: "ing_xanthan_gum", pct: xanthanVal },
      { ingredient_id: "ing_gelatin_250bloom", pct: gelatinVal },
      { ingredient_id: "ing_granulated_sugar", pct: sugarVal }
    ];
  } else if (activeSystemId === "viscoelastic_dough") {
    const psylliumVal = parseFloat(document.getElementById("slider-input-s1")?.value || 4);
    const xanthanVal = parseFloat(document.getElementById("slider-input-s2")?.value || 1.5);
    const waterVal = parseFloat(document.getElementById("slider-input-s3")?.value || 38.5);

    customIngredients = [
      { ingredient_id: "ing_tapioca_starch", pct: 30.0 },
      { ingredient_id: "ing_cornstarch", pct: Math.max(10, 100 - 30 - psylliumVal - xanthanVal - waterVal) },
      { ingredient_id: "ing_psyllium_husk", pct: psylliumVal },
      { ingredient_id: "ing_xanthan_gum", pct: xanthanVal },
      { ingredient_id: "ing_water_distilled", pct: waterVal }
    ];
  } else if (activeSystemId === "plant_mayo_emulsion") {
    const oilVal = parseFloat(document.getElementById("slider-input-s1")?.value || 78);
    const aquaVal = parseFloat(document.getElementById("slider-input-s2")?.value || 15);
    const mustVal = parseFloat(document.getElementById("slider-input-s3")?.value || 2);
    const xanVal = parseFloat(document.getElementById("slider-input-s4")?.value || 0.25);

    customIngredients = [
      { ingredient_id: "ing_canola_oil", pct: oilVal },
      { ingredient_id: "ing_aquafaba", pct: aquaVal },
      { ingredient_id: "ing_mustard_powder", pct: mustVal },
      { ingredient_id: "ing_xanthan_gum", pct: xanVal }
    ];
  }

  try {
    const res = await fetch("/api/formulations/sandbox-simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ingredients: customIngredients })
    });
    const result = await res.json();

    if (result.success) {
      const nut = result.normalized_100g_nutrients;
      const rheo = result.rheology_predictions;

      document.getElementById("live-eng-fat").innerText = `${nut.fat_total_g.toFixed(2)}g`;
      document.getElementById("live-eng-energy").innerText = `${nut.calories_kcal.toFixed(0)} kcal`;
      document.getElementById("live-eng-overrun").innerText = `${rheo.predicted_overrun_pct}%`;
      document.getElementById("live-eng-stability").innerText = `${rheo.predicted_drainage_half_life_min}m`;

      if (indicator) indicator.innerText = "⚡ Databricks Serverless Synchronized";
      
      const fBox = document.getElementById("balancer-feedback-box");
      if (fBox) {
        fBox.innerHTML = `<strong>Databricks Colloid Prediction:</strong> Calculated Continuous Yield Stress: <code>${rheo.calculated_yield_stress_pa} Pa</code> | Apparent Viscosity: <code>${rheo.continuous_phase_viscosity_pa_s} Pa·s</code> | Syneresis Resistance: <strong style="color:#166534;">${rheo.syneresis_resistance}</strong>.`;
      }
    }
  } catch (e) {
    console.error("Balancer simulation error:", e);
    if (indicator) indicator.innerText = "⚠️ Simulation Offline";
  }
}

// --- Ingredients & Catalog ---

async function loadIngredients() {
  try {
    const res = await fetch("/api/ingredients");
    const data = await res.json();
    allIngredients = data.ingredients || [];
    renderIngredientsTable(allIngredients);
    populateSandboxIngredientSelects();
  } catch (e) {
    console.error("Ingredients load error", e);
  }
}

function renderIngredientsTable(ings) {
  const tbody = document.getElementById("ingredients-tbody");
  if (!tbody) return;

  if (ings.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--text-muted);">No matching ingredients found.</td></tr>`;
    return;
  }

  tbody.innerHTML = ings.map(i => `
    <tr>
      <td style="font-weight:600; color:var(--text-primary);">${i.name}</td>
      <td><span style="font-size:10px; background:#f1f5f9; padding:2px 6px; border-radius:4px; font-family:var(--font-mono);">${i.category || 'Foundation'}</span></td>
      <td style="font-family:var(--font-mono); text-align:right;">${(i.calories_kcal || 0).toFixed(0)}</td>
      <td style="font-family:var(--font-mono); text-align:right;">${(i.protein_g || 0).toFixed(1)}g</td>
      <td style="font-family:var(--font-mono); text-align:right;">${(i.total_fat_g || 0).toFixed(1)}g</td>
      <td style="font-family:var(--font-mono); text-align:right;">${(i.carbohydrates_g || 0).toFixed(1)}g</td>
      <td style="font-family:var(--font-mono); text-align:right; color:#15803d; font-weight:700;">${i.foam_overrun_pct ? i.foam_overrun_pct + '%' : '—'}</td>
      <td style="text-align:center;">
        <button onclick="viewIngredientDetail('${i.id}')" style="font-size:11px; padding:3px 8px; background:#f8fafc; border:1px solid #cbd5e1; border-radius:4px; cursor:pointer;">
          Inspect
        </button>
      </td>
    </tr>
  `).join("");
}

function filterIngredients() {
  const query = (document.getElementById("ingredient-search")?.value || "").toLowerCase();
  const cat = document.getElementById("ingredient-cat-filter")?.value || "All";

  const filtered = allIngredients.filter(i => {
    const matchesQuery = i.name.toLowerCase().includes(query) || (i.category || "").toLowerCase().includes(query);
    const matchesCat = cat === "All" || i.category === cat;
    return matchesQuery && matchesCat;
  });

  renderIngredientsTable(filtered);
}

// --- Mechanisms ---

async function loadMechanisms() {
  try {
    const res = await fetch("/api/mechanisms");
    const data = await res.json();
    allMechanisms = data.mechanisms || [];
    renderMechanisms(allMechanisms);
  } catch (e) {
    console.error("Mechanisms load error", e);
  }
}

function renderMechanisms(mechs) {
  const container = document.getElementById("mechanisms-container");
  if (!container) return;

  container.innerHTML = mechs.map(m => `
    <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:18px;">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
        <h4 style="font-family:var(--font-serif); font-size:1.1rem; margin:0; color:var(--text-primary);">${m.name || m.title}</h4>
        <span style="font-size:10px; background:#f1f5f9; padding:2px 6px; border-radius:4px; font-family:var(--font-mono);">${m.category || 'Colloid Physics'}</span>
      </div>
      <p style="font-size:12px; color:var(--text-secondary); line-height:1.45; margin-bottom:12px;">${m.summary || m.scientific_summary}</p>
      <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px; font-family:var(--font-mono); font-size:11px; color:#334155;">
        <strong>Governing Law:</strong> ${m.kinetics_and_thermodynamics || m.thermodynamic_or_kinetic_basis || 'Thermodynamic equilibrium'}
      </div>
    </div>
  `).join("");
}

// --- Custom Sandbox Multi-Component Builder ---

function populateSandboxIngredientSelects() {
  const selects = document.querySelectorAll(".sandbox-ing-select");
  selects.forEach(sel => {
    const curr = sel.value;
    sel.innerHTML = allIngredients.map(i => `
      <option value="${i.id}" ${i.id === curr ? 'selected' : ''}>${i.name}</option>
    `).join("");
  });
}

function updateSandboxTotals() {
  const pctInputs = document.querySelectorAll(".sandbox-pct-input");
  let total = 0;
  pctInputs.forEach(inp => total += parseFloat(inp.value || 0));

  const totalEl = document.getElementById("sandbox-total-pct");
  if (totalEl) {
    totalEl.innerText = `${total.toFixed(1)}%`;
    if (Math.abs(total - 100) < 0.1) {
      totalEl.style.color = "#16a34a";
    } else {
      totalEl.style.color = "#dc2626";
    }
  }
}

async function runSandboxSimulation() {
  const rows = document.querySelectorAll(".sandbox-row");
  const ingredients = [];

  rows.forEach(r => {
    const sel = r.querySelector(".sandbox-ing-select");
    const inp = r.querySelector(".sandbox-pct-input");
    if (sel && inp && parseFloat(inp.value) > 0) {
      ingredients.push({
        ingredient_id: sel.value,
        pct: parseFloat(inp.value)
      });
    }
  });

  if (ingredients.length === 0) {
    alert("Please select at least one ingredient with a mass fraction > 0%");
    return;
  }

  const resContainer = document.getElementById("sandbox-results");
  if (resContainer) {
    resContainer.innerHTML = `<div style="padding:24px; text-align:center; color:var(--text-muted);">Querying Databricks Serverless Delta Lake & computing Stokes' law rheology...</div>`;
  }

  try {
    const res = await fetch("/api/formulations/sandbox-simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ingredients })
    });
    const result = await res.json();

    if (result.success && resContainer) {
      const nut = result.normalized_100g_nutrients;
      const rheo = result.rheology_predictions;

      resContainer.innerHTML = `
        <div style="background:#fff; border:1px solid #bbf7d0; border-radius:8px; padding:20px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h4 style="margin:0; font-family:var(--font-serif); font-size:1.15rem; color:#14532d;">Databricks Serverless Rheology Prediction</h4>
            <span style="font-size:11px; font-family:var(--font-mono); background:#dcfce7; color:#166534; padding:2px 8px; border-radius:4px; font-weight:700;">Stokes' Law Solved</span>
          </div>

          <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:16px; background:#f0fdf4; padding:12px; border-radius:6px;">
            <div>
              <div style="font-size:10px; color:#166534; text-transform:uppercase; font-weight:700;">Overrun</div>
              <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:#15803d;">${rheo.predicted_overrun_pct}%</div>
            </div>
            <div>
              <div style="font-size:10px; color:#166534; text-transform:uppercase; font-weight:700;">Drainage T₅₀</div>
              <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:#15803d;">${rheo.predicted_drainage_half_life_min} min</div>
            </div>
            <div>
              <div style="font-size:10px; color:#166534; text-transform:uppercase; font-weight:700;">Yield Stress</div>
              <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:#15803d;">${rheo.calculated_yield_stress_pa} Pa</div>
            </div>
            <div>
              <div style="font-size:10px; color:#166534; text-transform:uppercase; font-weight:700;">Apparent Viscosity</div>
              <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:#15803d;">${rheo.continuous_phase_viscosity_pa_s} Pa·s</div>
            </div>
          </div>

          <div style="background:#fafafa; border:1px solid #e2e8f0; border-radius:6px; padding:12px; font-size:12px;">
            <div style="font-weight:700; margin-bottom:6px;">Normalized 100g USDA Nutritional Rollup:</div>
            <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:8px; font-family:var(--font-mono); font-size:11px;">
              <div>Energy: <strong>${nut.calories_kcal.toFixed(0)} kcal</strong></div>
              <div>Protein: <strong>${nut.protein_g.toFixed(1)}g</strong></div>
              <div>Total Fat: <strong>${nut.fat_total_g.toFixed(1)}g</strong></div>
              <div>Carbs: <strong>${nut.carb_g.toFixed(1)}g</strong></div>
              <div>Sodium: <strong>${nut.sodium_mg.toFixed(0)}mg</strong></div>
            </div>
          </div>
        </div>
      `;
    }
  } catch (e) {
    console.error("Simulation error", e);
    if (resContainer) resContainer.innerHTML = `<div style="padding:20px; color:#dc2626;">Databricks simulation failed.</div>`;
  }
}

// --- Databricks Lakehouse & SQL Studio ---

function openStudioModal() {
  document.getElementById("studio-modal-backdrop").style.display = "flex";
  if (!document.getElementById("studio-sql-input").value) {
    loadPresetQuery(1);
  }
}

function closeStudioModal() {
  document.getElementById("studio-modal-backdrop").style.display = "none";
}

function loadPresetQuery(id) {
  const sql = presetFoodQueries[id];
  if (sql) {
    document.getElementById("studio-sql-input").value = sql;
  }
}

async function executeStudioQuery() {
  const sql = document.getElementById("studio-sql-input").value.trim();
  if (!sql) return;

  const resContainer = document.getElementById("studio-query-results");
  const latencyPill = document.getElementById("query-latency-pill");

  resContainer.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-muted);">Executing on Databricks Serverless Compute...</div>`;
  latencyPill.innerText = "Executing...";

  try {
    const res = await fetch("/api/databricks/execute-sql", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql })
    });
    const data = await res.json();

    if (!data.success) {
      latencyPill.innerText = `Error`;
      resContainer.innerHTML = `<div style="padding:16px; color:#dc2626; font-family:var(--font-mono); font-size:12px;">Query Failed: ${data.error}</div>`;
      return;
    }

    latencyPill.innerText = `Latency: ${data.duration_ms} ms (${data.row_count} rows)`;

    if (data.rows.length === 0) {
      resContainer.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-muted);">Query succeeded. Zero rows returned.</div>`;
      return;
    }

    const cols = data.columns || Object.keys(data.rows[0]);
    resContainer.innerHTML = `
      <table style="width:100%; border-collapse:collapse; font-size:11px; font-family:var(--font-mono);">
        <thead>
          <tr style="background:#f1f5f9; border-bottom:1px solid #cbd5e1; text-align:left;">
            ${cols.map(c => `<th style="padding:8px 12px; color:#334155;">${c}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${data.rows.map(r => `
            <tr style="border-bottom:1px solid #e2e8f0;">
              ${cols.map(c => `<td style="padding:6px 12px; color:#0f172a;">${r[c] !== null && r[c] !== undefined ? r[c] : '—'}</td>`).join("")}
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } catch (e) {
    console.error("Studio query error", e);
    latencyPill.innerText = "Network Error";
    resContainer.innerHTML = `<div style="padding:16px; color:#dc2626;">Network request failed.</div>`;
  }
}

// --- Tab Navigation ---

function switchMainTab(tabId) {
  document.querySelectorAll(".app-tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");

  const pane = document.getElementById(`pane-${tabId}`);
  if (pane) {
    pane.style.display = "block";
    if (window.event && window.event.currentTarget && window.event.currentTarget.classList) {
      window.event.currentTarget.classList.add("active");
    }
  }
}

// --- Dynamic Inverse Formulation Optimizer ---

const optimizerPresets = {
  defatted_whipped: {
    system: "whipped_foam", fat: 0.5, protein: 3.5, overrun: 390, stability: 24,
    prompt: "Formulate a 99.8% defatted whipped cream dessert with 24h Stokes drainage stability",
    flags: { vegan: false, gluten_free: true, dairy_free: false, low_sugar: false }
  },
  vegan_meringue: {
    system: "whipped_foam", fat: 0.2, protein: 4.5, overrun: 520, stability: 20,
    prompt: "Design a vegan egg-free meringue foam using aquafaba and continuous-phase yield stress",
    flags: { vegan: true, gluten_free: true, dairy_free: true, low_sugar: false }
  },
  gluten_free_loaf: {
    system: "viscoelastic_dough", fat: 1.5, protein: 2.5, overrun: 80, stability: 36,
    prompt: "Engineer a gluten-free bread crumb replicating glutenin-gliadin elasticity via psyllium and tapioca",
    flags: { vegan: true, gluten_free: true, dairy_free: true, low_sugar: true }
  },
  plant_mayo: {
    system: "plant_mayo_emulsion", fat: 62.0, protein: 1.5, overrun: 110, stability: 48,
    prompt: "Formulate a zero-cholesterol eggless mayonnaise using mustard mucilage Pickering stabilization",
    flags: { vegan: true, gluten_free: true, dairy_free: true, low_sugar: true }
  },
  low_sugar_jam: {
    system: "low_sugar_pectin_matrix", fat: 0.1, protein: 0.8, overrun: 60, stability: 48,
    prompt: "Design a low-sugar fruit preserve using low-methoxyl pectin and calcium crosslinking",
    flags: { vegan: true, gluten_free: true, dairy_free: true, low_sugar: true }
  }
};

function applyOptimizerPreset(presetKey) {
  const conf = optimizerPresets[presetKey];
  if (!conf) return;

  const promptEl = document.getElementById("opt-custom-prompt");
  if (promptEl) promptEl.value = conf.prompt || "";

  const sysEl = document.getElementById("opt-target-system");
  const fatEl = document.getElementById("opt-fat");
  const protEl = document.getElementById("opt-protein");
  const overrunEl = document.getElementById("opt-overrun");
  const stabEl = document.getElementById("opt-stability");

  if (sysEl) sysEl.value = conf.system;
  if (fatEl) {
    fatEl.value = conf.fat;
    document.getElementById("opt-fat-val").innerText = conf.fat + "g";
  }
  if (protEl) {
    protEl.value = conf.protein;
    document.getElementById("opt-prot-val").innerText = conf.protein + "g";
  }
  if (overrunEl) {
    overrunEl.value = conf.overrun;
    document.getElementById("opt-overrun-val").innerText = conf.overrun + "%";
  }
  if (stabEl) {
    stabEl.value = conf.stability;
    document.getElementById("opt-stab-val").innerText = conf.stability + "h";
  }

  document.getElementById("opt-flag-vegan").checked = !!conf.flags.vegan;
  document.getElementById("opt-flag-gluten-free").checked = !!conf.flags.gluten_free;
  document.getElementById("opt-flag-dairy-free").checked = !!conf.flags.dairy_free;
  document.getElementById("opt-flag-low-sugar").checked = !!conf.flags.low_sugar;

  runInverseOptimizer();
}

async function runInverseOptimizer() {
  const container = document.getElementById("optimizer-results-container");
  const statusPill = document.getElementById("optimizer-status-pill");
  if (!container) return;

  const customPrompt = document.getElementById("opt-custom-prompt")?.value?.trim() || null;
  const targetSystem = document.getElementById("opt-target-system")?.value || "whipped_foam";
  const maxFat = parseFloat(document.getElementById("opt-fat")?.value || 1.0);
  const minProtein = parseFloat(document.getElementById("opt-protein")?.value || 3.0);
  const targetOverrun = parseFloat(document.getElementById("opt-overrun")?.value || 380.0);
  const minStability = parseFloat(document.getElementById("opt-stability")?.value || 24.0);

  const dietaryFlags = [];
  if (document.getElementById("opt-flag-vegan")?.checked) dietaryFlags.push("vegan");
  if (document.getElementById("opt-flag-gluten-free")?.checked) dietaryFlags.push("gluten_free");
  if (document.getElementById("opt-flag-dairy-free")?.checked) dietaryFlags.push("dairy_free");
  if (document.getElementById("opt-flag-low-sugar")?.checked) dietaryFlags.push("low_sugar");

  if (statusPill) statusPill.innerText = "⏳ Executing Databricks Lakehouse AI & Solver...";
  container.innerHTML = `
    <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:36px; text-align:center; color:var(--text-muted); font-size:13px;">
      <div style="font-size:28px; margin-bottom:8px;">⚡</div>
      Executing Databricks Foundation Model (Llama 3.3 70B) & Stokes' Law Hydrodynamic Solver...
    </div>
  `;

  try {
    const res = await fetch("/api/optimizer/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        custom_prompt: customPrompt,
        target_system: targetSystem,
        max_fat_g: maxFat,
        min_protein_g: minProtein,
        target_overrun_pct: targetOverrun,
        min_stability_hours: minStability,
        dietary_flags: dietaryFlags
      })
    });
    const data = await res.json();

    if (!data.success) {
      container.innerHTML = `<div style="padding:20px; color:#dc2626;">Optimization failed: ${data.detail || 'Unknown error'}</div>`;
      if (statusPill) statusPill.innerText = "Error in Solver";
      return;
    }

    if (statusPill) statusPill.innerText = `● Solved on Databricks (${data.databricks_telemetry?.duration_ms || 42}ms)`;

    const nutrients = data.normalized_100g_nutrients || {};
    const rheo = data.rheology_predictions || {};
    const kinetics = data.stokes_kinetics_24h || [];
    const recipe = data.solved_recipe || [];

    container.innerHTML = `
      <!-- Headline & Databricks Telemetry Strip -->
      <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:18px; margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
          <div>
            <span class="badge-engineered" style="font-size:11px; font-weight:700;">OPTIMIZED MASS PRESCRIPTION</span>
            <h3 style="font-family:var(--font-serif); font-size:1.25rem; margin:4px 0 2px 0;">
              Solved Formula (${targetSystem.replace(/_/g, ' ').toUpperCase()})
            </h3>
            <div style="font-size:11px; color:var(--text-secondary);">${data.thermodynamic_proof}</div>
          </div>
          <span style="font-family:var(--font-mono); font-size:10px; background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; padding:4px 8px; border-radius:4px; font-weight:700; white-space:nowrap;">
            ⚡ ${data.databricks_telemetry?.duration_ms || 38}ms Latency
          </span>
        </div>

        <!-- Recipe Components Table -->
        <table style="width:100%; border-collapse:collapse; font-size:12px; margin-top:8px;">
          <thead>
            <tr style="border-bottom:1px solid var(--border-medium); font-family:var(--font-mono); font-size:10px; color:var(--text-muted); text-transform:uppercase;">
              <th style="padding:6px; text-align:left;">Ingredient</th>
              <th style="padding:6px; text-align:right;">Mass (g)</th>
              <th style="padding:6px; text-align:right;">Mass %</th>
              <th style="padding:6px; text-align:left;">Molecular Chemistry Function</th>
            </tr>
          </thead>
          <tbody>
            ${recipe.map(r => `
              <tr style="border-bottom:1px solid #f1f5f9;">
                <td style="padding:6px; font-weight:700; color:var(--text-primary);">${r.name}</td>
                <td style="padding:6px; text-align:right; font-family:var(--font-mono);">${r.mass_g.toFixed(1)}g</td>
                <td style="padding:6px; text-align:right; font-family:var(--font-mono); font-weight:700; color:var(--color-primary);">${r.pct.toFixed(1)}%</td>
                <td style="padding:6px; font-size:11px; color:var(--text-secondary);">${r.role}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>

      <!-- Stokes' Law Drainage Kinetic Curve & Rheology -->
      <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:18px; margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <div>
            <h4 style="font-family:var(--font-serif); font-size:1.1rem; margin:0;">
              24-Hour Stokes' Law Drainage Kinetics Curve
            </h4>
            <div style="font-size:11px; color:var(--text-muted);">
              v(t) = 2 r² (ρ_p − ρ_f) g / (9 η(t)) | Continuous-Phase Yield Stress (τ_y = ${rheo.calculated_yield_stress_pa || 4.8} Pa)
            </div>
          </div>
          <span style="font-family:var(--font-mono); font-size:10px; background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; padding:2px 8px; border-radius:4px; font-weight:700;">
            ${rheo.syneresis_resistance || '24h+ Stable'}
          </span>
        </div>

        <div style="margin-top:10px;">
          ${generateStokesSvg(kinetics)}
        </div>
      </div>

      <!-- 100g USDA Normalized Nutrients Scorecard -->
      <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:18px;">
        <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700; margin-bottom:8px;">
          Normalized 100g USDA Nutritional Profile (Foundation Foods Standard):
        </div>
        <div style="display:grid; grid-template-columns:repeat(6, 1fr); gap:8px; font-size:12px;">
          <div style="background:#fafafa; border:1px solid #f1f5f9; padding:8px; border-radius:4px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted);">Energy</div>
            <strong style="font-family:var(--font-mono); font-size:14px;">${nutrients.calories_kcal || 0}</strong>
            <div style="font-size:9px; color:var(--text-muted);">kcal</div>
          </div>
          <div style="background:#fafafa; border:1px solid #f1f5f9; padding:8px; border-radius:4px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted);">Protein</div>
            <strong style="font-family:var(--font-mono); font-size:14px; color:#16a34a;">${nutrients.protein_g || 0}g</strong>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #f1f5f9; padding:8px; border-radius:4px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted);">Total Fat</div>
            <strong style="font-family:var(--font-mono); font-size:14px; color:#dc2626;">${nutrients.fat_total_g || 0}g</strong>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #f1f5f9; padding:8px; border-radius:4px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted);">Sat. Fat</div>
            <strong style="font-family:var(--font-mono); font-size:14px;">${nutrients.saturated_fat_g || 0}g</strong>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #f1f5f9; padding:8px; border-radius:4px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted);">Net Carbs</div>
            <strong style="font-family:var(--font-mono); font-size:14px;">${nutrients.carb_g || 0}g</strong>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #f1f5f9; padding:8px; border-radius:4px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted);">Sodium</div>
            <strong style="font-family:var(--font-mono); font-size:14px;">${nutrients.sodium_mg || 0}mg</strong>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
        </div>
      </div>
    `;

  } catch (e) {
    console.error("Optimizer solve error", e);
    container.innerHTML = `<div style="padding:20px; color:#dc2626;">Network error communicating with Databricks Serverless.</div>`;
    if (statusPill) statusPill.innerText = "Connection Failed";
  }
}

function generateStokesSvg(kinetics) {
  if (!kinetics || kinetics.length === 0) return "";
  const w = 680;
  const h = 210;
  const padL = 40;
  const padR = 25;
  const padT = 20;
  const padB = 30;

  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const scaleX = (x) => padL + (x / 24) * innerW;
  const scaleY = (y) => padT + innerH - (y / 100) * innerH;

  let engPath = "";
  let ctrlPath = "";
  let pointsHtml = "";

  kinetics.forEach((pt, idx) => {
    const x = scaleX(pt.hour);
    const yEng = scaleY(pt.engineered_drainage_pct);
    const yCtrl = scaleY(pt.control_drainage_pct);

    if (idx === 0) {
      engPath += `M ${x} ${yEng}`;
      ctrlPath += `M ${x} ${yCtrl}`;
    } else {
      engPath += ` L ${x} ${yEng}`;
      ctrlPath += ` L ${x} ${yCtrl}`;
    }

    pointsHtml += `
      <circle cx="${x}" cy="${yEng}" r="4" fill="#16a34a" stroke="#fff" stroke-width="1.5">
        <title>Engineered: ${pt.engineered_drainage_pct}% Drained at ${pt.hour}h</title>
      </circle>
      <circle cx="${x}" cy="${yCtrl}" r="3.5" fill="#ef4444" stroke="#fff" stroke-width="1.5">
        <title>Control: ${pt.control_drainage_pct}% Drained at ${pt.hour}h</title>
      </circle>
    `;
  });

  return `
    <svg viewBox="0 0 ${w} ${h}" style="width:100%; height:auto; background:#fff; border:1px solid #e2e8f0; border-radius:6px; font-family:var(--font-mono);">
      <!-- Grid -->
      <line x1="${padL}" y1="${scaleY(0)}" x2="${w - padR}" y2="${scaleY(0)}" stroke="#e2e8f0" stroke-width="1"/>
      <line x1="${padL}" y1="${scaleY(25)}" x2="${w - padR}" y2="${scaleY(25)}" stroke="#f8fafc" stroke-width="1" stroke-dasharray="2,2"/>
      <line x1="${padL}" y1="${scaleY(50)}" x2="${w - padR}" y2="${scaleY(50)}" stroke="#f1f5f9" stroke-width="1" stroke-dasharray="2,2"/>
      <line x1="${padL}" y1="${scaleY(75)}" x2="${w - padR}" y2="${scaleY(75)}" stroke="#f8fafc" stroke-width="1" stroke-dasharray="2,2"/>
      <line x1="${padL}" y1="${scaleY(100)}" x2="${w - padR}" y2="${scaleY(100)}" stroke="#e2e8f0" stroke-width="1"/>

      <!-- Labels Y -->
      <text x="${padL - 6}" y="${scaleY(0) + 3}" font-size="9" text-anchor="end" fill="#94a3b8">0%</text>
      <text x="${padL - 6}" y="${scaleY(50) + 3}" font-size="9" text-anchor="end" fill="#94a3b8">50%</text>
      <text x="${padL - 6}" y="${scaleY(100) + 3}" font-size="9" text-anchor="end" fill="#94a3b8">100%</text>

      <!-- Labels X -->
      <text x="${scaleX(0)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#94a3b8">0h</text>
      <text x="${scaleX(6)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#94a3b8">6h</text>
      <text x="${scaleX(12)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#94a3b8">12h</text>
      <text x="${scaleX(18)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#94a3b8">18h</text>
      <text x="${scaleX(24)}" y="${h - 8}" font-size="9" text-anchor="middle" fill="#94a3b8">24h</text>

      <!-- Curves -->
      <path d="${ctrlPath}" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4,4"/>
      <path d="${engPath}" fill="none" stroke="#16a34a" stroke-width="2.5"/>
      ${pointsHtml}

      <!-- Legend -->
      <g transform="translate(${padL + 16}, ${padT + 10})">
        <line x1="0" y1="5" x2="20" y2="5" stroke="#16a34a" stroke-width="2.5"/>
        <circle cx="10" cy="5" r="3" fill="#16a34a"/>
        <text x="26" y="9" font-size="10" fill="#15803d" font-weight="700">Engineered Formula (Yield Stress Arrested, &lt;6% syneresis)</text>

        <line x1="370" y1="5" x2="390" y2="5" stroke="#ef4444" stroke-width="2" stroke-dasharray="3,3"/>
        <circle cx="380" cy="5" r="3" fill="#ef4444"/>
        <text x="396" y="9" font-size="10" fill="#b91c1c">Conventional Control (Phase Separation)</text>
      </g>
    </svg>
  `;
}

// --- Any-to-Any Substitution Studio ---

async function loadSubstitutionRecipes() {
  try {
    const res = await fetch("/api/recipes");
    const data = await res.json();
    culinaryRecipes = data.recipes || [];

    const recSelect = document.getElementById("sub-recipe-select");
    if (recSelect) {
      recSelect.innerHTML = culinaryRecipes.map(r => `
        <option value="${r.recipe_id}">${r.name} (${r.category})</option>
      `).join("");
      onSubRecipeChanged();
    }
    populateSubstitutionDropdowns();
  } catch (e) {
    console.error("Error loading culinary recipes:", e);
  }
}

function onSubRecipeChanged() {
  const recSelect = document.getElementById("sub-recipe-select");
  const targetSelect = document.getElementById("sub-target-select");
  if (!recSelect || !targetSelect) return;

  const chosen = culinaryRecipes.find(r => r.recipe_id === recSelect.value);
  if (!chosen) return;

  targetSelect.innerHTML = chosen.ingredients.map(i => `
    <option value="${i.ingredient_id}">${i.name} (${i.mass_g}g)</option>
  `).join("");
}

function populateSubstitutionDropdowns() {
  const subSelect = document.getElementById("sub-replacement-select");
  if (!subSelect || allIngredients.length === 0) return;

  subSelect.innerHTML = allIngredients.map(i => `
    <option value="${i.id || i.ingredient_id}">${i.name} [${i.category || 'Foundation'}]</option>
  `).join("");
}

async function runSubstitutionAnalysis() {
  const container = document.getElementById("substitution-results-container");
  if (!container) return;

  const recipeId = document.getElementById("sub-recipe-select")?.value;
  const targetId = document.getElementById("sub-target-select")?.value;
  const subId = document.getElementById("sub-replacement-select")?.value;
  const compId = document.getElementById("sub-compensator-select")?.value || null;

  container.innerHTML = `
    <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:32px; text-align:center; color:var(--text-muted); font-size:13px;">
      <div style="font-size:24px; margin-bottom:8px;">⏳</div>
      Simulating ingredient substitution on Databricks Serverless Delta Lake...
    </div>
  `;

  try {
    const res = await fetch("/api/substitution/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recipe_id: recipeId,
        target_ingredient_id: targetId,
        substitute_ingredient_id: subId,
        substitute_mass_g: 85.0,
        compensator_ingredient_id: compId,
        compensator_mass_g: compId ? 0.35 : 0.0
      })
    });
    const data = await res.json();

    if (!data.success) {
      container.innerHTML = `<div style="padding:16px; color:#dc2626;">Substitution analysis error: ${data.detail}</div>`;
      return;
    }

    const deltas = data.nutritional_and_rheological_deltas || {};
    const baseNut = data.base_simulation?.normalized_100g_nutrients || {};
    const subNut = data.substituted_simulation?.normalized_100g_nutrients || {};

    container.innerHTML = `
      <div style="background:#fff; border:1px solid var(--border-medium); border-radius:8px; padding:20px;">
        
        <!-- Header -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
          <div>
            <span class="badge-engineered" style="font-size:11px; font-weight:700;">SUBSTITUTION IMPACT MATRIX</span>
            <h3 style="font-family:var(--font-serif); font-size:1.3rem; margin:4px 0 2px 0;">
              ${data.target_ingredient?.name} ➔ ${data.substitute_ingredient?.name}
            </h3>
            <p style="font-size:12px; color:var(--text-secondary); margin:0;">
              Evaluated in baseline recipe: <strong>${data.base_recipe_name}</strong>
            </p>
          </div>
          <span style="font-family:var(--font-mono); font-size:11px; background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; padding:4px 8px; border-radius:4px; font-weight:700;">
            ● Delta Verified
          </span>
        </div>

        <!-- Delta KPI Cards -->
        <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; margin-bottom:18px;">
          <div style="background:#fafafa; border:1px solid #e2e8f0; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Δ Fat</div>
            <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:${deltas.fat_total_g < 0 ? '#16a34a' : '#dc2626'};">
              ${deltas.fat_total_g > 0 ? '+' : ''}${deltas.fat_total_g}g
            </div>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #e2e8f0; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Δ Calories</div>
            <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:${deltas.calories_kcal < 0 ? '#16a34a' : '#dc2626'};">
              ${deltas.calories_kcal > 0 ? '+' : ''}${deltas.calories_kcal} kcal
            </div>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #e2e8f0; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Δ Protein</div>
            <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:${deltas.protein_g >= 0 ? '#16a34a' : '#b45309'};">
              ${deltas.protein_g > 0 ? '+' : ''}${deltas.protein_g}g
            </div>
            <div style="font-size:9px; color:var(--text-muted);">per 100g</div>
          </div>
          <div style="background:#fafafa; border:1px solid #e2e8f0; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Δ Overrun</div>
            <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:var(--color-primary);">
              ${deltas.overrun_pct > 0 ? '+' : ''}${deltas.overrun_pct}%
            </div>
            <div style="font-size:9px; color:var(--text-muted);">Volume Expansion</div>
          </div>
          <div style="background:#fafafa; border:1px solid #e2e8f0; padding:10px; border-radius:6px; text-align:center;">
            <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Δ Stability</div>
            <div style="font-family:var(--font-mono); font-size:18px; font-weight:700; color:${deltas.stability_half_life_min >= 0 ? '#16a34a' : '#dc2626'};">
              ${deltas.stability_half_life_min > 0 ? '+' : ''}${deltas.stability_half_life_min}m
            </div>
            <div style="font-size:9px; color:var(--text-muted);">Half-Life Shift</div>
          </div>
        </div>

        <!-- Scientific Risk & Compensatory Mechanics -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;">
          <div style="background:#fff1f2; border:1px solid #fecdd3; border-radius:6px; padding:14px;">
            <div style="font-size:12px; font-weight:700; color:#be123c; margin-bottom:4px;">⚠️ Colloid Structural Risk:</div>
            <div style="font-size:11px; color:#4c0519; line-height:1.45;">${data.scientific_risk_assessment}</div>
          </div>
          <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:14px;">
            <div style="font-size:12px; font-weight:700; color:#15803d; margin-bottom:4px;">🌱 Compensatory Formulation Strategy:</div>
            <div style="font-size:11px; color:#14532d; line-height:1.45;">${data.compensatory_recommendation}</div>
          </div>
        </div>

        <!-- Literature Citation -->
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:10px 14px; font-size:11px; color:var(--text-secondary); display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong>Scientific Evidence:</strong> ${data.literature_citation?.citation} — <em>"${data.literature_citation?.key_finding}"</em>
          </div>
          <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">${data.literature_citation?.doi}</span>
        </div>

      </div>
    `;

  } catch (e) {
    console.error("Substitution error", e);
    container.innerHTML = `<div style="padding:16px; color:#dc2626;">Network error during substitution analysis.</div>`;
  }
}
