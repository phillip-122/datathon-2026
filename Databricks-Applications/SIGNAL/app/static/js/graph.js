/**
 * SIGNAL Ecosystem Relationship Map
 * Interactive SVG network graph renderer connecting Research -> Models -> Tools -> Benchmarks.
 */

class EcosystemGraph {
  constructor(containerId, svgId) {
    this.container = document.getElementById(containerId);
    this.svg = document.getElementById(svgId);
    this.overlay = document.getElementById("nodeDetailOverlay");
    this.overlayContent = document.getElementById("overlayContent");
    this.closeOverlayBtn = document.getElementById("closeOverlayBtn");

    this.graphData = { nodes: [], edges: [] };
    this.transform = { x: 40, y: 30, scale: 0.95 };
    this.isDragging = false;
    this.dragStart = { x: 0, y: 0 };
    this.activeNode = null;

    this.categoryColors = {
      Research: "#a855f7",
      Model: "#06b6d4",
      Tool: "#10b981",
      Infrastructure: "#f59e0b",
      Benchmark: "#f43f5e"
    };

    this.initEvents();
  }

  initEvents() {
    if (this.closeOverlayBtn) {
      this.closeOverlayBtn.addEventListener("click", () => this.hideOverlay());
    }

    // Pan / Drag handlers
    this.container.addEventListener("mousedown", (e) => {
      if (e.target.closest(".node-element")) return;
      this.isDragging = true;
      this.dragStart = { x: e.clientX - this.transform.x, y: e.clientY - this.transform.y };
      this.container.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isDragging) return;
      this.transform.x = e.clientX - this.dragStart.x;
      this.transform.y = e.clientY - this.dragStart.y;
      this.render();
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
      this.container.style.cursor = "grab";
    });

    // Zoom buttons
    const zoomIn = document.getElementById("zoomInBtn");
    const zoomOut = document.getElementById("zoomOutBtn");
    const resetBtn = document.getElementById("resetGraphBtn");

    if (zoomIn) zoomIn.addEventListener("click", () => this.zoom(1.15));
    if (zoomOut) zoomOut.addEventListener("click", () => this.zoom(0.85));
    if (resetBtn) resetBtn.addEventListener("click", () => this.resetView());

    // Search input
    const searchInput = document.getElementById("graphSearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        this.filterBySearch(e.target.value);
      });
    }
  }

  zoom(factor) {
    this.transform.scale = Math.max(0.4, Math.min(2.5, this.transform.scale * factor));
    this.render();
  }

  resetView() {
    this.transform = { x: 40, y: 30, scale: 0.95 };
    this.render();
  }

  setData(data) {
    this.graphData = data;
    this.layoutNodes();
    this.render();
  }

  layoutNodes() {
    const nodes = this.graphData.nodes || [];
    // Organize into vertical tiers: Research -> Models -> Tools -> Infra/Benchmarks
    const tiers = {
      Research: { x: 120, nodes: [] },
      Model: { x: 400, nodes: [] },
      Tool: { x: 700, nodes: [] },
      Infrastructure: { x: 960, nodes: [] },
      Benchmark: { x: 960, nodes: [] }
    };

    nodes.forEach(n => {
      const cat = n.category || "Tool";
      if (tiers[cat]) tiers[cat].nodes.push(n);
      else tiers.Tool.nodes.push(n);
    });

    // Spread nodes vertically per tier
    const startY = 70;
    const spacingY = 72;

    Object.values(tiers).forEach(tier => {
      tier.nodes.forEach((node, idx) => {
        node.x = tier.x;
        node.y = startY + idx * spacingY;
      });
    });
  }

  filterBySearch(query) {
    const q = query.toLowerCase().trim();
    const nodes = this.svg.querySelectorAll(".node-element");
    const edges = this.svg.querySelectorAll(".edge-line");

    if (!q) {
      nodes.forEach(n => n.classList.remove("faded", "highlighted"));
      edges.forEach(e => e.classList.remove("faded", "highlighted"));
      return;
    }

    const matchingIds = new Set();
    this.graphData.nodes.forEach(n => {
      if (n.label.toLowerCase().includes(q) || (n.description && n.description.toLowerCase().includes(q))) {
        matchingIds.add(n.id);
      }
    });

    nodes.forEach(n => {
      const id = n.dataset.id;
      if (matchingIds.has(id)) {
        n.classList.add("highlighted");
        n.classList.remove("faded");
      } else {
        n.classList.remove("highlighted");
        n.classList.add("faded");
      }
    });
  }

  render() {
    const { nodes, edges } = this.graphData;
    if (!nodes || nodes.length === 0) return;

    const nodeMap = new Map();
    nodes.forEach(n => nodeMap.set(n.id, n));

    // Clear svg
    this.svg.innerHTML = `
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#06b6d4"/>
        </marker>
      </defs>
      <g id="viewport" transform="translate(${this.transform.x}, ${this.transform.y}) scale(${this.transform.scale})">
        <g id="edgesLayer"></g>
        <g id="nodesLayer"></g>
      </g>
    `;

    const edgesLayer = this.svg.querySelector("#edgesLayer");
    const nodesLayer = this.svg.querySelector("#nodesLayer");

    // Draw edges
    edges.forEach(e => {
      const src = nodeMap.get(e.source);
      const tgt = nodeMap.get(e.target);
      if (!src || !tgt) return;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const cx1 = src.x + dx * 0.5;
      const cy1 = src.y;
      const cx2 = src.x + dx * 0.5;
      const cy2 = tgt.y;
      const d = `M ${src.x} ${src.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${tgt.x} ${tgt.y}`;

      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#334155");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("stroke-opacity", "0.6");
      path.setAttribute("marker-end", "url(#arrow)");
      path.setAttribute("class", "edge-line");
      path.dataset.source = e.source;
      path.dataset.target = e.target;
      edgesLayer.appendChild(path);
    });

    // Draw nodes
    nodes.forEach(n => {
      const color = this.categoryColors[n.category] || "#10b981";
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "node-element");
      g.setAttribute("transform", `translate(${n.x}, ${n.y})`);
      g.dataset.id = n.id;
      g.style.cursor = "pointer";

      // Node background pill
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const textWidth = Math.max(110, n.label.length * 8.5 + 24);
      rect.setAttribute("x", -textWidth / 2);
      rect.setAttribute("y", -16);
      rect.setAttribute("width", textWidth);
      rect.setAttribute("height", 32);
      rect.setAttribute("rx", 16);
      rect.setAttribute("fill", "#1e293b");
      rect.setAttribute("stroke", color);
      rect.setAttribute("stroke-width", "1.5");

      // Category indicator dot
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", -textWidth / 2 + 12);
      circle.setAttribute("cy", 0);
      circle.setAttribute("r", 5);
      circle.setAttribute("fill", color);

      // Label text
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", -textWidth / 2 + 24);
      text.setAttribute("y", 4);
      text.setAttribute("fill", "#f8fafc");
      text.setAttribute("font-size", "12px");
      text.setAttribute("font-family", "Inter, sans-serif");
      text.setAttribute("font-weight", "500");
      text.textContent = n.label;

      g.appendChild(rect);
      g.appendChild(circle);
      g.appendChild(text);

      g.addEventListener("click", (evt) => {
        evt.stopPropagation();
        this.selectNode(n);
      });

      nodesLayer.appendChild(g);
    });
  }

  selectNode(node) {
    this.activeNode = node;
    const { edges } = this.graphData;

    // Highlight connected edges
    const connectedEdges = edges.filter(e => e.source === node.id || e.target === node.id);
    const connectedNodeIds = new Set([node.id]);
    connectedEdges.forEach(e => {
      connectedNodeIds.add(e.source);
      connectedNodeIds.add(e.target);
    });

    const nodeEls = this.svg.querySelectorAll(".node-element");
    const edgeEls = this.svg.querySelectorAll(".edge-line");

    nodeEls.forEach(n => {
      if (connectedNodeIds.has(n.dataset.id)) {
        n.style.opacity = "1";
      } else {
        n.style.opacity = "0.2";
      }
    });

    edgeEls.forEach(e => {
      if (e.dataset.source === node.id || e.dataset.target === node.id) {
        e.setAttribute("stroke", "#06b6d4");
        e.setAttribute("stroke-width", "2.5");
        e.setAttribute("stroke-opacity", "1");
        e.setAttribute("marker-end", "url(#arrow-active)");
      } else {
        e.setAttribute("stroke", "#334155");
        e.setAttribute("stroke-width", "1");
        e.setAttribute("stroke-opacity", "0.15");
        e.setAttribute("marker-end", "url(#arrow)");
      }
    });

    this.showOverlay(node, connectedEdges);
  }

  showOverlay(node, edges) {
    if (!this.overlay || !this.overlayContent) return;

    const color = this.categoryColors[node.category] || "#10b981";
    let relHtml = "";
    if (edges.length > 0) {
      relHtml = `
        <div style="margin-top: 12px; font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase;">
          Connected Relationships (${edges.length})
        </div>
        <div style="display: flex; flex-direction: column; gap: 4px; margin-top: 6px; max-height: 140px; overflow-y: auto;">
          ${edges.map(e => `
            <div style="font-size: 12px; padding: 4px 8px; background: #0f172a; border-radius: 4px; display: flex; justify-content: space-between;">
              <span style="color: #cbd5e1;">${e.source === node.id ? `➔ ${e.target}` : `⬅ ${e.source}`}</span>
              <span style="color: #06b6d4; font-family: monospace;">${e.relationship}</span>
            </div>
          `).join("")}
        </div>
      `;
    }

    this.overlayContent.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: ${color};">${node.category}</span>
        <span style="font-size: 11px; color: #64748b;">${node.sub_type || ""}</span>
      </div>
      <h4 style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">${node.label}</h4>
      <p style="font-size: 12px; color: #94a3b8; line-height: 1.4;">${node.description || "Tracked ecosystem entity."}</p>
      ${relHtml}
      <button class="btn btn-sm btn-primary" style="margin-top: 14px; width: 100%;" onclick="window.signalApp.inspectEntityEvidence('${node.id}')">
        Inspect Source Evidence
      </button>
    `;

    this.overlay.classList.remove("hidden");
  }

  hideOverlay() {
    if (this.overlay) this.overlay.classList.add("hidden");
    // Reset highlights
    const nodeEls = this.svg.querySelectorAll(".node-element");
    const edgeEls = this.svg.querySelectorAll(".edge-line");
    nodeEls.forEach(n => n.style.opacity = "1");
    edgeEls.forEach(e => {
      e.setAttribute("stroke", "#334155");
      e.setAttribute("stroke-width", "1.5");
      e.setAttribute("stroke-opacity", "0.6");
      e.setAttribute("marker-end", "url(#arrow)");
    });
  }
}

window.EcosystemGraph = EcosystemGraph;
