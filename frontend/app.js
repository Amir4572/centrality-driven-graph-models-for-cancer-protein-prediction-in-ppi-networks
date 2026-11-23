/* OncoGraph-PPI — React 18 + Plotly.js Frontend SPA
 * Pages:
 *  1. Dashboard
 *  2. Protein Prediction
 *  3. Centrality Analysis
 *  4. PPI Network Visualization
 *  5. Model Comparison
 *  6. Explainability (SHAP)
 *  7. Dataset Information
 */
(function () {
  const { useState, useEffect, useRef } = React;
  const h = React.createElement;

  const PLOT_DARK_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(15,23,42,0.55)",
    font: { family: "Inter, sans-serif", color: "#e2e8f0", size: 12 },
    margin: { t: 42, r: 22, b: 48, l: 58 },
    xaxis: { gridcolor: "rgba(148,163,184,0.14)", zerolinecolor: "rgba(148,163,184,0.25)" },
    yaxis: { gridcolor: "rgba(148,163,184,0.14)", zerolinecolor: "rgba(148,163,184,0.25)" },
  };

  function PlotlyChart({ data, layout, config, style, onPointClick }) {
    const divRef = useRef(null);
    useEffect(() => {
      const el = divRef.current;
      if (!el || !window.Plotly) return;
      const mergedLayout = Object.assign({}, PLOT_DARK_LAYOUT, layout || {}, { autosize: true });
      const mergedConfig = Object.assign({ responsive: true, displaylogo: false }, config || {});
      window.Plotly.react(el, data || [], mergedLayout, mergedConfig);

      if (onPointClick && el.on) {
        el.removeAllListeners && el.removeAllListeners("plotly_click");
        el.on("plotly_click", onPointClick);
      }
      return () => {
        if (el && window.Plotly) {
          window.Plotly.purge(el);
        }
      };
    }, [data, layout]);

    return h("div", { ref: divRef, style: Object.assign({ width: "100%", height: "380px" }, style || {}) });
  }

  async function apiGet(path) {
    const res = await fetch(path);
    const body = await res.json();
    if (!res.ok) {
      const msg = (body && body.detail && body.detail.message) || body.detail || "API request failed";
      throw new Error(msg);
    }
    return body;
  }

  async function apiPost(path, payload) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();
    if (!res.ok) {
      const msg = (body && body.detail && body.detail.message) || body.detail || "API request failed";
      throw new Error(msg);
    }
    return body;
  }

  /* =========================================================================
   * PAGE 1: DASHBOARD
   * ========================================================================= */
  function DashboardPage({ onInspectProtein }) {
    const [data, setData] = useState(null);
    const [err, setErr] = useState(null);

    useEffect(() => {
      apiGet("/api/dashboard").then(setData).catch((e) => setErr(e.message));
    }, []);

    if (err) return h("div", { className: "error-box" }, err);
    if (!data) return h("div", { className: "loading" }, "Loading dashboard metrics...");

    const k = data.kpis;
    const ns = data.network_statistics || {};
    const bins = ns.degree_distribution_bins || [];
    const hubBias = data.hub_bias || [];

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Research Dashboard"),
        h("p", null, "Human PPI Network Topology, Centrality-Driven Cancer Protein Prediction Performance & Top Candidates")
      ),
      h(
        "div",
        { className: "kpi-grid" },
        h("div", { className: "kpi-card" },
          h("div", { className: "kpi-label" }, "Total PPI Proteins"),
          h("div", { className: "kpi-value" }, (k.total_proteins || 0).toLocaleString()),
          h("div", { className: "kpi-sub" }, `GCC: ${((ns.giant_component_fraction || 0) * 100).toFixed(1)}% (${ns.connected_components || 0} components)`)
        ),
        h("div", { className: "kpi-card" },
          h("div", { className: "kpi-label" }, "PPI Interactions"),
          h("div", { className: "kpi-value" }, (k.total_interactions || 0).toLocaleString()),
          h("div", { className: "kpi-sub" }, `Avg Degree: ${(ns.average_degree || 0).toFixed(2)} | Density: ${(ns.network_density || 0).toFixed(5)}`)
        ),
        h("div", { className: "kpi-card" },
          h("div", { className: "kpi-label" }, "COSMIC Cancer Proteins"),
          h("div", { className: "kpi-value", style: { color: "#f87171" } }, (k.cancer_proteins || 0).toLocaleString()),
          h("div", { className: "kpi-sub" }, `Cancer Avg Degree: ${(ns.cancer_avg_degree || 0).toFixed(1)} vs Non-Cancer: ${(ns.non_cancer_avg_degree || 0).toFixed(1)}`)
        ),
        h("div", { className: "kpi-card" },
          h("div", { className: "kpi-label" }, "Best Model Test PR-AUC"),
          h("div", { className: "kpi-value", style: { color: "#34d399" } }, (k.best_pr_auc || 0).toFixed(4)),
          h("div", { className: "kpi-sub" }, `ROC-AUC: ${(k.best_roc_auc || 0).toFixed(4)} | F1: ${(k.best_f1 || 0).toFixed(4)}`)
        )
      ),
      h(
        "div",
        { className: "grid-2" },
        h(
          "div",
          { className: "card" },
          h("h2", null, "PPI Degree Distribution"),
          h(PlotlyChart, {
            data: [
              {
                x: bins.map((b) => b.range),
                y: bins.map((b) => b.count),
                type: "bar",
                marker: { color: "#3b82f6" },
              },
            ],
            layout: {
              title: { text: "Proteins by Degree Interval", font: { size: 13 } },
              xaxis: { title: "Degree Bin", gridcolor: "rgba(148,163,184,0.12)" },
              yaxis: { title: "Protein Count", gridcolor: "rgba(148,163,184,0.12)" },
            },
            style: { height: "320px" },
          })
        ),
        h(
          "div",
          { className: "card" },
          h("h2", null, "Hub-Bias Diagnostic (Performance by Degree Tier)"),
          h(PlotlyChart, {
            data: [
              {
                x: hubBias.map((b) => b.bucket),
                y: hubBias.map((b) => b.pr_auc),
                name: "PR-AUC",
                type: "bar",
                marker: { color: "#06b6d4" },
              },
              {
                x: hubBias.map((b) => b.bucket),
                y: hubBias.map((b) => b.roc_auc),
                name: "ROC-AUC",
                type: "bar",
                marker: { color: "#10b981" },
              },
            ],
            layout: {
              barmode: "group",
              title: { text: "Held-Out Test Performance Across Degree Buckets", font: { size: 13 } },
              yaxis: { range: [0, 1.05], title: "Score", gridcolor: "rgba(148,163,184,0.12)" },
            },
            style: { height: "320px" },
          })
        )
      ),
      h(
        "div",
        { className: "grid-2" },
        h(
          "div",
          { className: "card" },
          h("h2", null, "Top-10 Novel Candidate Proteins (Unannotated in COSMIC)"),
          h(
            "div",
            { className: "table-wrap" },
            h(
              "table",
              null,
              h("thead", null, h("tr", null, h("th", null, "#"), h("th", null, "Gene"), h("th", null, "Cancer Prob."), h("th", null, "Centrality Score"), h("th", null, "Degree"))),
              h(
                "tbody",
                null,
                (data.top_novel_candidates || []).map((row, idx) =>
                  h(
                    "tr",
                    { key: row.protein_id },
                    h("td", { className: "mono" }, idx + 1),
                    h("td", null, h("button", { className: "link-btn", onClick: () => onInspectProtein(row.gene_symbol) }, row.gene_symbol)),
                    h("td", { className: "mono", style: { color: "#fcd34d" } }, Number(row.cancer_association_probability).toFixed(4)),
                    h("td", { className: "mono" }, Number(row.composite_centrality_score).toFixed(4)),
                    h("td", { className: "mono" }, row.degree)
                  )
                )
              )
            )
          )
        ),
        h(
          "div",
          { className: "card" },
          h("h2", null, "Top-10 Highest-Centrality Network Hubs"),
          h(
            "div",
            { className: "table-wrap" },
            h(
              "table",
              null,
              h("thead", null, h("tr", null, h("th", null, "#"), h("th", null, "Gene"), h("th", null, "Status"), h("th", null, "Centrality Score"), h("th", null, "Cancer Prob."))),
              h(
                "tbody",
                null,
                (data.top_centrality_hubs || []).map((row, idx) =>
                  h(
                    "tr",
                    { key: row.protein_id },
                    h("td", { className: "mono" }, idx + 1),
                    h("td", null, h("button", { className: "link-btn", onClick: () => onInspectProtein(row.gene_symbol) }, row.gene_symbol)),
                    h("td", null, h("span", { className: row.is_known_cancer ? "badge badge-cancer" : "badge badge-noncancer" }, row.is_known_cancer ? "Known Cancer" : "Unlabeled")),
                    h("td", { className: "mono", style: { color: "#60a5fa" } }, Number(row.composite_centrality_score).toFixed(4)),
                    h("td", { className: "mono" }, Number(row.cancer_association_probability).toFixed(4))
                  )
                )
              )
            )
          )
        )
      )
    );
  }

  /* =========================================================================
   * PAGE 2: PROTEIN PREDICTION
   * ========================================================================= */
  function ProteinPredictionPage({ initialQuery, onJumpNetwork, onJumpExplain }) {
    const [query, setQuery] = useState(initialQuery || "TP53");
    const [suggestions, setSuggestions] = useState([]);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const runPredict = async (target) => {
      const q = (target || query || "").trim();
      if (!q) {
        setError("Please enter a valid HGNC gene symbol (e.g., TP53, BRCA1) or STRING Ensembl ID.");
        return;
      }
      setLoading(true);
      setError(null);
      setSuggestions([]);
      try {
        const res = await apiGet(`/api/proteins/${encodeURIComponent(q)}`);
        setResult(res);
        setQuery(res.gene_symbol || q);
        const hist = await apiGet("/api/proteins/history/recent?limit=8");
        setHistory(hist.history || []);
      } catch (e) {
        setResult(null);
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    useEffect(() => {
      runPredict(initialQuery || "TP53");
    }, [initialQuery]);

    const handleInputChange = async (val) => {
      setQuery(val);
      if (val.trim().length >= 2) {
        try {
          const data = await apiGet(`/api/proteins/search?q=${encodeURIComponent(val.trim())}&limit=8`);
          setSuggestions(data.results || []);
        } catch (_) {
          setSuggestions([]);
        }
      } else {
        setSuggestions([]);
      }
    };

    const centEntries = result ? Object.entries(result.centralities || {}) : [];

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Protein Cancer-Association Prediction"),
        h("p", null, "Search any human protein by HGNC gene symbol (e.g., TP53, BRCA1, EGFR, CDK2) or STRING Ensembl ID")
      ),
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "controls-row" },
          h(
            "div",
            { className: "search-box-wrap" },
            h("input", {
              type: "text",
              value: query,
              placeholder: "Enter gene symbol or STRING ID (e.g., TP53, BRCA1, 9606.ENSP00000269305)...",
              onChange: (e) => handleInputChange(e.target.value),
              onKeyDown: (e) => e.key === "Enter" && runPredict(query),
            }),
            suggestions.length > 0 &&
              h(
                "div",
                { className: "autocomplete-list" },
                suggestions.map((s) =>
                  h(
                    "div",
                    {
                      key: s.protein_id,
                      className: "autocomplete-item",
                      onClick: () => runPredict(s.gene_symbol),
                    },
                    h("span", null, h("strong", null, s.gene_symbol), " ", h("span", { style: { color: "#64748b", fontSize: "0.78rem" } }, s.protein_id)),
                    h("span", { className: "mono", style: { color: "#60a5fa" } }, `P=${s.cancer_probability.toFixed(3)}`)
                  )
                )
              )
          ),
          h("button", { className: "btn", onClick: () => runPredict(query) }, "Predict Protein"),
          h("button", { className: "btn btn-secondary", onClick: () => runPredict("BRCA1") }, "BRCA1"),
          h("button", { className: "btn btn-secondary", onClick: () => runPredict("EGFR") }, "EGFR"),
          h("button", { className: "btn btn-secondary", onClick: () => runPredict("INVALID_XYZ_999") }, "Test Invalid ID")
        ),
        error && h("div", { className: "error-box" }, error),
        loading && h("div", { className: "loading" }, "Querying protein model & neighborhood...")
      ),
      result &&
        h(
          "div",
          null,
          h("div", { className: "disclaimer" }, result.scientific_disclaimer),
          h(
            "div",
            { className: "kpi-grid" },
            h("div", { className: "kpi-card" },
              h("div", { className: "kpi-label" }, "Protein / Gene Symbol"),
              h("div", { className: "kpi-value" }, result.gene_symbol),
              h("div", { className: "kpi-sub mono" }, result.protein_id)
            ),
            h("div", { className: "kpi-card" },
              h("div", { className: "kpi-label" }, "Cancer Association Probability"),
              h("div", { className: "kpi-value", style: { color: result.prediction.cancer_association_probability >= 0.5 ? "#f87171" : "#34d399" } },
                `${(result.prediction.cancer_association_probability * 100).toFixed(2)}%`
              ),
              h("div", { className: "kpi-sub" }, `Tier: ${result.prediction.confidence_tier} | Rank #${result.prediction.model_prediction_rank.toLocaleString()} / ${result.prediction.total_proteins.toLocaleString()}`)
            ),
            h("div", { className: "kpi-card" },
              h("div", { className: "kpi-label" }, "Network Importance (Centrality)"),
              h("div", { className: "kpi-value", style: { color: "#60a5fa" } }, result.network_importance.composite_centrality_score.toFixed(4)),
              h("div", { className: "kpi-sub" }, `Centrality Rank #${result.network_importance.centrality_rank.toLocaleString()} | Degree: ${result.network_importance.degree}`)
            ),
            h("div", { className: "kpi-card" },
              h("div", { className: "kpi-label" }, "COSMIC Reference Status"),
              h("div", { style: { marginTop: "8px" } },
                h("span", { className: result.known_label.is_known_cancer ? "badge badge-cancer" : "badge badge-novel" },
                  result.known_label.is_known_cancer ? `Known Cancer (Tier ${result.known_label.cosmic_tier})` : "Unannotated in COSMIC"
                )
              ),
              h("div", { className: "kpi-sub" }, `Role: ${result.known_label.role_in_cancer} | Split: ${result.known_label.dataset_split}`)
            )
          ),
          h(
            "div",
            { className: "card" },
            h("div", { className: "controls-row", style: { justifyContent: "space-between", marginBottom: "8px" } },
              h("h2", { style: { marginBottom: 0 } }, "Functional Annotation & Quick Actions"),
              h("div", { style: { display: "flex", gap: "8px" } },
                h("button", { className: "btn btn-secondary", onClick: () => onJumpNetwork(result.gene_symbol) }, "Explore in PPI Graph"),
                h("button", { className: "btn btn-secondary", onClick: () => onJumpExplain(result.gene_symbol) }, "View SHAP Explanation")
              )
            ),
            h("p", { style: { color: "#cbd5e1", fontSize: "0.88rem" } }, result.annotation)
          ),
          h(
            "div",
            { className: "grid-2" },
            h(
              "div",
              { className: "card" },
              h("h2", null, "8-Centrality Percentile Profile"),
              h(PlotlyChart, {
                data: [
                  {
                    type: "bar",
                    orientation: "h",
                    y: centEntries.map(([k]) => k.replace(/_/g, " ")),
                    x: centEntries.map(([, v]) => v.percentile),
                    text: centEntries.map(([, v]) => `${v.percentile.toFixed(1)}% (val=${v.value})`),
                    textposition: "outside",
                    marker: { color: "#06b6d4" },
                  },
                ],
                layout: {
                  xaxis: { range: [0, 118], title: "Network Percentile (%)", gridcolor: "rgba(148,163,184,0.12)" },
                  margin: { t: 20, r: 24, b: 44, l: 155 },
                },
                style: { height: "340px" },
              })
            ),
            h(
              "div",
              { className: "card" },
              h("h2", null, `Top Interacting PPI Partners (${result.network_importance.known_cancer_neighbors_count} known cancer / ${result.network_importance.total_neighbors_count} total)`),
              h(
                "div",
                { className: "table-wrap", style: { maxHeight: "340px", overflowY: "auto" } },
                h(
                  "table",
                  null,
                  h("thead", null, h("tr", null, h("th", null, "Partner"), h("th", null, "Score"), h("th", null, "Status"), h("th", null, "Cancer Prob."), h("th", null, "Degree"))),
                  h(
                    "tbody",
                    null,
                    (result.neighbors || []).map((n) =>
                      h(
                        "tr",
                        { key: n.protein_id },
                        h("td", null, h("button", { className: "link-btn", onClick: () => runPredict(n.gene_symbol) }, n.gene_symbol)),
                        h("td", { className: "mono" }, n.combined_score),
                        h("td", null, h("span", { className: n.is_known_cancer ? "badge badge-cancer" : "badge badge-noncancer" }, n.is_known_cancer ? "Cancer" : "Unlabeled")),
                        h("td", { className: "mono" }, n.cancer_probability.toFixed(4)),
                        h("td", { className: "mono" }, n.degree)
                      )
                    )
                  )
                )
              )
            )
          ),
          history.length > 0 &&
            h(
              "div",
              { className: "card" },
              h("h3", null, "Recent Prediction Queries (SQLite Persisted)"),
              h(
                "div",
                { style: { display: "flex", flexWrap: "wrap", gap: "8px" } },
                history.map((item, i) =>
                  h(
                    "button",
                    { key: i, className: "btn btn-secondary", style: { padding: "6px 12px", fontSize: "0.8rem" }, onClick: () => runPredict(item.gene_symbol) },
                    `${item.gene_symbol} (${(item.cancer_probability * 100).toFixed(1)}%)`
                  )
                )
              )
            )
        )
    );
  }

  /* =========================================================================
   * PAGE 3: CENTRALITY ANALYSIS
   * ========================================================================= */
  function CentralityAnalysisPage({ onInspectProtein }) {
    const [metric, setMetric] = useState("composite");
    const [filterCancer, setFilterCancer] = useState("all");
    const [limit, setLimit] = useState(20);
    const [data, setData] = useState(null);
    const [weights, setWeights] = useState({ degree: 0.2, betweenness: 0.2, closeness: 0.2, eigenvector: 0.2, pagerank: 0.2 });
    const [customResult, setCustomResult] = useState(null);

    useEffect(() => {
      apiGet(`/api/centrality/rankings?metric=${metric}&limit=${limit}&filter_cancer=${filterCancer}`).then((d) => {
        setData(d);
        if (d.validated_weights && d.validated_weights.weights && !customResult) {
          setWeights(d.validated_weights.weights);
        }
      });
    }, [metric, filterCancer, limit]);

    const applyCustomWeights = async () => {
      const res = await apiPost("/api/centrality/custom-score", {
        ...weights,
        limit: Number(limit),
        only_novel: filterCancer === "novel",
      });
      setCustomResult(res);
    };

    const resetValidatedWeights = () => {
      if (data && data.validated_weights && data.validated_weights.weights) {
        setWeights(data.validated_weights.weights);
        setCustomResult(null);
      }
    };

    const rows = data ? data.rows : [];
    const valOpt = (data && data.validated_weights) || {};

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Centrality Analysis & Configurable Ranking"),
        h("p", null, "Compare proteins across all centrality metrics and test validation-optimized vs custom composite centrality weights")
      ),
      h(
        "div",
        { className: "card" },
        h("h2", null, "Configurable Composite Centrality Score"),
        valOpt.weights &&
          h("p", { style: { fontSize: "0.84rem", color: "#94a3b8", marginBottom: "14px" } },
            `Validation-Optimized Weights PR-AUC: ${valOpt.optimized_val_pr_auc} (vs Equal Weights: ${valOpt.equal_weights_val_pr_auc}). Adjust sliders to experiment:`
          ),
        h(
          "div",
          { className: "slider-grid" },
          ["degree", "betweenness", "closeness", "eigenvector", "pagerank"].map((k) =>
            h(
              "div",
              { className: "slider-item", key: k },
              h("label", null, h("span", null, k.toUpperCase()), h("span", { className: "mono" }, Number(weights[k] || 0).toFixed(2))),
              h("input", {
                type: "range",
                min: "0",
                max: "1",
                step: "0.05",
                value: weights[k] || 0,
                onChange: (e) => setWeights(Object.assign({}, weights, { [k]: parseFloat(e.target.value) })),
              })
            )
          )
        ),
        h(
          "div",
          { className: "controls-row" },
          h("button", { className: "btn", onClick: applyCustomWeights }, "Compute Custom Composite Ranking"),
          h("button", { className: "btn btn-secondary", onClick: resetValidatedWeights }, "Reset to Validation-Optimized Weights")
        ),
        customResult &&
          h(
            "div",
            { style: { marginTop: "12px" } },
            h("p", { style: { fontSize: "0.82rem", color: "#6ee7b7", marginBottom: "8px" } },
              `Normalized Weights: ${Object.entries(customResult.normalized_weights).map(([k, v]) => `${k}=${v}`).join(" | ")}`
            ),
            h(
              "div",
              { className: "table-wrap", style: { maxHeight: "260px", overflowY: "auto" } },
              h(
                "table",
                null,
                h("thead", null, h("tr", null, h("th", null, "Rank"), h("th", null, "Gene"), h("th", null, "Status"), h("th", null, "Custom Centrality"), h("th", null, "Model Cancer Prob."), h("th", null, "Degree"))),
                h(
                  "tbody",
                  null,
                  customResult.rows.map((r) =>
                    h(
                      "tr",
                      { key: r.protein_id },
                      h("td", { className: "mono" }, r.rank),
                      h("td", null, h("button", { className: "link-btn", onClick: () => onInspectProtein(r.gene_symbol) }, r.gene_symbol)),
                      h("td", null, h("span", { className: r.is_known_cancer ? "badge badge-cancer" : "badge badge-novel" }, r.is_known_cancer ? "Known Cancer" : "Novel Candidate")),
                      h("td", { className: "mono", style: { color: "#22d3ee" } }, r.custom_centrality_score.toFixed(4)),
                      h("td", { className: "mono" }, r.cancer_probability.toFixed(4)),
                      h("td", { className: "mono" }, r.degree)
                    )
                  )
                )
              )
            )
          )
      ),
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "controls-row" },
          h("select", { value: metric, onChange: (e) => setMetric(e.target.value), style: { maxWidth: "250px" } },
            h("option", { value: "composite" }, "Composite Centrality Score"),
            h("option", { value: "model_probability" }, "Model Cancer Probability"),
            h("option", { value: "degree" }, "Degree Centrality"),
            h("option", { value: "betweenness" }, "Betweenness Centrality"),
            h("option", { value: "closeness" }, "Closeness Centrality"),
            h("option", { value: "eigenvector" }, "Eigenvector Centrality"),
            h("option", { value: "pagerank" }, "PageRank"),
            h("option", { value: "k_core" }, "K-Core Centrality"),
            h("option", { value: "clustering" }, "Clustering Coefficient"),
            h("option", { value: "constraint" }, "Network Constraint (Lowest First)")
          ),
          h("select", { value: filterCancer, onChange: (e) => setFilterCancer(e.target.value), style: { maxWidth: "200px" } },
            h("option", { value: "all" }, "All Proteins"),
            h("option", { value: "novel" }, "Novel Candidates Only"),
            h("option", { value: "known" }, "Known Cancer Only")
          ),
          h("select", { value: limit, onChange: (e) => setLimit(Number(e.target.value)), style: { maxWidth: "150px" } },
            h("option", { value: 10 }, "Top 10"),
            h("option", { value: 20 }, "Top 20"),
            h("option", { value: 50 }, "Top 50")
          )
        ),
        rows.length > 0 &&
          h(PlotlyChart, {
            data: [
              {
                x: rows.slice(0, 20).map((r) => r.gene_symbol),
                y: rows.slice(0, 20).map((r) => r[data.column]),
                type: "bar",
                marker: { color: rows.slice(0, 20).map((r) => (r.is_known_cancer ? "#ef4444" : "#3b82f6")) },
              },
            ],
            layout: {
              title: { text: `Top 20 Proteins by ${metric.toUpperCase()} (Red = Known COSMIC Cancer, Blue = Unlabeled)`, font: { size: 13 } },
              yaxis: { title: metric },
            },
            style: { height: "310px", marginBottom: "16px" },
          }),
        h(
          "div",
          { className: "table-wrap" },
          h(
            "table",
            null,
            h("thead", null, h("tr", null,
              h("th", null, "Rank"), h("th", null, "Gene"), h("th", null, "Status"),
              h("th", null, "Degree"), h("th", null, "Betweenness"), h("th", null, "Closeness"),
              h("th", null, "Eigenvector"), h("th", null, "PageRank"), h("th", null, "K-Core"),
              h("th", null, "Composite"), h("th", null, "Cancer Prob.")
            )),
            h(
              "tbody",
              null,
              rows.map((r) =>
                h(
                  "tr",
                  { key: r.protein_id },
                  h("td", { className: "mono" }, r.rank),
                  h("td", null, h("button", { className: "link-btn", onClick: () => onInspectProtein(r.gene_symbol) }, r.gene_symbol)),
                  h("td", null, h("span", { className: r.is_known_cancer ? "badge badge-cancer" : "badge badge-noncancer" }, r.is_known_cancer ? "Cancer" : "Unlabeled")),
                  h("td", { className: "mono" }, r.degree),
                  h("td", { className: "mono" }, r.betweenness_centrality.toFixed(5)),
                  h("td", { className: "mono" }, r.closeness_centrality.toFixed(4)),
                  h("td", { className: "mono" }, r.eigenvector_centrality.toFixed(5)),
                  h("td", { className: "mono" }, r.pagerank.toFixed(5)),
                  h("td", { className: "mono" }, r.k_core),
                  h("td", { className: "mono", style: { color: "#60a5fa" } }, r.composite_centrality_score.toFixed(4)),
                  h("td", { className: "mono", style: { color: "#fcd34d" } }, r.cancer_probability.toFixed(4))
                )
              )
            )
          )
        )
      )
    );
  }

  /* =========================================================================
   * PAGE 4: PPI NETWORK VISUALIZATION
   * ========================================================================= */
  function NetworkPage({ initialProtein, onInspectProtein }) {
    const [mode, setMode] = useState(initialProtein ? "ego" : "hubs");
    const [centerQuery, setCenterQuery] = useState(initialProtein || "TP53");
    const [maxNodes, setMaxNodes] = useState(35);
    const [graphData, setGraphData] = useState(null);
    const [selectedNode, setSelectedNode] = useState(null);
    const [error, setError] = useState(null);

    const loadGraph = async (m, q, lim) => {
      setError(null);
      try {
        const res =
          m === "ego"
            ? await apiGet(`/api/network/ego/${encodeURIComponent(q)}?max_nodes=${lim}`)
            : await apiGet(`/api/network/top-hubs?limit=${lim}`);
        setGraphData(res);
      } catch (e) {
        setError(e.message);
      }
    };

    useEffect(() => {
      loadGraph(mode, centerQuery, maxNodes);
    }, [mode, maxNodes]);

    const buildPlotData = () => {
      if (!graphData) return [];
      const nodeMap = {};
      graphData.nodes.forEach((n) => (nodeMap[n.id] = n));

      const edgeX = [];
      const edgeY = [];
      graphData.edges.forEach((e) => {
        const s = nodeMap[e.source];
        const t = nodeMap[e.target];
        if (s && t) {
          edgeX.push(s.x, t.x, null);
          edgeY.push(s.y, t.y, null);
        }
      });

      const edgeTrace = {
        x: edgeX,
        y: edgeY,
        mode: "lines",
        line: { width: 0.9, color: "rgba(148,163,184,0.28)" },
        hoverinfo: "none",
        name: "PPI Interactions",
      };

      const groups = {
        known: { name: "Known COSMIC Cancer", color: "#ef4444", nodes: [] },
        candidate: { name: "High-Prob Novel Candidate (P >= 0.45)", color: "#f59e0b", nodes: [] },
        other: { name: "Other / Low-Prob Protein", color: "#3b82f6", nodes: [] },
      };

      graphData.nodes.forEach((n) => {
        if (n.is_known_cancer) groups.known.nodes.push(n);
        else if (n.cancer_probability >= 0.45) groups.candidate.nodes.push(n);
        else groups.other.nodes.push(n);
      });

      const traces = [edgeTrace];
      Object.values(groups).forEach((g) => {
        if (!g.nodes.length) return;
        traces.push({
          x: g.nodes.map((n) => n.x),
          y: g.nodes.map((n) => n.y),
          text: g.nodes.map((n) => n.gene_symbol),
          customdata: g.nodes,
          hovertemplate:
            "<b>%{text}</b><br>Cancer Prob: %{customdata.cancer_probability:.4f}<br>" +
            "Centrality: %{customdata.composite_centrality_score:.4f}<br>Degree: %{customdata.degree}<extra></extra>",
          mode: "markers+text",
          textposition: "top center",
          textfont: { size: 10, color: "#e2e8f0" },
          marker: {
            color: g.color,
            size: g.nodes.map((n) => (n.is_center ? 24 : Math.max(10, Math.min(22, 8 + Math.sqrt(n.degree) * 0.7)))),
            line: { width: g.nodes.map((n) => (n.is_center ? 3 : 1.2)), color: "#ffffff" },
          },
          name: g.name,
        });
      });
      return traces;
    };

    const handleNodeClick = (evt) => {
      if (evt && evt.points && evt.points[0] && evt.points[0].customdata) {
        const node = evt.points[0].customdata;
        setSelectedNode(node);
      }
    };

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Interactive PPI Network Visualization"),
        h("p", null, "Explore high-centrality hubs or inspect any protein's 1-hop interaction neighborhood. Click any node to inspect or re-center.")
      ),
      h(
        "div",
        { className: "card" },
        h(
          "div",
          { className: "controls-row" },
          h("button", { className: mode === "hubs" ? "btn" : "btn btn-secondary", onClick: () => setMode("hubs") }, "Top Centrality Hubs"),
          h("button", { className: mode === "ego" ? "btn" : "btn btn-secondary", onClick: () => { setMode("ego"); loadGraph("ego", centerQuery, maxNodes); } }, "Protein Ego-Network"),
          h("input", {
            type: "text",
            value: centerQuery,
            style: { maxWidth: "230px" },
            placeholder: "Gene symbol (e.g., TP53)",
            onChange: (e) => setCenterQuery(e.target.value),
            onKeyDown: (e) => e.key === "Enter" && (setMode("ego"), loadGraph("ego", centerQuery, maxNodes)),
          }),
          h("button", { className: "btn", onClick: () => { setMode("ego"); loadGraph("ego", centerQuery, maxNodes); } }, "Focus Protein"),
          h("select", { value: maxNodes, onChange: (e) => setMaxNodes(Number(e.target.value)), style: { maxWidth: "150px" } },
            h("option", { value: 25 }, "25 Nodes"),
            h("option", { value: 35 }, "35 Nodes"),
            h("option", { value: 55 }, "55 Nodes")
          )
        ),
        error && h("div", { className: "error-box" }, error),
        graphData &&
          h(PlotlyChart, {
            data: buildPlotData(),
            layout: {
              title: { text: mode === "ego" ? `1-Hop PPI Ego-Network centered on ${centerQuery.toUpperCase()}` : "Top High-Centrality PPI Hub Subgraph", font: { size: 13 } },
              showlegend: true,
              legend: { orientation: "h", y: -0.05 },
              xaxis: { showgrid: false, zeroline: false, showticklabels: false },
              yaxis: { showgrid: false, zeroline: false, showticklabels: false },
              margin: { t: 40, r: 16, b: 36, l: 16 },
              hovermode: "closest",
            },
            style: { height: "540px" },
            onPointClick: handleNodeClick,
          }),
        selectedNode &&
          h(
            "div",
            { style: { marginTop: "14px", padding: "14px", background: "#162033", borderRadius: "10px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" } },
            h("div", null,
              h("strong", { style: { fontSize: "1rem" } }, selectedNode.gene_symbol),
              " ",
              h("span", { className: "mono", style: { color: "#94a3b8", fontSize: "0.82rem" } }, selectedNode.id),
              h("div", { style: { fontSize: "0.85rem", color: "#cbd5e1", marginTop: "4px" } },
                `Cancer Prob: ${(selectedNode.cancer_probability * 100).toFixed(2)}% | Centrality: ${selectedNode.composite_centrality_score.toFixed(4)} | Degree: ${selectedNode.degree}`
              )
            ),
            h("div", { style: { display: "flex", gap: "8px" } },
              h("button", { className: "btn", onClick: () => { setCenterQuery(selectedNode.gene_symbol); setMode("ego"); loadGraph("ego", selectedNode.gene_symbol, maxNodes); } }, "Re-Center Neighborhood"),
              h("button", { className: "btn btn-secondary", onClick: () => onInspectProtein(selectedNode.gene_symbol) }, "Full Prediction & SHAP")
            )
          )
      )
    );
  }

  /* =========================================================================
   * PAGE 5: MODEL COMPARISON
   * ========================================================================= */
  function ModelComparisonPage() {
    const [data, setData] = useState(null);
    const [cmModel, setCmModel] = useState("");

    useEffect(() => {
      apiGet("/api/models/comparison").then((d) => {
        setData(d);
        setCmModel(d.best_model || (d.comparison_table[0] && d.comparison_table[0].model) || "");
      });
    }, []);

    if (!data) return h("div", { className: "loading" }, "Loading model evaluation metrics...");

    const detail = data.models_detail || {};
    const rocTraces = Object.entries(detail).map(([name, m]) => ({
      x: m.test_metrics.roc_curve.fpr,
      y: m.test_metrics.roc_curve.tpr,
      mode: "lines",
      name: `${name} (${m.test_metrics.roc_auc.toFixed(3)})`,
    }));
    const prTraces = Object.entries(detail).map(([name, m]) => ({
      x: m.test_metrics.pr_curve.recall,
      y: m.test_metrics.pr_curve.precision,
      mode: "lines",
      name: `${name} (${m.test_metrics.pr_auc.toFixed(3)})`,
    }));

    const ab = data.ablation_table || [];
    const selectedCm = detail[cmModel] ? detail[cmModel].test_metrics.confusion_matrix : null;

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Model Comparison & 5-Way Ablation Study"),
        h("p", null, "Held-out test evaluation across reference heuristic, ML baselines, PyTorch GNNs, and Centrality-Driven Graph Fusion")
      ),
      h(
        "div",
        { className: "card" },
        h("h2", null, "All Trained Models — Held-Out Test Performance (Prioritizing PR-AUC)"),
        h(
          "div",
          { className: "table-wrap" },
          h(
            "table",
            null,
            h("thead", null, h("tr", null,
              h("th", null, "Model"), h("th", null, "Features"), h("th", null, "Val PR-AUC"),
              h("th", null, "Test PR-AUC"), h("th", null, "Test ROC-AUC"), h("th", null, "F1"),
              h("th", null, "Precision"), h("th", null, "Recall"), h("th", null, "Accuracy"), h("th", null, "Brier")
            )),
            h(
              "tbody",
              null,
              (data.comparison_table || []).map((r) =>
                h(
                  "tr",
                  { key: r.model },
                  h("td", null, h("strong", null, r.model)),
                  h("td", { style: { color: "#94a3b8", fontSize: "0.78rem" } }, r.feature_set),
                  h("td", { className: "mono" }, Number(r.val_pr_auc).toFixed(4)),
                  h("td", { className: "mono", style: { color: "#34d399", fontWeight: 700 } }, Number(r.test_pr_auc).toFixed(4)),
                  h("td", { className: "mono", style: { color: "#60a5fa" } }, Number(r.test_roc_auc).toFixed(4)),
                  h("td", { className: "mono" }, Number(r.test_f1_score).toFixed(4)),
                  h("td", { className: "mono" }, Number(r.test_precision).toFixed(4)),
                  h("td", { className: "mono" }, Number(r.test_recall).toFixed(4)),
                  h("td", { className: "mono" }, Number(r.test_accuracy).toFixed(4)),
                  h("td", { className: "mono" }, Number(r.test_brier_score).toFixed(4))
                )
              )
            )
          )
        )
      ),
      h(
        "div",
        { className: "grid-2" },
        h(
          "div",
          { className: "card" },
          h("h2", null, "Precision-Recall Curves (Primary Minority-Class Metric)"),
          h(PlotlyChart, {
            data: prTraces,
            layout: {
              xaxis: { title: "Recall", range: [0, 1] },
              yaxis: { title: "Precision", range: [0, 1.02] },
              legend: { font: { size: 10 } },
            },
            style: { height: "390px" },
          })
        ),
        h(
          "div",
          { className: "card" },
          h("h2", null, "ROC Curves"),
          h(PlotlyChart, {
            data: rocTraces.concat([{ x: [0, 1], y: [0, 1], mode: "lines", line: { dash: "dash", color: "#64748b" }, name: "Random" }]),
            layout: {
              xaxis: { title: "False Positive Rate", range: [0, 1] },
              yaxis: { title: "True Positive Rate", range: [0, 1.02] },
              legend: { font: { size: 10 } },
            },
            style: { height: "390px" },
          })
        )
      ),
      h(
        "div",
        { className: "grid-2" },
        h(
          "div",
          { className: "card" },
          h("h2", null, "5-Way Feature Ablation Study (A–E)"),
          h(PlotlyChart, {
            data: [
              { x: ab.map((a) => a.ablation_id), y: ab.map((a) => a.test_pr_auc), name: "Test PR-AUC", type: "bar", marker: { color: "#3b82f6" } },
              { x: ab.map((a) => a.ablation_id), y: ab.map((a) => a.test_roc_auc), name: "Test ROC-AUC", type: "bar", marker: { color: "#10b981" } },
              { x: ab.map((a) => a.ablation_id), y: ab.map((a) => a.test_f1_score), name: "Test F1", type: "bar", marker: { color: "#f59e0b" } },
            ],
            layout: { barmode: "group", yaxis: { range: [0, 1.05], title: "Score" } },
            style: { height: "350px" },
          })
        ),
        h(
          "div",
          { className: "card" },
          h("div", { className: "controls-row", style: { justifyContent: "space-between" } },
            h("h2", { style: { marginBottom: 0 } }, "Confusion Matrix Inspector"),
            h("select", { value: cmModel, onChange: (e) => setCmModel(e.target.value), style: { maxWidth: "280px" } },
              Object.keys(detail).map((k) => h("option", { key: k, value: k }, k))
            )
          ),
          selectedCm &&
            h(PlotlyChart, {
              data: [
                {
                  z: [
                    [selectedCm.tn, selectedCm.fp],
                    [selectedCm.fn, selectedCm.tp],
                  ],
                  x: ["Predicted Non-Cancer", "Predicted Cancer"],
                  y: ["Actual Non-Cancer", "Actual Cancer"],
                  type: "heatmap",
                  colorscale: "Blues",
                  text: [
                    [`TN: ${selectedCm.tn}`, `FP: ${selectedCm.fp}`],
                    [`FN: ${selectedCm.fn}`, `TP: ${selectedCm.tp}`],
                  ],
                  texttemplate: "<b>%{text}</b>",
                  textfont: { size: 15 },
                  showscale: false,
                },
              ],
              layout: {
                title: { text: cmModel, font: { size: 12 } },
                yaxis: { autorange: "reversed" },
              },
              style: { height: "330px" },
            })
        )
      )
    );
  }

  /* =========================================================================
   * PAGE 6: EXPLAINABILITY (SHAP)
   * ========================================================================= */
  function ExplainabilityPage({ initialProtein }) {
    const [globalData, setGlobalData] = useState(null);
    const [proteinQuery, setProteinQuery] = useState(initialProtein || "TP53");
    const [localData, setLocalData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
      apiGet("/api/explainability/global?top_n=20").then(setGlobalData);
    }, []);

    const fetchLocal = async (q) => {
      setError(null);
      try {
        const res = await apiGet(`/api/explainability/protein/${encodeURIComponent(q.trim())}`);
        setLocalData(res);
      } catch (e) {
        setLocalData(null);
        setError(e.message);
      }
    };

    useEffect(() => {
      fetchLocal(initialProtein || "TP53");
    }, [initialProtein]);

    const catColors = {
      Centrality: "#3b82f6",
      "Functional Annotation": "#10b981",
      "Graph Neighborhood / Topology": "#8b5cf6",
      "GNN Embedding": "#f59e0b",
    };

    const topFeats = globalData ? globalData.top_features.slice(0, 18).reverse() : [];
    const cats = globalData ? globalData.category_importance_pct : [];
    const contribs = localData && localData.shap_explanation ? localData.shap_explanation.top_contributors.slice().reverse() : [];

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Explainable AI (SHAP Feature Attributions)"),
        h("p", null, "Understand which centrality, functional, and graph features drive global and individual protein predictions")
      ),
      globalData && h("div", { className: "disclaimer" }, globalData.disclaimer),
      globalData &&
        h(
          "div",
          { className: "grid-2" },
          h(
            "div",
            { className: "card" },
            h("h2", null, `Global Top-18 SHAP Feature Importance (${globalData.method})`),
            h(PlotlyChart, {
              data: [
                {
                  type: "bar",
                  orientation: "h",
                  y: topFeats.map((f) => f.feature),
                  x: topFeats.map((f) => f.mean_abs_shap),
                  marker: { color: topFeats.map((f) => catColors[f.category] || "#64748b") },
                },
              ],
              layout: {
                xaxis: { title: "Mean |SHAP Value|" },
                margin: { t: 20, r: 20, b: 44, l: 175 },
              },
              style: { height: "420px" },
            })
          ),
          h(
            "div",
            { className: "card" },
            h("h2", null, "Feature Category Share of Total |SHAP|"),
            h(PlotlyChart, {
              data: [
                {
                  type: "bar",
                  x: cats.map((c) => c.category),
                  y: cats.map((c) => c.importance_share_pct),
                  text: cats.map((c) => `${Number(c.importance_share_pct).toFixed(1)}%`),
                  textposition: "outside",
                  marker: { color: cats.map((c) => catColors[c.category] || "#64748b") },
                },
              ],
              layout: { yaxis: { title: "Share of Total |SHAP| (%)", range: [0, 100] } },
              style: { height: "420px" },
            })
          )
        ),
      h(
        "div",
        { className: "card" },
        h("h2", null, "Individual Protein Local SHAP Explanation"),
        h(
          "div",
          { className: "controls-row" },
          h("input", {
            type: "text",
            value: proteinQuery,
            style: { maxWidth: "280px" },
            placeholder: "Enter gene symbol (e.g., TP53, BRCA1)...",
            onChange: (e) => setProteinQuery(e.target.value),
            onKeyDown: (e) => e.key === "Enter" && fetchLocal(proteinQuery),
          }),
          h("button", { className: "btn", onClick: () => fetchLocal(proteinQuery) }, "Explain Protein")
        ),
        error && h("div", { className: "error-box" }, error),
        localData &&
          h(
            "div",
            null,
            h("p", { style: { fontSize: "0.88rem", color: "#cbd5e1", marginBottom: "10px" } },
              `${localData.gene_symbol} (${localData.protein_id}) — Cancer Prob: ${(localData.prediction.cancer_association_probability * 100).toFixed(2)}% | ` +
              `Centrality Score: ${localData.network_importance.composite_centrality_score.toFixed(4)} | ` +
              `Known Cancer Neighbors: ${localData.network_importance.known_cancer_neighbors_count} / ${localData.network_importance.total_neighbors_count}`
            ),
            h(PlotlyChart, {
              data: [
                {
                  type: "bar",
                  orientation: "h",
                  y: contribs.map((c) => `${c.feature} (${c.category})`),
                  x: contribs.map((c) => c.shap_value),
                  marker: { color: contribs.map((c) => (c.shap_value >= 0 ? "#ef4444" : "#3b82f6")) },
                },
              ],
              layout: {
                title: { text: `Local SHAP Contributions for ${localData.gene_symbol} (Red = increases cancer score, Blue = decreases)`, font: { size: 13 } },
                xaxis: { title: "SHAP Value (Log-Odds Impact)" },
                margin: { t: 40, r: 24, b: 46, l: 260 },
              },
              style: { height: "380px" },
            })
          )
      )
    );
  }

  /* =========================================================================
   * PAGE 7: DATASET INFORMATION
   * ========================================================================= */
  function DatasetInfoPage() {
    const [data, setData] = useState(null);
    useEffect(() => {
      apiGet("/api/dataset").then(setData);
    }, []);
    if (!data) return h("div", { className: "loading" }, "Loading dataset provenance...");

    const s = data.statistics || {};
    const p = data.preprocessing || {};
    const c = data.centrality_computation || {};

    return h(
      "div",
      null,
      h(
        "div",
        { className: "page-header" },
        h("h1", null, "Dataset Provenance & Preprocessing Information"),
        h("p", null, "Transparent biological data sources, attribution, cleaning pipeline, and anti-leakage safeguards")
      ),
      h(
        "div",
        { className: "card" },
        h("h2", null, `Active Dataset: ${data.dataset_name}`),
        h(
          "div",
          { className: "table-wrap" },
          h(
            "table",
            null,
            h(
              "tbody",
              null,
              Object.entries(data.sources || {}).map(([k, v]) =>
                h("tr", { key: k }, h("td", { style: { fontWeight: 600, width: "260px" } }, k.replace(/_/g, " ").toUpperCase()), h("td", null, String(v)))
              )
            )
          )
        )
      ),
      h(
        "div",
        { className: "grid-2" },
        h(
          "div",
          { className: "card" },
          h("h2", null, "Network & Label Statistics"),
          h(
            "table",
            null,
            h(
              "tbody",
              null,
              Object.entries(s).map(([k, v]) =>
                h("tr", { key: k }, h("td", null, k.replace(/_/g, " ")), h("td", { className: "mono" }, String(v)))
              )
            )
          )
        ),
        h(
          "div",
          { className: "card" },
          h("h2", null, "Preprocessing & Anti-Leakage Report"),
          h(
            "table",
            null,
            h(
              "tbody",
              null,
              [
                ["Initial Filtered Edge Rows", p.initial_filtered_edge_rows],
                ["Self-Loops Removed", p.self_loops_removed],
                ["Duplicate Reverse Edges Removed", p.duplicate_reverse_edges_removed],
                ["Final Undirected Edges", p.final_undirected_edges],
                ["Missing Functional Annotations", p.missing_functional_annotations],
                ["Cancer-Leak Text Terms Scrubbed", String(p.cancer_leak_terms_stripped)],
                ["Betweenness Computation Method", c.betweenness_method],
                ["Closeness Computation Method", c.closeness_method],
              ].map(([k, v]) => h("tr", { key: k }, h("td", null, k), h("td", { className: "mono" }, String(v ?? "N/A"))))
            )
          )
        )
      )
    );
  }

  /* =========================================================================
   * ROOT APP SHELL
   * ========================================================================= */
  const NAV_ITEMS = [
    { id: "dashboard", label: "Dashboard" },
    { id: "predict", label: "Protein Prediction" },
    { id: "centrality", label: "Centrality Analysis" },
    { id: "network", label: "PPI Network Graph" },
    { id: "models", label: "Model Comparison" },
    { id: "explain", label: "Explainability (SHAP)" },
    { id: "dataset", label: "Dataset Information" },
  ];

  function App() {
    const [page, setPage] = useState("dashboard");
    const [focusProtein, setFocusProtein] = useState("TP53");

    const jumpToPredict = (sym) => {
      setFocusProtein(sym);
      setPage("predict");
    };
    const jumpToNetwork = (sym) => {
      setFocusProtein(sym);
      setPage("network");
    };
    const jumpToExplain = (sym) => {
      setFocusProtein(sym);
      setPage("explain");
    };

    return h(
      "div",
      { className: "app-shell" },
      h(
        "aside",
        { className: "sidebar" },
        h(
          "div",
          { className: "brand" },
          h("div", { className: "brand-title" }, "OncoGraph-PPI"),
          h("div", { className: "brand-sub" }, "Centrality-Driven Cancer Protein Prediction")
        ),
        h(
          "ul",
          { className: "nav-list" },
          NAV_ITEMS.map((item) =>
            h(
              "li",
              { key: item.id },
              h(
                "button",
                {
                  className: "nav-btn" + (page === item.id ? " active" : ""),
                  onClick: () => setPage(item.id),
                },
                item.label
              )
            )
          )
        ),
        h(
          "div",
          { className: "sidebar-footer" },
          "STRING v12.5 + COSMIC v104 | Leak-Free Graph ML & SHAP"
        )
      ),
      h(
        "main",
        { className: "main-content" },
        page === "dashboard" && h(DashboardPage, { onInspectProtein: jumpToPredict }),
        page === "predict" && h(ProteinPredictionPage, { initialQuery: focusProtein, onJumpNetwork: jumpToNetwork, onJumpExplain: jumpToExplain }),
        page === "centrality" && h(CentralityAnalysisPage, { onInspectProtein: jumpToPredict }),
        page === "network" && h(NetworkPage, { initialProtein: focusProtein, onInspectProtein: jumpToPredict }),
        page === "models" && h(ModelComparisonPage),
        page === "explain" && h(ExplainabilityPage, { initialProtein: focusProtein }),
        page === "dataset" && h(DatasetInfoPage)
      )
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(h(App));
})();
