import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const DATA_ROOT = "/data/processed";
const colors = { green: "#335c67", lime: "#e09f3e", orange: "#9e2a2b", blue: "#540b0e", ink: "#540b0e", muted: "#335c67", line: "rgba(84, 11, 14, .22)", pale: "#fff3b0" };
const number = new Intl.NumberFormat("ru-RU");
const percent = d3.format(".1%");
const tooltip = d3.select("#tooltip");

function showTooltip(event, html) {
  tooltip.html(html).style("left", `${event.clientX}px`).style("top", `${event.clientY}px`).classed("visible", true).attr("aria-hidden", "false");
}

function hideTooltip() { tooltip.classed("visible", false).attr("aria-hidden", "true"); }

function svgFor(selector, height, minWidth = 0) {
  const node = document.querySelector(selector);
  node.replaceChildren();
  const width = Math.max(minWidth, node.clientWidth || 640);
  const svg = d3.select(node).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height);
  return { node, svg, width, height };
}

function grid(svg, scale, margin, width) {
  svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(scale).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
}

function renderMetrics(summary) {
  const values = [
    ["Заявки", number.format(summary.applications), `${summary.features_used} признаков`],
    ["Доля дефолтов", percent(summary.default_rate), `${number.format(summary.defaults)} случаев`],
    ["ROC-AUC", d3.format(".3f")(summary.roc_auc), "0.500 = случайно"],
    ["Gini", d3.format(".3f")(summary.gini), "2 × AUC − 1"],
  ];
  d3.select("#metrics").selectAll("article").data(values).join("article").attr("class", "metric")
    .html(([label, value, note]) => `<span class="metric-label">${label}</span><strong class="metric-value">${value}</strong><small>${note}</small>`);
}

function renderDecision(summary) {
  const values = [
    ["Recall дефолтов", percent(summary.recall_at_threshold)],
    ["Precision", percent(summary.precision_at_recall)],
    ["Порог риска", percent(summary.operating_threshold)],
    ["Approval rate", percent(summary.approval_rate)],
  ];
  d3.select("#decision-metrics").selectAll("div").data(values).join("div")
    .html(([label, value]) => `<span>${label}</span><strong>${value}</strong>`);
}

function renderRoc(data, summary) {
  const { svg, width, height } = svgFor("#roc-chart", 350, 430);
  const margin = { top: 18, right: 24, bottom: 48, left: 54 };
  const x = d3.scaleLinear([0, 1], [margin.left, width - margin.right]);
  const y = d3.scaleLinear([0, 1], [height - margin.bottom, margin.top]);
  grid(svg, y, margin, width);
  svg.append("path").attr("d", d3.line()([[x(0), y(0)], [x(1), y(1)]])).attr("stroke", colors.muted).attr("stroke-dasharray", "5 5").attr("fill", "none");
  svg.append("path").datum(data).attr("d", d3.line().x(d => x(+d.fpr)).y(d => y(+d.tpr)).curve(d3.curveMonotoneX)).attr("stroke", colors.green).attr("stroke-width", 3).attr("fill", "none");
  svg.append("path").datum([{ fpr: 0, tpr: 0 }, ...data, { fpr: 1, tpr: 0 }]).attr("d", d3.area().x(d => x(+d.fpr)).y0(y(0)).y1(d => y(+d.tpr))).attr("fill", colors.green).attr("opacity", .08);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  svg.append("text").attr("class", "chart-label").attr("x", width - margin.right).attr("y", height - 8).attr("text-anchor", "end").text("False positive rate");
  svg.append("text").attr("class", "chart-kpi").attr("x", margin.left + 18).attr("y", margin.top + 34).text(`AUC ${d3.format(".3f")(summary.roc_auc)}`);
}

function renderPr(data, summary) {
  const { svg, width, height } = svgFor("#pr-chart", 350, 430);
  const margin = { top: 18, right: 24, bottom: 48, left: 54 };
  const x = d3.scaleLinear([0, 1], [margin.left, width - margin.right]);
  const y = d3.scaleLinear([0, 1], [height - margin.bottom, margin.top]);
  grid(svg, y, margin, width);
  svg.append("line").attr("x1", x(0)).attr("x2", x(1)).attr("y1", y(summary.baseline_average_precision)).attr("y2", y(summary.baseline_average_precision)).attr("stroke", colors.muted).attr("stroke-dasharray", "5 5");
  svg.append("path").datum(data.sort((a, b) => +a.recall - +b.recall)).attr("d", d3.line().x(d => x(+d.recall)).y(d => y(+d.precision)).curve(d3.curveMonotoneX)).attr("stroke", colors.orange).attr("stroke-width", 3).attr("fill", "none");
  svg.append("circle").attr("cx", x(summary.recall_at_threshold)).attr("cy", y(summary.precision_at_recall)).attr("r", 6).attr("fill", colors.lime).attr("stroke", colors.ink).attr("stroke-width", 2);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  svg.append("text").attr("class", "chart-label").attr("x", width - margin.right).attr("y", height - 8).attr("text-anchor", "end").text("Recall");
  svg.append("text").attr("class", "chart-kpi").attr("x", margin.left + 18).attr("y", margin.top + 34).text(`AP ${d3.format(".3f")(summary.average_precision)}`);
}

function renderDeciles(data) {
  const { svg, width, height } = svgFor("#decile-chart", 390, 720);
  const margin = { top: 22, right: 62, bottom: 54, left: 58 };
  const x = d3.scaleBand(data.map(d => +d.risk_decile), [margin.left, width - margin.right]).padding(.25);
  const yRate = d3.scaleLinear([0, d3.max(data, d => +d.default_rate) * 1.15], [height - margin.bottom, margin.top]);
  const yCapture = d3.scaleLinear([0, 1], [height - margin.bottom, margin.top]);
  grid(svg, yRate, margin, width);
  svg.append("g").selectAll("rect").data(data).join("rect")
    .attr("x", d => x(+d.risk_decile)).attr("width", x.bandwidth()).attr("y", yRate(0)).attr("height", 0).attr("rx", 2).attr("fill", colors.green)
    .on("mousemove", (event, d) => showTooltip(event, `<strong>Дециль ${d.risk_decile}</strong><br>Дефолтность ${percent(+d.default_rate)}<br>Найдено накопленно ${percent(+d.captured_defaults)}`)).on("mouseleave", hideTooltip)
    .transition().duration(550).attr("y", d => yRate(+d.default_rate)).attr("height", d => yRate(0) - yRate(+d.default_rate));
  svg.append("path").datum(data).attr("d", d3.line().x(d => x(+d.risk_decile) + x.bandwidth() / 2).y(d => yCapture(+d.captured_defaults)).curve(d3.curveMonotoneX)).attr("stroke", colors.orange).attr("stroke-width", 3).attr("fill", "none");
  svg.append("g").selectAll("circle").data(data).join("circle").attr("cx", d => x(+d.risk_decile) + x.bandwidth() / 2).attr("cy", d => yCapture(+d.captured_defaults)).attr("r", 4).attr("fill", colors.orange).attr("stroke", "white").attr("stroke-width", 2);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).tickSize(0).tickPadding(12));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(yRate).ticks(5).tickFormat(d3.format(".0%")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${width - margin.right},0)`).call(d3.axisRight(yCapture).ticks(5).tickFormat(d3.format(".0%")));
  svg.append("text").attr("class", "chart-label").attr("x", width / 2).attr("y", height - 9).attr("text-anchor", "middle").text("От высокого риска → к низкому");
}

function renderImportance(data, count) {
  const top = data.slice(0, count).reverse();
  const { svg, width, height } = svgFor("#importance-chart", Math.max(350, top.length * 34 + 70), 720);
  const margin = { top: 18, right: 46, bottom: 42, left: Math.min(230, width * .32) };
  const x = d3.scaleLinear([0, d3.max(top, d => +d.importance) * 1.08], [margin.left, width - margin.right]);
  const y = d3.scaleBand(top.map(d => d.feature), [height - margin.bottom, margin.top]).padding(.28);
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).tickSize(0).tickPadding(12)).call(g => g.select(".domain").remove());
  svg.append("g").selectAll("rect").data(top).join("rect").attr("x", margin.left).attr("y", d => y(d.feature)).attr("height", y.bandwidth()).attr("width", d => x(+d.importance) - margin.left).attr("rx", 2).attr("fill", d => d.direction === "risk_up" ? colors.orange : colors.green)
    .on("mousemove", (event, d) => showTooltip(event, `<strong>${d.feature}</strong><br>${d.direction === "risk_up" ? "Связан с ростом" : "Связан со снижением"} риска<br>Доминирующий коэффициент ${d3.format("+.3f")(+d.coefficient)}`)).on("mouseleave", hideTooltip);
  svg.append("g").selectAll("text.value").data(top).join("text").attr("class", "bar-value").attr("x", d => x(+d.importance) + 7).attr("y", d => y(d.feature) + y.bandwidth() / 2 + 4).text(d => d.direction === "risk_up" ? "↑ риск" : "↓ риск").attr("fill", d => d.direction === "risk_up" ? colors.orange : colors.green);
  svg.append("text").attr("class", "chart-label").attr("x", margin.left).attr("y", height - 8).text("Сумма абсолютных коэффициентов по исходному признаку");
}

function renderDistribution(data, summary) {
  const totals = d3.rollup(data, values => d3.sum(values, d => +d.applications), d => +d.target);
  data.forEach(d => d.share = +d.applications / totals.get(+d.target));
  const series = d3.groups(data, d => +d.target);
  const { node, svg, width, height } = svgFor("#distribution-chart", 360, 720);
  const margin = { top: 38, right: 28, bottom: 48, left: 58 };
  const x = d3.scaleLinear([0, 1], [margin.left, width - margin.right]);
  const y = d3.scaleLinear([0, d3.max(data, d => d.share) * 1.15], [height - margin.bottom, margin.top]);
  grid(svg, y, margin, width);
  const palette = new Map([[0, colors.green], [1, colors.orange]]);
  const labels = new Map([[0, "Без дефолта"], [1, "Дефолт"]]);
  d3.select(node).insert("div", "svg").attr("class", "legend").selectAll("span").data(series).join("span").html(([key]) => `<i style="background:${palette.get(key)}"></i>${labels.get(key)}`);
  const area = d3.area().x(d => x((+d.score_from + +d.score_to) / 2)).y0(y(0)).y1(d => y(d.share)).curve(d3.curveMonotoneX);
  series.forEach(([key, values]) => svg.append("path").datum(values.sort((a, b) => +a.bucket - +b.bucket)).attr("d", area).attr("fill", palette.get(key)).attr("opacity", .27).attr("stroke", palette.get(key)).attr("stroke-width", 2));
  svg.append("line").attr("x1", x(summary.operating_threshold)).attr("x2", x(summary.operating_threshold)).attr("y1", margin.top).attr("y2", height - margin.bottom).attr("stroke", colors.ink).attr("stroke-dasharray", "5 4");
  svg.append("text").attr("x", x(summary.operating_threshold) + 7).attr("y", margin.top + 12).attr("class", "chart-label").text(`Порог ${percent(summary.operating_threshold)}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(10).tickFormat(d3.format(".0%")));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
}

function empty(selector, message) { document.querySelector(selector).innerHTML = `<div class="empty-state">${message}</div>`; }

async function loadDashboard() {
  try {
    const [summary, roc, pr, deciles, importance, distribution] = await Promise.all([
      d3.json(`${DATA_ROOT}/summary.json`),
      d3.csv(`${DATA_ROOT}/roc_curve.csv`, d3.autoType),
      d3.csv(`${DATA_ROOT}/pr_curve.csv`, d3.autoType),
      d3.csv(`${DATA_ROOT}/risk_deciles.csv`, d3.autoType),
      d3.csv(`${DATA_ROOT}/feature_importance.csv`, d3.autoType),
      d3.csv(`${DATA_ROOT}/score_distribution.csv`, d3.autoType),
    ]);
    document.querySelector("#validation-label").textContent = `${number.format(summary.validation_applications)} заявок · ${summary.validation}`;
    document.querySelector("#mode-label").textContent = summary.data_mode === "demo" ? "Синтетические данные" : "Данные Kaggle";
    document.querySelector("#demo-banner").classList.toggle("hidden", summary.data_mode !== "demo");
    renderMetrics(summary); renderDecision(summary); renderRoc(roc, summary); renderPr(pr, summary); renderDeciles(deciles); renderDistribution(distribution, summary);
    if (!importance.length) empty("#importance-chart", "Нет коэффициентов для отображения");
    else {
      const slider = document.querySelector("#features-filter");
      slider.max = Math.min(20, importance.length);
      slider.min = Math.min(5, importance.length);
      slider.value = Math.min(12, importance.length);
      const update = () => { document.querySelector("#features-value").textContent = slider.value; renderImportance(importance, +slider.value); };
      slider.addEventListener("input", update); update();
    }
  } catch (error) {
    const box = document.querySelector("#error-state");
    box.classList.remove("hidden");
    box.innerHTML = `<strong>Не удалось загрузить результаты.</strong><br>Сначала запустите <code>make demo</code> или <code>make analyze</code>, затем обновите страницу.`;
    console.error(error);
  }
}

loadDashboard();
