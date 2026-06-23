/* =========================================================================
   Apex Power & Utilities - 24h Demand Forecast Dashboard
   Vanilla JS. Same-origin API (relative URLs). Chart.js via CDN.

   Strategy:
     1. Try ONE call to /api/dashboard (composite forecast + weather + holidays).
     2. If that endpoint is missing/incomplete, fall back to the individual
        /api/forecast, /api/weather, /api/holidays endpoints in parallel.
     3. Normalize every payload defensively so the UI renders regardless of
        small shape differences, then draw header KPIs, the demand chart
        (TOTAL + F1/F2/F3 toggleable), a temp/humidity weather chart with
        cloud/wind cards, and a localized holidays table with badges +
        in-window highlighting & chart annotations.
   ========================================================================= */
"use strict";

/* ----------------------------------------------------------------- Config */
const SERIES = {
  TOTAL: { label: "Total", color: getCssVar("--total", "#ffb300"), width: 2.4, fill: true },
  F1:    { label: "Feeder F1", color: getCssVar("--f1", "#4dabf7"), width: 1.6, fill: false },
  F2:    { label: "Feeder F2", color: getCssVar("--f2", "#51cf66"), width: 1.6, fill: false },
  F3:    { label: "Feeder F3", color: getCssVar("--f3", "#e599f7"), width: 1.6, fill: false },
};
const FEEDER_KEYS = ["F1", "F2", "F3"];

/* Per-series visibility (TOTAL on by default, feeders off to reduce clutter). */
const visible = { TOTAL: true, F1: false, F2: false, F3: false };

/* Chart handles + last-rendered state (for redraws on toggle). */
let demandChart = null;
let weatherChart = null;
let lastForecast = null;     // normalized forecast object
let lastHolidaysInWindow = [];

/* ------------------------------------------------------------- DOM helpers */
function $(id) { return document.getElementById(id); }

function getCssVar(name, fallback) {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (_e) {
    return fallback;
  }
}

function show(el) { if (el) el.classList.add("show"); }
function hide(el) { if (el) el.classList.remove("show"); }

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function notice(containerId, message, muted) {
  const el = $(containerId);
  if (!el) return;
  if (!message) { el.innerHTML = ""; return; }
  el.innerHTML =
    '<div class="notice' + (muted ? " muted" : "") + '">' + escapeHtml(message) + "</div>";
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* --------------------------------------------------------------- Formatting */
function fmtTime(d) {
  // HH:MM 24h
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

function fmtDateTime(iso) {
  const d = parseLocalDate(iso);
  if (!d) return "—";
  return d.toLocaleString("en-GB", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

function fmtDateShort(iso) {
  const d = parseLocalDate(iso);
  if (!d) return iso || "—";
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

function fmtKw(v) {
  if (v == null || isNaN(v)) return "—";
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(2) + " MW";
  return Math.round(v).toLocaleString() + " kW";
}

function fmtNum(v, digits) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(digits == null ? 2 : digits);
}

/* Parse an ISO timestamp that may be tz-naive local wall-clock (from the API).
   We treat naive strings as LOCAL time (the backend emits Asia/Kolkata local). */
function parseLocalDate(iso) {
  if (!iso) return null;
  if (iso instanceof Date) return iso;
  let s = String(iso);
  // If it already carries a tz (Z or +hh:mm), trust it.
  const hasTz = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(s);
  if (hasTz) {
    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }
  // Naive "YYYY-MM-DDTHH:MM:SS" -> parse components as local time.
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?/);
  if (m) {
    return new Date(
      Number(m[1]), Number(m[2]) - 1, Number(m[3]),
      Number(m[4] || 0), Number(m[5] || 0), Number(m[6] || 0)
    );
  }
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

/* ------------------------------------------------------------- API fetching */
async function fetchJson(url) {
  const resp = await fetch(url, { headers: { Accept: "application/json" } });
  if (!resp.ok) throw new Error(url + " -> HTTP " + resp.status);
  return resp.json();
}

/* Try the composite dashboard endpoint, then fall back to the individual ones. */
async function loadAllData() {
  // 1) Preferred: single composite call.
  try {
    const dash = await fetchJson("/api/dashboard");
    const norm = normalizeDashboard(dash);
    if (norm.forecast) {
      // Backfill any missing piece from dedicated endpoints (best effort).
      if (!norm.weather) {
        norm.weather = await safeFetch("/api/weather").then(normalizeWeather);
      }
      if (!norm.holidays) {
        norm.holidays = await safeFetch("/api/holidays").then(normalizeHolidays);
      }
      return norm;
    }
  } catch (_e) {
    // fall through to individual endpoints
  }

  // 2) Fallback: fetch the three endpoints in parallel. Forecast is required.
  const [fc, wx, hol] = await Promise.allSettled([
    fetchJson("/api/forecast"),
    fetchJson("/api/weather"),
    fetchJson("/api/holidays"),
  ]);

  if (fc.status !== "fulfilled") {
    throw new Error("Forecast endpoint unavailable");
  }

  return {
    forecast: normalizeForecast(fc.value),
    weather: wx.status === "fulfilled" ? normalizeWeather(wx.value) : null,
    holidays: hol.status === "fulfilled" ? normalizeHolidays(hol.value) : null,
  };
}

async function safeFetch(url) {
  try { return await fetchJson(url); }
  catch (_e) { return null; }
}

/* ----------------------------------------------------- Payload normalizers */

/* Composite /api/dashboard may wrap pieces under various keys; be liberal. */
function normalizeDashboard(d) {
  if (!d || typeof d !== "object") return {};

  const fcRaw = d.forecast && (d.forecast.forecast || Array.isArray(d.forecast))
    ? d.forecast
    : (Array.isArray(d.forecast) ? { forecast: d.forecast } : d);

  const forecast = (fcRaw && (Array.isArray(fcRaw.forecast) || Array.isArray(fcRaw)))
    ? normalizeForecast(fcRaw, d)
    : (Array.isArray(d.forecast) ? normalizeForecast({ forecast: d.forecast }, d) : null);

  const weatherRaw = d.weather || (d.weather === undefined ? null : d.weather);
  const holidaysRaw = d.holidays || (d.holidays === undefined ? null : d.holidays);

  return {
    forecast: forecast,
    weather: weatherRaw ? normalizeWeather(weatherRaw) : null,
    holidays: holidaysRaw ? normalizeHolidays(holidaysRaw) : null,
  };
}

/* Forecast -> { generatedAt, location, metrics, source, peak, blocks:[{date,TOTAL,F1,F2,F3}] }
   `meta` is an optional outer object (composite payload) to read top-level fields from. */
function normalizeForecast(payload, meta) {
  if (!payload) return null;
  const arr = Array.isArray(payload) ? payload
            : Array.isArray(payload.forecast) ? payload.forecast
            : null;
  if (!arr) return null;

  const top = (payload && typeof payload === "object") ? payload : {};
  const outer = meta && typeof meta === "object" ? meta : {};

  const blocks = arr.map((row) => {
    const date = parseLocalDate(row.timestamp || row.time || row.datetime);
    const total = pickNum(row, ["total_load_kw", "total", "TOTAL", "total_kw"]);
    return {
      date: date,
      iso: row.timestamp || row.time || row.datetime || null,
      TOTAL: total,
      F1: pickNum(row, ["F1", "f1"]),
      F2: pickNum(row, ["F2", "f2"]),
      F3: pickNum(row, ["F3", "f3"]),
    };
  }).filter((b) => b.date);

  const metrics = top.model_metrics || outer.model_metrics ||
                  top.metrics || outer.metrics || {};

  let peak = -Infinity, peakAt = null;
  for (const b of blocks) {
    if (b.TOTAL != null && b.TOTAL > peak) { peak = b.TOTAL; peakAt = b.date; }
  }
  if (peak === -Infinity) peak = null;

  return {
    generatedAt: top.generated_at || outer.generated_at || null,
    location: top.location || outer.location || null,
    source: top.weather_source || outer.weather_source || null,
    metrics: metrics,
    peak: peak,
    peakAt: peakAt,
    blocks: blocks,
  };
}

/* Weather -> { source, hourly:[{date, temp, humidity, cloud, wind}] } */
function normalizeWeather(payload) {
  if (!payload) return null;
  const arr = Array.isArray(payload) ? payload
            : Array.isArray(payload.hourly) ? payload.hourly
            : Array.isArray(payload.weather) ? payload.weather
            : null;
  if (!arr) return null;

  const hourly = arr.map((row) => ({
    date: parseLocalDate(row.timestamp || row.time || row.datetime),
    temp: pickNum(row, ["temperature_c", "temperature", "temp", "Temperature"]),
    humidity: pickNum(row, ["humidity_pct", "humidity", "Humidity"]),
    cloud: pickNum(row, ["cloud_cover_pct", "cloud_cover", "cloud", "CloudCover"]),
    wind: pickNum(row, ["wind_speed_kmh", "wind_speed", "wind", "WindSpeed"]),
  })).filter((h) => h.date);

  return {
    source: payload.source || null,
    hourly: hourly,
  };
}

/* Holidays -> array of { date, name, type, kinds:[festive|industrial|national|religious] } */
function normalizeHolidays(payload) {
  if (!payload) return null;
  const arr = Array.isArray(payload) ? payload
            : Array.isArray(payload.holidays) ? payload.holidays
            : null;
  if (!arr) return null;

  return arr.map((row) => {
    const kinds = [];
    if (truthy(row.is_festive)) kinds.push("festive");
    if (truthy(row.is_industrial)) kinds.push("industrial");
    if (truthy(row.is_national)) kinds.push("national");
    // Derive from the `type` string when explicit flags are absent.
    const type = String(row.type || "").toLowerCase();
    if (!kinds.length) {
      if (type.includes("festive")) kinds.push("festive");
      else if (type.includes("industrial")) kinds.push("industrial");
      else if (type.includes("national")) kinds.push("national");
      else if (type.includes("religious")) kinds.push("religious");
    }
    if (type.includes("religious") && !kinds.includes("religious") && !kinds.length) {
      kinds.push("religious");
    }
    if (!kinds.length) kinds.push("neutral");
    return {
      date: row.date || row.timestamp || null,
      name: row.name || "Holiday",
      type: row.type || "",
      kinds: kinds,
    };
  }).filter((h) => h.date);
}

function pickNum(obj, keys) {
  for (const k of keys) {
    if (obj[k] != null && !isNaN(Number(obj[k]))) return Number(obj[k]);
  }
  return null;
}
function truthy(v) {
  return v === true || v === 1 || v === "1" || v === "true" || v === "True";
}

/* ============================================================ RENDERING === */

function renderHeader(forecast, weather) {
  if (forecast && forecast.location) setText("locationText", forecast.location);

  setText("kpiGenerated", forecast && forecast.generatedAt
    ? fmtDateTime(forecast.generatedAt) : "—");

  const m = (forecast && forecast.metrics) || {};
  const total = m.TOTAL || m.total || {};
  setText("kpiMape", total.MAPE != null ? fmtNum(total.MAPE, 2) + " %" : "—");
  setText("kpiRmse", total.RMSE != null ? fmtNum(total.RMSE, 0) + " kW" : "—");
  setText("kpiPeak", forecast && forecast.peak != null
    ? fmtKw(forecast.peak) + (forecast.peakAt ? " @ " + fmtTime(forecast.peakAt) : "")
    : "—");

  // Weather source chip (live = open-meteo, fallback = climatology).
  const src = (weather && weather.source) || (forecast && forecast.source) || null;
  const chip = $("kpiSource");
  if (chip) {
    chip.classList.remove("live", "fallback");
    if (src) {
      const isLive = /open.?meteo/i.test(src);
      chip.classList.add(isLive ? "live" : "fallback");
      setText("kpiSourceText", isLive ? "Live (Open-Meteo)" : "Climatology");
    } else {
      setText("kpiSourceText", "—");
    }
  }
}

/* --------------------------------------------------- Demand (hero) chart */
function renderDemandChart(forecast) {
  const box = $("demandChart");
  if (!box) return;

  if (!forecast || !forecast.blocks.length) {
    if (demandChart) { demandChart.destroy(); demandChart = null; }
    notice("demandNotice", "No forecast data to display.", true);
    return;
  }
  notice("demandNotice", "");

  setText("forecastSub",
    forecast.blocks.length + " × 10-minute blocks · " +
    fmtTime(forecast.blocks[0].date) + " → " +
    fmtTime(forecast.blocks[forecast.blocks.length - 1].date) + " · load (kW)");

  const labels = forecast.blocks.map((b) => b.date);
  const datasets = buildDemandDatasets(forecast);

  const annotations = buildHolidayAnnotations(forecast);

  if (demandChart) {
    demandChart.data.labels = labels;
    demandChart.data.datasets = datasets;
    demandChart.options.plugins.annotation.annotations = annotations;
    demandChart.update();
    return;
  }

  demandChart = new Chart(box.getContext("2d"), {
    type: "line",
    data: { labels: labels, datasets: datasets },
    options: baseLineOptions({
      yTitle: "Demand (kW)",
      tooltipUnit: "kW",
      annotations: annotations,
    }),
  });
}

function buildDemandDatasets(forecast) {
  const order = ["TOTAL", "F1", "F2", "F3"];
  return order.map((key) => {
    const cfg = SERIES[key];
    return {
      label: cfg.label,
      data: forecast.blocks.map((b) => b[key]),
      borderColor: cfg.color,
      backgroundColor: key === "TOTAL" ? hexToRgba(cfg.color, 0.14) : hexToRgba(cfg.color, 0.08),
      borderWidth: cfg.width,
      fill: cfg.fill,
      tension: 0.32,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBorderColor: "#fff",
      hidden: !visible[key],
      spanGaps: true,
    };
  });
}

/* Highlight any holiday date that falls inside the forecast window. */
function buildHolidayAnnotations(forecast) {
  const annotations = {};
  if (!lastHolidaysInWindow.length || !forecast.blocks.length) return annotations;

  const start = forecast.blocks[0].date.getTime();
  const end = forecast.blocks[forecast.blocks.length - 1].date.getTime();

  lastHolidaysInWindow.forEach((h, i) => {
    const d = parseLocalDate(h.date);
    if (!d) return;
    // Use midnight of the holiday if within the window; otherwise clamp to start.
    let t = d.getTime();
    if (t < start) t = start;
    if (t > end) return;
    annotations["hol" + i] = {
      type: "line",
      xMin: t,
      xMax: t,
      borderColor: getCssVar("--accent", "#ffb300"),
      borderWidth: 1.5,
      borderDash: [5, 4],
      label: {
        display: true,
        content: "★ " + h.name,
        position: "start",
        backgroundColor: "rgba(255,179,0,0.92)",
        color: "#1a1300",
        font: { size: 10, weight: "bold" },
        padding: 4,
      },
    };
  });
  return annotations;
}

/* --------------------------------------------------- Weather chart + cards */
function renderWeather(weather) {
  const box = $("weatherChart");

  if (!weather || !weather.hourly.length) {
    if (weatherChart) { weatherChart.destroy(); weatherChart = null; }
    fillWeatherCards(null);
    notice("weatherNotice", "Weather data is currently unavailable.", true);
    return;
  }
  notice("weatherNotice", "");
  fillWeatherCards(weather);

  const labels = weather.hourly.map((h) => h.date);
  const tempColor = getCssVar("--temp", "#ff6b6b");
  const humColor = getCssVar("--humidity", "#4dabf7");

  const datasets = [
    {
      label: "Temperature (°C)",
      yAxisID: "yTemp",
      data: weather.hourly.map((h) => h.temp),
      borderColor: tempColor,
      backgroundColor: hexToRgba(tempColor, 0.12),
      borderWidth: 2,
      fill: true,
      tension: 0.35,
      pointRadius: 0,
      pointHoverRadius: 4,
      spanGaps: true,
    },
    {
      label: "Humidity (%)",
      yAxisID: "yHum",
      data: weather.hourly.map((h) => h.humidity),
      borderColor: humColor,
      backgroundColor: "transparent",
      borderWidth: 1.8,
      borderDash: [4, 3],
      fill: false,
      tension: 0.35,
      pointRadius: 0,
      pointHoverRadius: 4,
      spanGaps: true,
    },
  ];

  if (weatherChart) {
    weatherChart.data.labels = labels;
    weatherChart.data.datasets = datasets;
    weatherChart.update();
    return;
  }

  const opts = baseLineOptions({ tooltipUnit: "" });
  // Dual y-axes for temp & humidity.
  opts.scales.y = undefined;
  opts.scales.yTemp = {
    position: "left",
    grid: { color: "rgba(255,255,255,0.05)" },
    ticks: { color: getCssVar("--temp", "#ff6b6b"), callback: (v) => v + "°" },
    title: { display: true, text: "Temp (°C)", color: getCssVar("--text-muted", "#9aa7b5") },
  };
  opts.scales.yHum = {
    position: "right",
    min: 0, max: 100,
    grid: { drawOnChartArea: false },
    ticks: { color: getCssVar("--humidity", "#4dabf7"), callback: (v) => v + "%" },
    title: { display: true, text: "Humidity (%)", color: getCssVar("--text-muted", "#9aa7b5") },
  };
  opts.plugins.legend.display = true;

  weatherChart = new Chart(box.getContext("2d"), {
    type: "line",
    data: { labels: labels, datasets: datasets },
    options: opts,
  });
}

function fillWeatherCards(weather) {
  if (!weather || !weather.hourly.length) {
    ["wCloud", "wWind", "wTemp", "wHum"].forEach((id) => setText(id, "—"));
    ["wCloudRange", "wWindRange", "wTempRange", "wHumRange"].forEach((id) => setText(id, ""));
    return;
  }
  const h = weather.hourly;
  const stats = (key) => {
    const vals = h.map((x) => x[key]).filter((v) => v != null && !isNaN(v));
    if (!vals.length) return null;
    const sum = vals.reduce((a, b) => a + b, 0);
    return { avg: sum / vals.length, min: Math.min(...vals), max: Math.max(...vals) };
  };

  const cloud = stats("cloud"), wind = stats("wind"), temp = stats("temp"), hum = stats("humidity");

  setText("wCloud", cloud ? Math.round(cloud.avg) + " %" : "—");
  setText("wCloudRange", cloud ? "range " + Math.round(cloud.min) + "–" + Math.round(cloud.max) + " %" : "");

  setText("wWind", wind ? fmtNum(wind.avg, 1) + " km/h" : "—");
  setText("wWindRange", wind ? "gusts to " + fmtNum(wind.max, 1) + " km/h" : "");

  setText("wTemp", temp ? fmtNum(temp.min, 1) + "° – " + fmtNum(temp.max, 1) + "°C" : "—");
  setText("wTempRange", temp ? "avg " + fmtNum(temp.avg, 1) + " °C" : "");

  setText("wHum", hum ? Math.round(hum.avg) + " %" : "—");
  setText("wHumRange", hum ? "range " + Math.round(hum.min) + "–" + Math.round(hum.max) + " %" : "");
}

/* ----------------------------------------------------- Holidays table */
function renderHolidays(holidays, forecast) {
  const body = $("holidaysBody");
  if (!body) return;
  body.innerHTML = "";
  lastHolidaysInWindow = [];

  if (!holidays || !holidays.length) {
    notice("holidaysNotice", "No upcoming holidays in the calendar.", true);
    return;
  }
  notice("holidaysNotice", "");

  // Determine the forecast window for in-window highlighting.
  let winStart = null, winEnd = null;
  if (forecast && forecast.blocks.length) {
    winStart = startOfDay(forecast.blocks[0].date);
    winEnd = forecast.blocks[forecast.blocks.length - 1].date;
  }

  // Sort by date ascending and keep upcoming (relative to first forecast block / today).
  const refTime = forecast && forecast.blocks.length
    ? forecast.blocks[0].date.getTime()
    : Date.now();

  const sorted = holidays
    .map((h) => ({ ...h, _t: parseLocalDate(h.date) }))
    .filter((h) => h._t)
    .sort((a, b) => a._t - b._t);

  // Show upcoming first; if everything is in the past, just show the latest few.
  let upcoming = sorted.filter((h) => h._t.getTime() >= startOfDayTime(refTime));
  if (!upcoming.length) upcoming = sorted.slice(-8);
  const rows = upcoming.slice(0, 12);

  rows.forEach((h) => {
    const inWindow = winStart && winEnd &&
      h._t.getTime() >= winStart.getTime() && h._t.getTime() <= winEnd.getTime();
    if (inWindow) lastHolidaysInWindow.push(h);

    const tr = document.createElement("tr");
    if (inWindow) tr.className = "in-window";

    const tdDate = document.createElement("td");
    tdDate.className = "hdate";
    tdDate.textContent = fmtDateShort(h.date);

    const tdName = document.createElement("td");
    tdName.className = "hname";
    tdName.textContent = h.name;
    if (inWindow) {
      const pill = document.createElement("span");
      pill.className = "pill-window";
      pill.textContent = "in window";
      tdName.appendChild(pill);
    }

    const tdType = document.createElement("td");
    const badges = document.createElement("div");
    badges.className = "badges";
    h.kinds.forEach((k) => {
      const b = document.createElement("span");
      b.className = "badge " + k;
      b.textContent = k;
      badges.appendChild(b);
    });
    tdType.appendChild(badges);

    tr.appendChild(tdDate);
    tr.appendChild(tdName);
    tr.appendChild(tdType);
    body.appendChild(tr);
  });

  if (lastHolidaysInWindow.length && demandChart) {
    demandChart.options.plugins.annotation.annotations = buildHolidayAnnotations(forecast);
    demandChart.update();
  }
}

/* ---------------------------------------------------- Chart.js base options */
function baseLineOptions(opts) {
  opts = opts || {};
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    animation: { duration: 350 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(20,26,34,0.96)",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        titleColor: "#e7ecf3",
        bodyColor: "#cdd6e0",
        padding: 10,
        callbacks: {
          title: (items) => {
            if (!items.length) return "";
            const d = items[0].parsed && items[0].parsed.x != null
              ? new Date(items[0].parsed.x) : null;
            return d ? d.toLocaleString("en-GB", {
              weekday: "short", hour: "2-digit", minute: "2-digit",
            }) : items[0].label;
          },
          label: (item) => {
            const u = opts.tooltipUnit ? " " + opts.tooltipUnit : "";
            const v = item.parsed.y;
            return " " + item.dataset.label + ": " +
              (v == null ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: 1 })) + u;
          },
        },
      },
      annotation: { annotations: opts.annotations || {} },
    },
    scales: {
      x: {
        type: "time",
        time: {
          unit: "hour",
          stepSize: 2,
          tooltipFormat: "HH:mm",
          displayFormats: { hour: "HH:mm" },
        },
        grid: { color: "rgba(255,255,255,0.04)" },
        ticks: {
          color: getCssVar("--text-muted", "#9aa7b5"),
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 13,
        },
      },
      y: {
        grid: { color: "rgba(255,255,255,0.05)" },
        ticks: {
          color: getCssVar("--text-muted", "#9aa7b5"),
          callback: (v) => Number(v).toLocaleString(),
        },
        title: opts.yTitle
          ? { display: true, text: opts.yTitle, color: getCssVar("--text-muted", "#9aa7b5") }
          : { display: false },
      },
    },
  };
}

/* Chart.js v4 needs a time adapter for type:"time". We provide a tiny built-in
   adapter so we don't depend on date-fns/luxon CDNs. It only does what the
   linear-time axis needs (label formatting for hour ticks + tooltips). */
function installTimeAdapter() {
  if (typeof Chart === "undefined" || !Chart._adapters) return;
  const fmtHM = (ts) => {
    const d = new Date(ts);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return hh + ":" + mm;
  };
  Chart._adapters._date.override({
    _id: "apex-min",
    formats: () => ({
      hour: "HH:mm", minute: "HH:mm", day: "dd MMM",
      millisecond: "HH:mm:ss", second: "HH:mm:ss", week: "dd MMM",
      month: "MMM", quarter: "QQQ", year: "yyyy",
    }),
    parse: (v) => {
      if (v == null) return null;
      if (v instanceof Date) return v.getTime();
      if (typeof v === "number") return v;
      const d = parseLocalDate(v);
      return d ? d.getTime() : null;
    },
    format: (ts) => fmtHM(ts),
    add: (ts, amount, unit) => {
      const d = new Date(ts);
      const mult = { hour: 3600e3, minute: 60e3, day: 864e5, second: 1e3 }[unit] || 0;
      if (mult) return ts + amount * mult;
      if (unit === "month") { d.setMonth(d.getMonth() + amount); return d.getTime(); }
      if (unit === "year") { d.setFullYear(d.getFullYear() + amount); return d.getTime(); }
      return ts;
    },
    diff: (a, b, unit) => {
      const mult = { hour: 3600e3, minute: 60e3, day: 864e5, second: 1e3 }[unit] || 1;
      return (a - b) / mult;
    },
    startOf: (ts, unit) => {
      const d = new Date(ts);
      if (unit === "hour") { d.setMinutes(0, 0, 0); }
      else if (unit === "minute") { d.setSeconds(0, 0); }
      else if (unit === "day") { d.setHours(0, 0, 0, 0); }
      return d.getTime();
    },
    endOf: (ts, unit) => {
      const d = new Date(ts);
      if (unit === "hour") { d.setMinutes(59, 59, 999); }
      else if (unit === "day") { d.setHours(23, 59, 59, 999); }
      return d.getTime();
    },
  });
}

/* --------------------------------------------------------------- Utilities */
function hexToRgba(hex, alpha) {
  let h = String(hex).replace("#", "").trim();
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  if (h.length !== 6) return "rgba(255,179,0," + alpha + ")";
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
}

function startOfDay(d) {
  const x = new Date(d.getTime());
  x.setHours(0, 0, 0, 0);
  return x;
}
function startOfDayTime(ts) {
  const x = new Date(ts);
  x.setHours(0, 0, 0, 0);
  return x.getTime();
}

/* ---------------------------------------------------- Series toggle wiring */
function wireToggles() {
  const container = $("seriesToggles");
  if (!container) return;
  container.querySelectorAll(".toggle").forEach((btn) => {
    const key = btn.getAttribute("data-series");
    syncToggleClass(btn, key);
    btn.addEventListener("click", () => {
      visible[key] = !visible[key];
      syncToggleClass(btn, key);
      if (demandChart) {
        const idx = ["TOTAL", "F1", "F2", "F3"].indexOf(key);
        if (idx >= 0) {
          demandChart.setDatasetVisibility(idx, visible[key]);
          demandChart.update();
        }
      }
    });
    btn.setAttribute("aria-pressed", String(!!visible[key]));
  });
}
function syncToggleClass(btn, key) {
  btn.classList.toggle("off", !visible[key]);
  btn.setAttribute("aria-pressed", String(!!visible[key]));
}

/* ------------------------------------------------------------- Orchestration */
async function refresh() {
  hide($("errorOverlay"));
  show($("loadingOverlay"));
  try {
    const data = await loadAllData();
    lastForecast = data.forecast;

    renderHeader(data.forecast, data.weather);
    renderHolidays(data.holidays, data.forecast); // sets lastHolidaysInWindow first
    renderDemandChart(data.forecast);              // uses lastHolidaysInWindow for annotations
    renderWeather(data.weather);

    hide($("loadingOverlay"));
  } catch (err) {
    hide($("loadingOverlay"));
    setText("errorMessage",
      "We couldn't reach the forecast service (" + (err && err.message ? err.message : "unknown error") +
      "). Make sure the backend is running at this address, then try again.");
    show($("errorOverlay"));
  }
}

function init() {
  installTimeAdapter();
  wireToggles();
  const retry = $("retryBtn");
  if (retry) retry.addEventListener("click", refresh);
  refresh();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
