"""Simple operator dashboard HTML for the MVP."""


def render_dashboard(site_id: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Adaptive Ads Router Dashboard</title>
    <style>
      :root {{
        --bg: #f7f2e8;
        --card: #fffdf7;
        --ink: #17212b;
        --muted: #5a6270;
        --line: #d8d4ca;
        --accent: #0f766e;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "SF Pro Display", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(255, 192, 120, 0.45), transparent 32%),
          radial-gradient(circle at top right, rgba(111, 214, 201, 0.3), transparent 28%),
          var(--bg);
      }}
      main {{
        width: min(1100px, 92vw);
        margin: 0 auto;
        padding: 32px 0 56px;
      }}
      .hero, .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 24px;
      }}
      .hero h1 {{
        margin: 0 0 10px;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.02;
        max-width: 10ch;
      }}
      .hero p, .card p, .card li {{
        color: var(--muted);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-top: 16px;
      }}
      .stack {{
        display: grid;
        gap: 16px;
        margin-top: 16px;
      }}
      .two-col {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }}
      .metric {{
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
      }}
      .label {{
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        padding: 12px 10px;
        border-bottom: 1px solid var(--line);
      }}
      code, pre {{
        font-family: "SF Mono", "Monaco", monospace;
      }}
      pre {{
        white-space: pre-wrap;
        background: #f3efe5;
        padding: 14px;
        border-radius: 14px;
        border: 1px solid var(--line);
      }}
      .copy-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
      }}
      .copy-row code {{
        display: block;
        overflow-wrap: anywhere;
        color: var(--ink);
      }}
      .copy-btn {{
        appearance: none;
        border: 1px solid var(--line);
        background: #f8f4ea;
        border-radius: 999px;
        padding: 8px 12px;
        font-weight: 600;
        color: var(--ink);
        cursor: pointer;
      }}
      .copy-btn:hover {{
        border-color: var(--accent);
        color: var(--accent);
      }}
      .primary-btn {{
        appearance: none;
        border: 0;
        background: var(--accent);
        color: #fff;
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 700;
        cursor: pointer;
      }}
      .primary-btn:hover {{
        opacity: 0.92;
      }}
      .muted {{
        color: var(--muted);
      }}
      .status {{
        min-height: 24px;
        margin-top: 10px;
        color: var(--muted);
      }}
      .table-wrap {{
        overflow-x: auto;
      }}
      .filter-bar {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
      }}
      .filter-bar label {{
        display: grid;
        gap: 6px;
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .filter-bar input, .filter-bar select {{
        width: 100%;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
      }}
      .pill {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #e5f5f2;
        color: var(--accent);
        font-weight: 600;
      }}
      @media (max-width: 860px) {{
        .grid, .two-col {{ grid-template-columns: 1fr; }}
        .filter-bar {{ grid-template-columns: 1fr; }}
        .copy-row {{
          flex-direction: column;
          align-items: flex-start;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <span class="pill">MVP Dashboard</span>
        <h1 id="site-name">Loading site...</h1>
        <p>Route paid clicks, record conversions, and watch the winning variant emerge without wiring a separate analytics app.</p>
      </section>

      <section class="grid">
        <article class="card">
          <div class="label">Site ID</div>
          <p class="metric" id="site-id">{site_id}</p>
        </article>
        <article class="card">
          <div class="label">Regime</div>
          <p class="metric" id="regime">-</p>
        </article>
        <article class="card">
          <div class="label">Total Sessions</div>
          <p class="metric" id="sessions">0</p>
        </article>
      </section>

      <section class="stack">
        <article class="card">
          <h2>Variant Performance</h2>
          <table>
            <thead>
              <tr>
                <th>Variant</th>
                <th>Destination</th>
                <th>Sessions</th>
                <th>Conversions</th>
                <th>Rate</th>
              </tr>
            </thead>
            <tbody id="variants-table">
              <tr><td colspan="5">Loading variants...</td></tr>
            </tbody>
          </table>
        </article>

        <article class="card">
          <h2>Integration</h2>
          <p>Copy these values directly into your ads, landing pages, or backend handlers.</p>
          <div class="two-col">
            <div>
              <div class="copy-row">
                <strong>Management URL</strong>
                <button class="copy-btn" data-copy-target="dashboard-url">Copy</button>
              </div>
              <pre><code id="dashboard-url">Loading management URL...</code></pre>
              <p class="muted">This private URL includes the site owner token. Treat it like a password.</p>
            </div>
            <div>
              <div class="copy-row">
                <strong>Ad URL</strong>
                <button class="copy-btn" data-copy-target="route-url">Copy</button>
              </div>
              <pre><code id="route-url">Loading route URL...</code></pre>
              <p class="muted">Use this as the Google Ads destination URL or behind your reverse proxy.</p>
            </div>
            <div>
              <div class="copy-row">
                <strong>Conversion URL Template</strong>
                <button class="copy-btn" data-copy-target="conversion-url">Copy</button>
              </div>
              <pre><code id="conversion-url">Loading conversion URL...</code></pre>
              <p class="muted">Trigger this after a successful lead, purchase, or booked call.</p>
            </div>
          </div>
        </article>

        <article class="card">
          <h2>Drop-In Snippets</h2>
          <div class="two-col">
            <div>
              <div class="copy-row">
                <strong>Frontend Thank-You Page</strong>
                <button class="copy-btn" data-copy-target="frontend-snippet">Copy</button>
              </div>
              <pre id="frontend-snippet">Loading frontend snippet...</pre>
            </div>
            <div>
              <div class="copy-row">
                <strong>Server-to-Server Conversion</strong>
                <button class="copy-btn" data-copy-target="backend-snippet">Copy</button>
              </div>
              <pre id="backend-snippet">Loading backend snippet...</pre>
            </div>
          </div>
        </article>

        <article class="card">
          <h2>Implementation Checklist</h2>
          <ul>
            <li>Point paid traffic to the Ad URL shown above.</li>
            <li>Keep the `aar_session_id` query param when the visitor moves through your funnel.</li>
            <li>Call the conversion URL or server-to-server endpoint when the primary goal completes.</li>
            <li>Refresh this dashboard after a few visits to verify sessions and conversions are moving.</li>
          </ul>
        </article>

        <article class="card">
          <h2>Test This Integration</h2>
          <p>Run one sample routed visit and one sample conversion to verify the dashboard updates end to end.</p>
          <button class="primary-btn" id="test-integration-btn">Run sample visit + conversion</button>
          <div class="status" id="test-status"></div>
        </article>

        <article class="card">
          <h2>Last 7 Days</h2>
          <p>Quick operating summary for the most recent week of tracked activity.</p>
          <div class="copy-row">
            <strong>Weekly Client Report</strong>
            <a id="weekly-report-link" class="copy-btn" href="#" target="_blank" rel="noreferrer">Open weekly report</a>
          </div>
          <table>
            <thead>
              <tr>
                <th>Routes</th>
                <th>Conversions</th>
                <th>CVR</th>
                <th>Revenue</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td id="report-routes">0</td>
                <td id="report-conversions">0</td>
                <td id="report-rate">0.00%</td>
                <td id="report-revenue">0.00</td>
              </tr>
            </tbody>
          </table>
        </article>

        <article class="card">
          <h2>Delivery Status</h2>
          <p>Latest scheduled weekly-report send attempts for this site.</p>
          <div class="filter-bar">
            <label>
              Override email (optional)
              <input id="test-report-email" type="email" placeholder="client@example.com" />
            </label>
            <label>
              Send test report
              <button class="primary-btn" id="send-test-report-btn" type="button">Send test report now</button>
            </label>
            <label>
              Send status
              <div class="status" id="delivery-status"></div>
            </label>
          </div>
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Status</th>
                <th>Email</th>
                <th>Week</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody id="delivery-table">
              <tr><td colspan="5">Loading delivery logs...</td></tr>
            </tbody>
          </table>
        </article>

        <article class="card">
          <div class="copy-row">
            <div>
              <h2>Recent Events</h2>
              <p>Inspect raw routed visits and conversion events, then export CSV for reporting.</p>
            </div>
            <button class="copy-btn" data-copy-target="events-url">Copy JSON URL</button>
          </div>
          <div class="filter-bar">
            <label>
              Start date
              <input id="events-start" type="date" />
            </label>
            <label>
              End date
              <input id="events-end" type="date" />
            </label>
            <label>
              Event type
              <select id="events-type">
                <option value="">All events</option>
                <option value="route">Route</option>
                <option value="outcome">Outcome</option>
              </select>
            </label>
            <label>
              Apply filters
              <button class="primary-btn" id="apply-event-filters" type="button">Refresh events</button>
            </label>
          </div>
          <pre><code id="events-url">Loading events URL...</code></pre>
          <div class="copy-row">
            <strong>CSV Export</strong>
            <a id="events-csv-link" class="copy-btn" href="#" target="_blank" rel="noreferrer">Open CSV export</a>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Type</th>
                  <th>Variant</th>
                  <th>Session</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody id="events-table">
                <tr><td colspan="5">Loading recent events...</td></tr>
              </tbody>
            </table>
          </div>
        </article>
      </section>
    </main>

    <script>
      const siteId = {site_id!r};
      const accessToken = new URLSearchParams(window.location.search).get("token") || "";

      function withToken(path) {{
        if (!accessToken) return path;
        const separator = path.includes("?") ? "&" : "?";
        return `${{path}}${{separator}}token=${{encodeURIComponent(accessToken)}}`;
      }}

      function readEventFilters() {{
        return {{
          start: document.getElementById("events-start").value,
          end: document.getElementById("events-end").value,
          type: document.getElementById("events-type").value,
        }};
      }}

      function buildQuery(params) {{
        const query = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {{
          if (value !== undefined && value !== null && value !== "") {{
            query.set(key, value);
          }}
        }});
        return query.toString();
      }}

      function attachCopyHandlers() {{
        document.querySelectorAll("[data-copy-target]").forEach((button) => {{
          button.addEventListener("click", async () => {{
            const target = document.getElementById(button.dataset.copyTarget);
            if (!target) return;
            const text = target.textContent || "";
            try {{
              await navigator.clipboard.writeText(text);
              const original = button.textContent;
              button.textContent = "Copied";
              setTimeout(() => {{
                button.textContent = original;
              }}, 1200);
            }} catch (_error) {{
              button.textContent = "Copy failed";
            }}
          }});
        }});
      }}

      async function runIntegrationTest() {{
        const button = document.getElementById("test-integration-btn");
        const status = document.getElementById("test-status");
        button.disabled = true;
        status.textContent = "Running sample visit...";

        try {{
          const routeResponse = await fetch(`/route/${{siteId}}`, {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify({{
              visitor_id: `dashboard-test-${{Date.now()}}`,
              device_type: "desktop",
              utm_source: "dashboard-test",
            }}),
          }});

          if (!routeResponse.ok) {{
            throw new Error("Failed to route sample visit.");
          }}

          const routeData = await routeResponse.json();
          status.textContent = "Recording sample conversion...";

          const outcomeResponse = await fetch("/outcome", {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify({{
              site_id: routeData.site_id,
              page_id: routeData.page_id,
              session_id: routeData.session_id,
              converted: true,
              revenue: 0,
            }}),
          }});

          if (!outcomeResponse.ok) {{
            throw new Error("Failed to record sample conversion.");
          }}

          await loadDashboard();
          status.textContent = `Recorded test conversion for ${{routeData.page_id}}.`;
        }} catch (error) {{
          status.textContent = error.message;
        }} finally {{
          button.disabled = false;
        }}
      }}

      async function sendTestReport() {{
        const button = document.getElementById("send-test-report-btn");
        const status = document.getElementById("delivery-status");
        const overrideEmail = document.getElementById("test-report-email").value.trim();
        const query = buildQuery(overrideEmail ? {{ email: overrideEmail }} : {{}});
        const url = withToken(`/reports/${{siteId}}/weekly-summary/send-test${{query ? `?${{query}}` : \"\"}}`);
        button.disabled = true;
        status.textContent = "Sending test report...";
        try {{
          const response = await fetch(url, {{ method: "POST" }});
          const payload = await response.json().catch(() => ({{}}));
          if (!response.ok) {{
            throw new Error(payload.detail || "Failed to send test report.");
          }}
          const result = payload.result || {{}};
          if (result.status === "sent") {{
            status.textContent = `Sent to ${{result.report_email}} for ${{result.week_id}}.`;
          }} else if (result.status === "failed") {{
            status.textContent = `Send failed: ${{result.error || "unknown error"}}`;
          }} else {{
            status.textContent = result.reason || "No report sent.";
          }}
          await loadDashboard();
        }} catch (error) {{
          status.textContent = error.message;
        }} finally {{
          button.disabled = false;
        }}
      }}

      async function loadDashboard() {{
        const filters = readEventFilters();
        const eventsQuery = buildQuery({{ limit: 12, ...filters }});
        const reportQuery = buildQuery({{ days: 7 }});
        const [configRes, statsRes, eventsRes, reportRes, deliveriesRes] = await Promise.all([
          fetch(withToken(`/sites/${{siteId}}`)),
          fetch(withToken(`/stats/${{siteId}}`)),
          fetch(withToken(`/events/${{siteId}}?${{eventsQuery}}`)),
          fetch(withToken(`/reports/${{siteId}}/daily?${{reportQuery}}`)),
          fetch(withToken(`/reports/${{siteId}}/deliveries?limit=10`)),
        ]);

        if (!configRes.ok || !statsRes.ok || !eventsRes.ok || !reportRes.ok || !deliveriesRes.ok) {{
          document.getElementById("test-status").textContent =
            accessToken ? "Failed to load dashboard data." : "Management token missing or invalid.";
          document.getElementById("variants-table").innerHTML =
            '<tr><td colspan="5">Failed to load dashboard data.</td></tr>';
          document.getElementById("events-table").innerHTML =
            '<tr><td colspan="5">Failed to load recent events.</td></tr>';
          document.getElementById("delivery-table").innerHTML =
            '<tr><td colspan="5">Failed to load delivery logs.</td></tr>';
          return;
        }}

        const config = await configRes.json();
        const stats = await statsRes.json();
        const eventsPayload = await eventsRes.json();
        const reportPayload = await reportRes.json();
        const deliveriesPayload = await deliveriesRes.json();
        const statsByPage = Object.fromEntries((stats.arms || []).map((arm) => [arm.page_id, arm]));

        document.getElementById("site-name").textContent = config.site_name || siteId;
        document.getElementById("regime").textContent = stats.regime || "-";
        document.getElementById("sessions").textContent = stats.total_sessions || 0;

        const rows = (config.variants || []).map((variant) => {{
          const arm = statsByPage[variant.page_id] || {{}};
          return `
            <tr>
              <td>${{variant.label}}</td>
              <td><code>${{variant.url}}</code></td>
              <td>${{arm.sessions ?? 0}}</td>
              <td>${{arm.conversions ?? 0}}</td>
              <td>${{arm.rate ?? "0.00%"}}</td>
            </tr>
          `;
        }});
        document.getElementById("variants-table").innerHTML = rows.join("") || '<tr><td colspan="5">No variants configured.</td></tr>';

        const origin = window.location.origin;
        const dashboardUrl = config.dashboard_url || withToken(`${{origin}}/dashboard/${{siteId}}`);
        const routeUrl = `${{origin}}/r/${{siteId}}`;
        const conversionUrl = `${{origin}}/convert/${{siteId}}/<aar_session_id>?converted=true`;
        const eventsUrl = withToken(`${{origin}}/events/${{siteId}}?${{buildQuery({{ limit: 100, ...filters }})}}`);
        const eventsCsvUrl = withToken(`${{origin}}/events/${{siteId}}.csv?${{buildQuery({{ limit: 1000, ...filters }})}}`);
        const weeklyReportUrl = withToken(`${{origin}}/reports/${{siteId}}/weekly-summary/html`);

        document.getElementById("dashboard-url").textContent = dashboardUrl;
        document.getElementById("route-url").textContent = routeUrl;
        document.getElementById("conversion-url").textContent = conversionUrl;
        document.getElementById("events-url").textContent = eventsUrl;
        document.getElementById("events-csv-link").href = eventsCsvUrl;
        document.getElementById("weekly-report-link").href = weeklyReportUrl;
        document.getElementById("report-routes").textContent = reportPayload.totals?.routes ?? 0;
        document.getElementById("report-conversions").textContent = reportPayload.totals?.conversions ?? 0;
        document.getElementById("report-rate").textContent = reportPayload.totals?.conversion_rate ?? "0.00%";
        document.getElementById("report-revenue").textContent = (reportPayload.totals?.revenue ?? 0).toFixed(2);
        document.getElementById("frontend-snippet").textContent =
`<script>
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get("aar_session_id");
  if (sessionId) {{
    fetch("${{origin}}/convert/${{siteId}}/" + sessionId + "?converted=true");
  }}
</script>`;
        document.getElementById("backend-snippet").textContent =
`POST ${{origin}}/outcome
content-type: application/json

{{
  "site_id": "${{siteId}}",
  "page_id": "<aar_page_id>",
  "session_id": "<aar_session_id>",
  "converted": true,
  "revenue": 0
}}`;

        const eventRows = (eventsPayload.events || []).map((event) => {{
          const details = event.event_type === "route"
            ? `visitor=${{event.visitor_id || "-"}} | regime=${{event.regime || "-"}}`
            : `converted=${{event.converted ? "true" : "false"}} | winner=${{event.winner_page_id || "-"}}`;
          return `
            <tr>
              <td><code>${{event.timestamp || "-"}}</code></td>
              <td>${{event.event_type || "-"}}</td>
              <td><code>${{event.page_id || "-"}}</code></td>
              <td><code>${{event.session_id || "-"}}</code></td>
              <td>${{details}}</td>
            </tr>
          `;
        }});
        document.getElementById("events-table").innerHTML =
          eventRows.join("") || '<tr><td colspan="5">No events recorded yet.</td></tr>';

        const deliveryRows = (deliveriesPayload.deliveries || []).map((entry) => {{
          const detail = entry.status === "failed" ? (entry.error || "Send failed") : "Delivered";
          return `
            <tr>
              <td><code>${{entry.timestamp || "-"}}</code></td>
              <td>${{entry.status || "-"}}</td>
              <td><code>${{entry.report_email || "-"}}</code></td>
              <td><code>${{entry.week_id || "-"}}</code></td>
              <td>${{detail}}</td>
            </tr>
          `;
        }});
        document.getElementById("delivery-table").innerHTML =
          deliveryRows.join("") || '<tr><td colspan="5">No delivery logs yet.</td></tr>';
      }}

      attachCopyHandlers();
      document.getElementById("test-integration-btn").addEventListener("click", runIntegrationTest);
      document.getElementById("apply-event-filters").addEventListener("click", loadDashboard);
      document.getElementById("send-test-report-btn").addEventListener("click", sendTestReport);
      loadDashboard();
    </script>
  </body>
</html>"""


def render_weekly_report(site_config: dict, report: dict, report_url: str, dashboard_url: str) -> str:
    summary = report["summary"]
    top_variant = summary.get("top_variant")
    recent_wins = report.get("recent_wins", [])
    recent_losses = report.get("recent_losses", [])
    trend_rows = "".join(
        f"""
          <tr>
            <td>{day['date']}</td>
            <td>{day['routes']}</td>
            <td>{day['conversions']}</td>
            <td>{day['conversion_rate']}</td>
            <td>{day['revenue']:.2f}</td>
          </tr>
        """
        for day in report.get("daily", [])
    ) or '<tr><td colspan="5">No activity in this period.</td></tr>'
    win_rows = "".join(
        f"<li><code>{event['timestamp']}</code> {event['page_label']} converted session <code>{event['session_id']}</code>.</li>"
        for event in recent_wins
    ) or "<li>No conversion wins recorded this week.</li>"
    loss_rows = "".join(
        f"<li><code>{event['timestamp']}</code> {event['page_label']} did not convert for session <code>{event['session_id']}</code>.</li>"
        for event in recent_losses
    ) or "<li>No non-converting outcomes recorded this week.</li>"
    top_variant_html = (
        f"""
        <div class="callout">
          <strong>Top Variant:</strong> {top_variant['label']}<br />
          Routes: {top_variant['routes']}<br />
          Conversions: {top_variant['conversions']}<br />
          Conversion rate: {top_variant['conversion_rate']}<br />
          Revenue: {top_variant['revenue']:.2f}
        </div>
        """
        if top_variant else
        '<div class="callout"><strong>Top Variant:</strong> Not enough conversion data yet.</div>'
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{site_config['site_name']} Weekly Report</title>
    <style>
      :root {{
        --bg: #f4efe4;
        --card: #fffdf8;
        --ink: #19212a;
        --muted: #5a6270;
        --line: #d7d0c3;
        --accent: #0f766e;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "SF Pro Display", "Segoe UI", sans-serif;
        color: var(--ink);
        background: linear-gradient(180deg, #f8f4ea 0%, var(--bg) 100%);
      }}
      main {{
        width: min(980px, 92vw);
        margin: 0 auto;
        padding: 32px 0 48px;
      }}
      .hero, .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 16px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
      }}
      .metric {{
        font-size: 2rem;
        font-weight: 700;
        margin: 6px 0 0;
      }}
      .label {{
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
      }}
      .callout {{
        margin-top: 16px;
        padding: 14px;
        border-radius: 14px;
        background: #edf7f5;
        border: 1px solid #c8ebe6;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        padding: 12px 10px;
        border-bottom: 1px solid var(--line);
      }}
      ul {{
        margin: 0;
        padding-left: 20px;
      }}
      a {{
        color: var(--accent);
        text-decoration: none;
      }}
      .links {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-top: 12px;
      }}
      @media print {{
        body {{ background: #fff; }}
        .hero, .card {{ box-shadow: none; border-color: #ddd; }}
      }}
      @media (max-width: 860px) {{
        .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div class="label">Weekly Client Report</div>
        <h1>{site_config['site_name']}</h1>
        <p>Reporting window: {report['filters']['start']} through {report['filters']['end']}</p>
        <div class="links">
          <a href="{report_url}">JSON summary</a>
          <a href="{dashboard_url}">Back to dashboard</a>
        </div>
      </section>

      <section class="grid">
        <article class="card">
          <div class="label">Routes</div>
          <p class="metric">{summary['routes']}</p>
        </article>
        <article class="card">
          <div class="label">Conversions</div>
          <p class="metric">{summary['conversions']}</p>
        </article>
        <article class="card">
          <div class="label">Conversion Rate</div>
          <p class="metric">{summary['conversion_rate']}</p>
        </article>
        <article class="card">
          <div class="label">Revenue</div>
          <p class="metric">{summary['revenue']:.2f}</p>
        </article>
      </section>

      <section class="card">
        <h2>Top Variant</h2>
        {top_variant_html}
      </section>

      <section class="card">
        <h2>Daily Trend</h2>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Routes</th>
              <th>Conversions</th>
              <th>CVR</th>
              <th>Revenue</th>
            </tr>
          </thead>
          <tbody>{trend_rows}</tbody>
        </table>
      </section>

      <section class="card">
        <h2>Recent Wins</h2>
        <ul>{win_rows}</ul>
      </section>

      <section class="card">
        <h2>Recent Losses</h2>
        <ul>{loss_rows}</ul>
      </section>
    </main>
  </body>
</html>"""
