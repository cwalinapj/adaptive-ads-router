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
      </section>
    </main>

    <script>
      const siteId = {site_id!r};

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

      async function loadDashboard() {{
        const [configRes, statsRes] = await Promise.all([
          fetch(`/sites/${{siteId}}`),
          fetch(`/stats/${{siteId}}`),
        ]);

        if (!configRes.ok || !statsRes.ok) {{
          document.getElementById("variants-table").innerHTML =
            '<tr><td colspan="5">Failed to load dashboard data.</td></tr>';
          return;
        }}

        const config = await configRes.json();
        const stats = await statsRes.json();
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
        const routeUrl = `${{origin}}/r/${{siteId}}`;
        const conversionUrl = `${{origin}}/convert/${{siteId}}/<aar_session_id>?converted=true`;

        document.getElementById("route-url").textContent = routeUrl;
        document.getElementById("conversion-url").textContent = conversionUrl;
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
      }}

      attachCopyHandlers();
      document.getElementById("test-integration-btn").addEventListener("click", runIntegrationTest);
      loadDashboard();
    </script>
  </body>
</html>"""
