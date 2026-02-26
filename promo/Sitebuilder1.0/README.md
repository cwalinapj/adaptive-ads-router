# Landing Page Quick Deploy

This folder contains a static lead-gen page for Adaptive Ads Router.

## Publish with GitHub Pages

1. Push this repository.
2. In GitHub repo settings, open **Pages**.
3. Set source to `main` and folder to `/promo/Sitebuilder1.0`.
4. Save and wait for deployment URL.

## Capture Leads

Do not commit personal email addresses directly in HTML.

Default behavior in `index.html` uses a safe placeholder action:

```html
action="https://formsubmit.co/replace-with-sales@yourdomain.com"
```

Set the real lead inbox intentionally via env + local config file:

```bash
LEAD_EMAIL="sales@yourdomain.com" ./promo/Sitebuilder1.0/set-lead-email.sh
```

This writes `promo/Sitebuilder1.0/config.js` (gitignored). The page will use
`window.LEAD_EMAIL` to set the FormSubmit endpoint at runtime.
