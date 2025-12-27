/**
 * GTM Binding Script for Page Containers
 *
 * This script connects the page container to the Edge Composer via WebSocket,
 * enabling real-time GTM data layer events to flow to the router for
 * bandit updates and AI accelerator training.
 *
 * Include this in your React page container's public/index.html or main entry.
 */

(function() {
  'use strict';

  // Configuration (injected by container env vars)
  const CONFIG = {
    sessionId: window.__EDGE_SESSION_ID__ || new URLSearchParams(location.search).get('session_id'),
    siteId: window.__EDGE_SITE_ID__ || new URLSearchParams(location.search).get('site_id'),
    containerId: window.__EDGE_CONTAINER_ID__ || 'unknown',
    edgeWsUrl: window.__EDGE_WS_URL__ || 'ws://localhost:8040',
    gtmContainerId: window.__GTM_CONTAINER_ID__ || 'GTM-DEFAULT',
    debug: window.__EDGE_DEBUG__ || false
  };

  // WebSocket connection to Edge Composer
  let ws = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_ATTEMPTS = 5;
  const RECONNECT_DELAY = 2000;

  // Event queue for offline buffering
  const eventQueue = [];

  // =============================================================================
  // WEBSOCKET CONNECTION
  // =============================================================================

  function connect() {
    if (!CONFIG.sessionId) {
      console.warn('[GTM-Binding] No session ID, skipping WebSocket connection');
      return;
    }

    const wsUrl = `${CONFIG.edgeWsUrl}/gtm/${CONFIG.sessionId}`;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = function() {
        if (CONFIG.debug) console.log('[GTM-Binding] Connected to Edge Composer');
        reconnectAttempts = 0;

        // Flush queued events
        while (eventQueue.length > 0) {
          const event = eventQueue.shift();
          sendEvent(event);
        }

        // Send initial page view
        sendEvent({
          event: 'page_view',
          page_path: location.pathname,
          page_title: document.title
        });
      };

      ws.onmessage = function(msg) {
        try {
          const data = JSON.parse(msg.data);
          if (CONFIG.debug) console.log('[GTM-Binding] Received:', data);

          // Handle commands from Edge Composer
          if (data.command === 'advance_funnel') {
            window.dispatchEvent(new CustomEvent('funnel:advance', { detail: data }));
          }
        } catch (e) {
          // Ignore parse errors
        }
      };

      ws.onclose = function() {
        if (CONFIG.debug) console.log('[GTM-Binding] Disconnected');
        attemptReconnect();
      };

      ws.onerror = function(err) {
        console.error('[GTM-Binding] WebSocket error:', err);
      };

    } catch (e) {
      console.error('[GTM-Binding] Failed to connect:', e);
      attemptReconnect();
    }
  }

  function attemptReconnect() {
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      reconnectAttempts++;
      setTimeout(connect, RECONNECT_DELAY * reconnectAttempts);
    }
  }

  function sendEvent(eventData) {
    const payload = {
      ...eventData,
      site_id: CONFIG.siteId,
      container_id: CONFIG.containerId,
      timestamp: new Date().toISOString(),
      url: location.href,
      user_agent: navigator.userAgent
    };

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    } else {
      // Queue for later
      eventQueue.push(payload);
    }
  }

  // =============================================================================
  // GTM DATA LAYER INTEGRATION
  // =============================================================================

  // Initialize dataLayer if not exists
  window.dataLayer = window.dataLayer || [];

  // Store original push
  const originalPush = window.dataLayer.push.bind(window.dataLayer);

  // Override dataLayer.push to intercept events
  window.dataLayer.push = function() {
    // Call original
    const result = originalPush.apply(window.dataLayer, arguments);

    // Forward to Edge Composer
    for (let i = 0; i < arguments.length; i++) {
      const item = arguments[i];
      if (item && typeof item === 'object') {
        sendEvent({
          event: item.event || 'dataLayer_push',
          data: item
        });
      }
    }

    return result;
  };

  // =============================================================================
  // AUTOMATIC EVENT TRACKING
  // =============================================================================

  // Track scroll depth
  let maxScroll = 0;
  let scrollThrottled = false;

  window.addEventListener('scroll', function() {
    if (scrollThrottled) return;
    scrollThrottled = true;

    setTimeout(function() {
      const scrollPercent = Math.round(
        (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
      );

      if (scrollPercent > maxScroll) {
        maxScroll = scrollPercent;

        // Send at 25%, 50%, 75%, 100%
        if ([25, 50, 75, 100].includes(scrollPercent)) {
          sendEvent({
            event: 'scroll_depth',
            scroll_depth: scrollPercent
          });
        }
      }

      scrollThrottled = false;
    }, 100);
  });

  // Track time on page
  const pageStartTime = Date.now();
  const timeIntervals = [10, 30, 60, 120, 300]; // seconds
  let timeIndex = 0;

  setInterval(function() {
    const timeOnPage = (Date.now() - pageStartTime) / 1000;

    if (timeIndex < timeIntervals.length && timeOnPage >= timeIntervals[timeIndex]) {
      sendEvent({
        event: 'time_on_page',
        time_on_page: timeIntervals[timeIndex]
      });
      timeIndex++;
    }
  }, 1000);

  // Track clicks on important elements
  document.addEventListener('click', function(e) {
    const target = e.target.closest('a, button, [data-track]');
    if (!target) return;

    const trackData = {
      event: 'click',
      click_element: target.tagName.toLowerCase(),
      click_id: target.id || null,
      click_class: target.className || null,
      click_text: (target.textContent || '').substring(0, 50)
    };

    // Check if it's the main CTA
    if (target.id === 'main-cta' || target.classList.contains('cta')) {
      trackData.event = 'cta_click';
      trackData.cta_intent_score = 0.8;
    }

    sendEvent(trackData);
  });

  // Track form submissions
  document.addEventListener('submit', function(e) {
    const form = e.target;

    sendEvent({
      event: 'form_submit',
      form_submit: true,
      form_id: form.id || null,
      form_action: form.action || null
    });
  });

  // Track visibility (tab focus)
  document.addEventListener('visibilitychange', function() {
    sendEvent({
      event: 'visibility_change',
      visible: !document.hidden
    });
  });

  // =============================================================================
  // FUNNEL NAVIGATION HELPERS
  // =============================================================================

  window.EdgeFunnel = {
    // Call when user completes current step
    advance: function(stepData) {
      sendEvent({
        event: 'funnel_advance',
        step_data: stepData || {}
      });
    },

    // Call on conversion
    convert: function(conversionData) {
      sendEvent({
        event: 'conversion',
        conversion: true,
        revenue: conversionData.revenue || 0,
        conversion_type: conversionData.type || 'lead'
      });
    },

    // Get current session info
    getSession: function() {
      return {
        sessionId: CONFIG.sessionId,
        siteId: CONFIG.siteId,
        containerId: CONFIG.containerId
      };
    }
  };

  // =============================================================================
  // INITIALIZE
  // =============================================================================

  // Connect when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', connect);
  } else {
    connect();
  }

  // Send beacon on page unload
  window.addEventListener('beforeunload', function() {
    const timeOnPage = (Date.now() - pageStartTime) / 1000;

    // Use sendBeacon for reliability
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        `${CONFIG.edgeWsUrl.replace('ws', 'http')}/beacon`,
        JSON.stringify({
          event: 'page_exit',
          session_id: CONFIG.sessionId,
          site_id: CONFIG.siteId,
          container_id: CONFIG.containerId,
          time_on_page: timeOnPage,
          max_scroll: maxScroll
        })
      );
    }
  });

  if (CONFIG.debug) {
    console.log('[GTM-Binding] Initialized', CONFIG);
  }

})();
