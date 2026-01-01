// embed.js (sağlamlaştırılmış)
(function(){
  const script = document.currentScript;

  function normBaseUrl(u) {
    if (!u) return "";
    u = String(u).trim();

    // Bazı durumlar: "null", "undefined"
    if (!u || u === "null" || u === "undefined") return "";

    // Relative ise absolute yap
    if (u.startsWith("/")) {
      u = window.location.origin + u;
    }

    // Sondaki slash temizle
    u = u.replace(/\/+$/, "");
    return u;
  }

  function pickApiBase() {
    // Öncelik: data-api > (opsiyonel) query ?api= > origin
    const dataApi = script ? script.getAttribute("data-api") : "";
    const qsApi = new URLSearchParams(window.location.search).get("api");
    return normBaseUrl(dataApi) || normBaseUrl(qsApi) || window.location.origin;
  }

  function pickWidgetUrl() {
    // data-widget relative/absolute olabilir
    const widgetPath = (script && script.getAttribute("data-widget")) || "/widget.html";
    // URL() ile normalize edelim
    return new URL(widgetPath, window.location.origin);
  }

  const CFG = {
    api:    pickApiBase(),
    key:    (script && script.getAttribute('data-key')) || '', // X-Internal-Key
    theme:  (script && script.getAttribute('data-theme')) || 'light',
    title:  (script && script.getAttribute('data-title')) || 'SAILY',
    v:      (script && script.getAttribute('data-v')) || ''   // opsiyonel cache buster
  };

  // container
  const root = document.createElement('div');
  root.id = 'saily-root';
  root.style.cssText = `
    position:fixed; right:20px; bottom:20px; z-index:2147483000; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
  `;
  document.body.appendChild(root);

  // launcher button
  const btn = document.createElement('button');
  btn.id = 'saily-launcher';
  btn.innerHTML = '💬';
  btn.title = 'Sohbet';
  btn.style.cssText = `
    width:56px; height:56px; border-radius:50%; border:none; cursor:pointer;
    box-shadow:0 6px 20px rgba(0,0,0,.2); font-size:24px;
    background:#2E86AB; color:#fff;
  `;
  root.appendChild(btn);

  // iframe wrapper (hidden)
  const wrap = document.createElement('div');
  wrap.id = 'saily-wrap';
  wrap.style.cssText = `
    position:fixed; right:20px; bottom:90px; width:360px; height:520px;
    max-width:calc(100vw - 40px); max-height:calc(100vh - 120px);
    display:none; box-shadow:0 12px 32px rgba(0,0,0,.25); border-radius:14px; overflow:hidden;
  `;
  root.appendChild(wrap);

  // iframe
  const iframe = document.createElement('iframe');

  const widgetUrl = pickWidgetUrl();
  widgetUrl.searchParams.set("api", CFG.api);
  if (CFG.key)   widgetUrl.searchParams.set("key", CFG.key);
  if (CFG.theme) widgetUrl.searchParams.set("theme", CFG.theme);
  if (CFG.title) widgetUrl.searchParams.set("title", CFG.title);
  if (CFG.v)     widgetUrl.searchParams.set("v", CFG.v); // cache buster

  iframe.src = widgetUrl.toString();
  iframe.allow = 'clipboard-write;'; // isterseniz microphone vb eklenebilir
  iframe.style.cssText = 'width:100%; height:100%; border:0; background:#fff;';
  wrap.appendChild(iframe);

  // toggle
  btn.addEventListener('click', ()=> {
    wrap.style.display = (wrap.style.display === 'none' || !wrap.style.display) ? 'block' : 'none';
  });

  // dışarıdan kapatma mesajını dinle
  window.addEventListener('message', (e) => {
    if(!e.data || typeof e.data !== 'object') return;
    if(e.data.type === 'closeWidget') {
      wrap.style.display = 'none';
    }
  });
})();
