// embed.js
(function(){
  const CFG = {
    api:    (document.currentScript.getAttribute('data-api') || 'https://chatlog.api-saily.com'),
    key:    (document.currentScript.getAttribute('data-key') || ''), // X-Internal-Key
    theme:  (document.currentScript.getAttribute('data-theme') || 'light'),
    title:  (document.currentScript.getAttribute('data-title') || 'SAILY'),
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
  const qs = new URLSearchParams({
    api: CFG.api,
    key: CFG.key,
    theme: CFG.theme,
    title: CFG.title
  }).toString();

  // widget.html dosyanızın tam yolu (aynı sitenizden servis edin)
  iframe.src = (document.currentScript.getAttribute('data-widget') || '/widget.html') + '?' + qs;
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