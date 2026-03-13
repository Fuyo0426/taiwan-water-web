/**
 * Page slide transition system
 * Intercepts bottom-nav clicks and animates pages in/out
 * based on tab order (left ← / right →)
 */
(function () {
  const PAGE_ORDER = [
    'index.html',
    'schedule.html',
    'cost.html',
    'pump.html',
    'history.html',
    'physics.html',
    'pareto.html',
  ];

  function pageIndex(url) {
    if (!url) return -1;
    const file = url.split('/').pop().split('?')[0].split('#')[0];
    const idx = PAGE_ORDER.indexOf(file);
    // treat bare '/' or empty as index.html
    if (idx === -1 && (file === '' || file === '/')) return 0;
    return idx;
  }

  const currentIndex = pageIndex(window.location.pathname);

  /* ── Apply enter animation ── */
  const ref = document.referrer;
  if (ref) {
    const fromIdx = pageIndex(ref);
    if (fromIdx !== -1 && fromIdx !== currentIndex) {
      const enterFrom = fromIdx < currentIndex ? 'right' : 'left';
      document.documentElement.setAttribute('data-page-enter', enterFrom);
    }
  } else {
    // first load — fade in only
    document.documentElement.setAttribute('data-page-enter', 'fade');
  }

  /* ── Intercept nav clicks ── */
  document.addEventListener('click', function (e) {
    const link = e.target.closest('a[href]');
    if (!link) return;

    const href = link.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('http') || href.startsWith('mailto')) return;

    const targetIdx = pageIndex(href);
    if (targetIdx === -1 || targetIdx === currentIndex) return;

    e.preventDefault();

    const direction = targetIdx > currentIndex ? 'left' : 'right';
    document.documentElement.setAttribute('data-page-exit', direction);

    setTimeout(function () {
      window.location.href = href;
    }, 60);
  });
})();
