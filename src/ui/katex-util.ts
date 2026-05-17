// Walk a DOM subtree replacing `$…$` (inline) and `$$…$$` (block) text nodes
// with KaTeX-rendered spans. Avoids pulling in `katex/contrib/auto-render` —
// our templates only use plain `$…$` / `$$…$$`, no edge-case delimiters.

import katex from 'katex';

export function renderMath(root: HTMLElement): void {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
  const targets: Text[] = [];
  let n: Node | null;
  while ((n = walker.nextNode())) targets.push(n as Text);

  for (const text of targets) {
    const t = text.nodeValue ?? '';
    if (!t.includes('$')) continue;
    const frag = document.createDocumentFragment();
    let i = 0;
    while (i < t.length) {
      if (t[i] === '$' && t[i + 1] === '$') {
        const end = t.indexOf('$$', i + 2);
        if (end === -1) { frag.appendChild(document.createTextNode(t.slice(i))); break; }
        const formula = t.slice(i + 2, end);
        const span = document.createElement('span');
        try { katex.render(formula, span, { displayMode: true, throwOnError: false }); }
        catch { span.textContent = formula; }
        frag.appendChild(span);
        i = end + 2;
        continue;
      }
      if (t[i] === '$') {
        const end = t.indexOf('$', i + 1);
        if (end === -1) { frag.appendChild(document.createTextNode(t.slice(i))); break; }
        const formula = t.slice(i + 1, end);
        const span = document.createElement('span');
        try { katex.render(formula, span, { displayMode: false, throwOnError: false }); }
        catch { span.textContent = formula; }
        frag.appendChild(span);
        i = end + 1;
        continue;
      }
      const next$ = t.indexOf('$', i);
      const end = next$ === -1 ? t.length : next$;
      frag.appendChild(document.createTextNode(t.slice(i, end)));
      i = end;
    }
    text.parentNode?.replaceChild(frag, text);
  }
}
