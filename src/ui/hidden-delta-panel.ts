// Hidden δ panel — for a clicked hidden neuron h[c], shows the responsibility
// flow backward: 10 per-output contributions w[h_c → y_k] · δ[2,k], their sum,
// then the multiply by σ'(z[1,c]) to land on δ[1,c].

import { renderMath } from './katex-util';
import type { SampleSnapshot, Phase as PhaseT } from '../types';
import { Phase } from '../types';

export interface HiddenDeltaState {
  focusedHidden: number | null;
  snapshot: SampleSnapshot | null;
  phase: PhaseT;
}

export class HiddenDeltaPanel {
  readonly element: HTMLElement;
  private titleEl: HTMLElement;
  private hintEl: HTMLElement;
  private bodyEl: HTMLElement;

  constructor() {
    this.element = document.createElement('div');
    this.element.className = 'panel hidden-delta-panel';

    this.titleEl = document.createElement('div');
    this.titleEl.className = 'panel-header';
    this.titleEl.style.color = 'var(--text-dim)';
    this.titleEl.textContent = 'Hidden δ — click a cell in a[1]';
    this.element.appendChild(this.titleEl);

    this.hintEl = document.createElement('div');
    this.hintEl.className = 'panel-hint';
    this.hintEl.textContent =
      'Click a cell in the a[1] row (hidden activations) to inspect one hidden neuron — see how it aggregates blame from all 10 output neurons.';
    this.element.appendChild(this.hintEl);

    this.bodyEl = document.createElement('div');
    this.bodyEl.className = 'hidden-delta-body';
    this.element.appendChild(this.bodyEl);
  }

  render(state: HiddenDeltaState): void {
    if (state.focusedHidden == null) {
      this.titleEl.textContent = 'Hidden δ — click a cell in a[1]';
      this.hintEl.style.display = '';
      this.bodyEl.innerHTML = '';
      return;
    }
    const c = state.focusedHidden;
    const snap = state.snapshot;

    this.titleEl.textContent = `Hidden δ — h[${c}]  (one hidden neuron's backward responsibility)`;
    this.hintEl.style.display = 'none';

    if (!snap) {
      this.bodyEl.innerHTML = '<div class="muted">(start training to populate values)</div>';
      return;
    }

    if (state.phase < Phase.HiddenDelta) {
      this.bodyEl.innerHTML = '<div class="muted">Available at Phase 5 — advance the phase to see this.</div>';
      return;
    }

    // Per-output contributions w[h_c → y_k] · δ[2,k]. Sort by |contribution| so
    // the dominant terms appear at the top of the list.
    const K = snap.outputDelta.length;          // = 10
    const hiddenSize = snap.hiddenActivation.length; // = 30 (and W[2] is 10×30)
    const contribs: { k: number; w: number; d: number; prod: number }[] = [];
    let sumPre = 0;
    for (let k = 0; k < K; k++) {
      const w = snap.weightsW2[k * hiddenSize + c];
      const d = snap.outputDelta[k];
      const prod = w * d;
      sumPre += prod;
      contribs.push({ k, w, d, prod });
    }
    contribs.sort((a, b) => Math.abs(b.prod) - Math.abs(a.prod));

    const zHidden = snap.hiddenWeightedInput[c];
    const sp = snap.hiddenSigmoidPrime[c];
    const deltaH = snap.hiddenDelta[c];

    const sections: string[] = [];

    sections.push(`<div class="section-header">Each output sends its δ back via $w_{h_{${c}} \\to y_k}$ — top contributors by $|w \\cdot \\delta|$:</div>`);
    sections.push('<div class="contrib-list">');
    for (const t of contribs) {
      sections.push(
        `<div class="contrib-line">$w_{h_{${c}} \\to y_{${t.k}}} \\cdot \\delta_{2,${t.k}}
         \\;=\\; (${fmt(t.w)}) \\cdot (${fmt(t.d)})
         \\;=\\; ${fmt(t.prod)}$</div>`,
      );
    }
    sections.push('</div>');

    sections.push(
      `<div class="math-block">$$\\sum_{k}\\; w_{h_{${c}} \\to y_k} \\cdot \\delta_{2,k}
       \\;=\\; ${fmt(sumPre)}$$</div>`,
    );
    sections.push('<div class="panel-hint">That sum is the hidden neuron\'s incoming error before the sigmoid slope is applied.</div>');

    sections.push(`<div class="section-header">Multiply by local sensitivity at $h_{${c}}$:</div>`);
    sections.push(
      `<div class="math-block">$$\\sigma'(z_{1,${c}}) = \\sigma(${fmt(zHidden)}) \\cdot (1 - \\sigma(${fmt(zHidden)}))
       \\;=\\; ${fmt(sp)}$$</div>`,
    );

    sections.push(
      `<div class="math-block highlight">$$\\delta_{1,${c}} \\;=\\; ${parens(fmt(sumPre))} \\cdot ${parens(fmt(sp))}
       \\;=\\; \\mathbf{${fmt(deltaH)}}$$</div>`,
    );

    this.bodyEl.innerHTML = sections.join('');
    renderMath(this.bodyEl);
  }
}

function fmt(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10)  return v.toFixed(1);
  return v.toFixed(4);
}
function parens(s: string): string { return s.startsWith('-') ? `(${s})` : `(+${s})`; }
