// Focus panel — renders the chain-rule derivation for one clicked weight
// w[h_c → y_r] via KaTeX. Replaces the WPF version's hand-stacked ASCII-art
// fractions with real mathematical typography.

import { renderMath } from './katex-util';
import type { SampleSnapshot } from '../types';

export interface FocusState {
  focused: { row: number; col: number } | null;
  snapshot: SampleSnapshot | null;
  learningRate: number;
  miniBatchSize: number;
}

export class FocusPanel {
  readonly element: HTMLElement;
  private titleEl: HTMLElement;
  private hintEl: HTMLElement;
  private bodyEl: HTMLElement;

  constructor() {
    this.element = document.createElement('div');
    this.element.className = 'panel focus-panel';

    this.titleEl = document.createElement('div');
    this.titleEl.className = 'panel-header';
    this.titleEl.style.color = 'var(--gold)';
    this.titleEl.textContent = 'Focus: click a weight';
    this.element.appendChild(this.titleEl);

    this.hintEl = document.createElement('div');
    this.hintEl.className = 'panel-hint';
    this.hintEl.textContent =
      'Click a W[2] cell, a gradient-table cell, or a hidden → output line in the diagram.';
    this.element.appendChild(this.hintEl);

    this.bodyEl = document.createElement('div');
    this.bodyEl.className = 'focus-body';
    this.element.appendChild(this.bodyEl);
  }

  render(state: FocusState): void {
    if (!state.focused) {
      this.titleEl.textContent = 'Focus: click a weight';
      this.hintEl.style.display = '';
      this.bodyEl.innerHTML = '';
      return;
    }
    const { row: r, col: c } = state.focused;
    const snap = state.snapshot;

    // Use the same arrow notation as the WPF version — direction is explicit.
    this.titleEl.textContent = `Focus: w[h${c} → y${r}]   (hidden neuron h${c} → output neuron y${r})`;
    this.hintEl.style.display = 'none';

    if (!snap) {
      this.bodyEl.innerHTML = '<div class="muted">(start training to populate values)</div>';
      return;
    }

    const wVal = snap.weightsW2[r * snap.hiddenActivation.length + c];
    const err = snap.error[r];
    const sp = snap.outputSigmoidPrime[r];
    const delta = snap.outputDelta[r];
    const a1 = snap.hiddenActivation[c];
    const grad = delta * a1;

    // Cost C summed over the 10 outputs — the scalar SGD is descending on.
    let cTotal = 0;
    for (let k = 0; k < snap.outputActivation.length; k++) {
      const d = snap.outputActivation[k] - snap.target[k];
      cTotal += d * d;
    }
    cTotal *= 0.5;

    const sigmoidAtZ = 1 / (1 + Math.exp(-snap.outputWeightedInput[r]));

    // With batch size fixed at 1 in this teaching view, the accumulated value is
    // just this sample's gradient for the update.
    let updateGradient = grad;
    if (snap.accumNablaW2) {
      updateGradient = snap.accumNablaW2[r * snap.hiddenActivation.length + c];
    }
    const eta = state.learningRate;
    const newW = wVal - eta * updateGradient;

    // Build the document.
    const wRef = `w_{h_{${c}} \\to y_{${r}}}`;
    const aRef = `a_{2,${r}}`;
    const zRef = `z_{2,${r}}`;
    const yRef = `y_{${r}}`;
    const deltaRef = `\\delta_{2,${r}}`;
    const a1Ref = `a_{1,${c}}`;
    const dwRef = `\\Delta w_{h_{${c}} \\to y_{${r}}}`;

    const sections: string[] = [];

    sections.push(`<div class="kv"><span>Current value:</span> $${wRef} = ${fmt(wVal)}$</div>`);

    sections.push(`<div class="section-header">Chain rule for this sample:</div>`);
    sections.push(
      `<div class="math-block">$$\\frac{\\partial C}{\\partial ${wRef}}
       = \\frac{\\partial C}{\\partial ${aRef}}
       \\cdot \\frac{\\partial ${aRef}}{\\partial ${zRef}}
       \\cdot \\frac{\\partial ${zRef}}{\\partial ${wRef}}$$</div>`,
    );
    sections.push(
      `<div class="math-block">$$= (${aRef} - ${yRef}) \\cdot \\sigma'(${zRef}) \\cdot ${a1Ref}$$</div>`,
    );
    sections.push(
      `<div class="math-block">$$= ${parens(fmt(err))} \\cdot ${parens(fmt(sp))} \\cdot ${parens(fmt(a1))}$$</div>`,
    );
    sections.push(
      `<div class="math-block highlight">$$= ${deltaRef} \\cdot ${a1Ref} = ${parens(fmt(delta))} \\cdot ${parens(fmt(a1))} = \\mathbf{${fmt(grad)}}$$</div>`,
    );

    sections.push(`<div class="section-header">C (quadratic cost) and its derivative:</div>`);
    sections.push(
      `<div class="math-block">$$C = \\tfrac12 \\sum_k (a_{2,k} - y_k)^2 \\quad\\Rightarrow\\quad C = ${fmt(cTotal)}$$</div>`,
    );
    sections.push(
      `<div class="math-block">$$\\frac{\\partial C}{\\partial ${aRef}} = ${aRef} - ${yRef}
       \\;=\\; ${parens(fmt(snap.outputActivation[r]))} - ${parens(fmt(snap.target[r]))}
       \\;=\\; ${fmt(err)}$$</div>`,
    );

    sections.push(`<div class="section-header">σ (sigmoid) and σ′:</div>`);
    sections.push(
      `<div class="math-block">$$\\sigma(z) = \\frac{1}{1+e^{-z}}, \\quad \\sigma'(z) = \\sigma(z)(1-\\sigma(z))$$</div>`,
    );
    sections.push(
      `<div class="math-block">$$\\sigma(${fmt(snap.outputWeightedInput[r])}) = ${fmt(sigmoidAtZ)}, \\quad
       \\sigma'(${fmt(snap.outputWeightedInput[r])}) = ${fmt(sp)}$$</div>`,
    );

    sections.push(`<div class="section-header">This sample's update (batch size = 1):</div>`);
    sections.push(
      `<div class="math-block">$$${dwRef} \\;=\\; \\frac{\\partial C}{\\partial ${wRef}}
       \\;=\\; ${fmt(updateGradient)}$$</div>`,
    );

    sections.push(`<div class="section-header">After this sample is applied:</div>`);
    sections.push(
      `<div class="math-block green">$$${wRef} \\;\\leftarrow\\; ${wRef} - \\eta \\cdot ${dwRef}$$</div>`,
    );
    sections.push(
      `<div class="math-block green">$$\\;=\\; ${fmt(wVal)} - ${eta.toFixed(1)} \\cdot ${parens(fmt(updateGradient))}
       \\;=\\; ${fmt(newW)}
       \\quad (\\Delta = ${fmt(newW - wVal)})$$</div>`,
    );

    this.bodyEl.innerHTML = sections.join('');
    // KaTeX auto-render: walk the body and replace $…$ / $$…$$ inline.
    renderMath(this.bodyEl);
  }
}

// ── helpers ────────────────────────────────────────────────────────────────

function fmt(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10)  return v.toFixed(1);
  return v.toFixed(4);
}
function parens(s: string): string { return s.startsWith('-') ? `(${s})` : `(+${s})`; }
