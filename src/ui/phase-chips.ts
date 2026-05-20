// Phase chip strip + accompanying explainer banner. Dispatches 'phase-jump'
// (CustomEvent with detail = phase number) when a chip is clicked.

import { Phase, PhaseCount } from '../types';
import { renderMath } from './katex-util';

const CHIP_LABELS: Record<Phase, string> = {
  [Phase.Forward]: '1. Forward',
  [Phase.Loss]: '2. Loss',
  [Phase.OutputDelta]: '3. Output δ',
  [Phase.Gradient]: '4. Gradient',
  [Phase.HiddenDelta]: '5. Hidden δ',
};

const LOCKED_EXPLAINER =
  'Press Start to load the first MNIST training sample. Once a sample is available, the steps unlock so you can walk through the forward pass, loss, gradients, and backpropagation.';

/**
 * Voice-over text for each phase. Written so a student who only reads the
 * banner can still follow what's happening — what formula, which quantities
 * are now in play, and (crucially for Forward) which are NOT.
 */
const EXPLAINERS: Record<Phase, string> = {
  [Phase.Forward]:
    "Feed a sample $x$ (196-dimensional input vector, each element in the range $[0,1]$ representing the brightness of a pixel). $W_1$ and $W_2$ are matrices representing the network's mappings from input $\\to$ hidden and hidden $\\to$ output, respectively. The elements in $W_1$ and $W_2$ are originally initialized to random numbers; $b_1$ and $b_2$ are randomized bias vectors:\n" +
    "$$\\begin{array}{ll}" +
    "z_1 = W_1x + b_1 & a_1 = \\sigma(z_1) \\\\" +
    "z_2 = W_2a_1 + b_2 & a_2 = \\sigma(z_2)" +
    "\\end{array}\\qquad\\qquad \\sigma(z)=\\dfrac{1}{1+e^{-z}}$$" +
    "The sigmoid function $\\sigma$ turns any $z$ into an activation between $0$ and $1$. " +
    "Output $a_2$ is the network's raw 10-dimensional prediction. Whatever digit $d \\in \\{0,1,2,\\ldots,9\\}$ with the greatest $a_{2,d}$ is the network's guess.",
  [Phase.Loss]:
    "$y$ is the target vector: a 10-dimensional vector with $1$ at the correct digit and $0$ everywhere else.\n" +
    "$$C = \\frac12 \\sum_k (a_{2,k} - y_k)^2$$" +
    "This loss $C$ measures how far the network's output vector $a_2$ is from the target vector $y$. Smaller $C$ means the prediction is closer to the correct answer. We also compute the error vector $a_2-y$, which feeds into Phase 3.",
  [Phase.OutputDelta]:
    "$\\delta_2$ is the output layer's error signal. For each output digit $k$, $\\delta_{2,k}$ asks: how much would the loss $C$ change if $z_{2,k}$ changed?\n" +
    "$$\\delta_{2,k} = \\frac{\\partial C}{\\partial z_{2,k}} = (a_{2,k} - y_k)\\,\\sigma'(z_{2,k})$$" +
    "This result comes from the chain rule: loss changes through the output activation $a_{2,k}$, and $a_{2,k}$ changes through the sigmoid at $z_{2,k}$. $\\frac{\\partial C}{\\partial z_{2,k}}$ means the sensitivity of the loss to $z_{2,k}$. The term $a_{2,k}-y_k$ is the output error from Phase 2, and $\\sigma'(z_{2,k})$ is the sigmoid's local slope at that output.",
  [Phase.Gradient]:
    "Phase 3 gave us $\\delta_2$, which tells us how the loss changes when each output value $z_2$ changes. That matters because the weights and biases are what create $z_2$: $z_{2,r}=\\sum_c w_{h_c\\to y_r}a_{1,c}+b_{2,r}$.\n" +
    "$$\\frac{\\partial C}{\\partial w_{h_c \\to y_r}} = \\delta_{2,r}a_{1,c} \\qquad \\frac{\\partial C}{\\partial b_{2,r}} = \\delta_{2,r}$$" +
    "These partial derivatives are the gradient: they tell us which direction each weight or bias should move to reduce the loss. A larger derivative means changing that parameter would affect the loss more. Here, batch size is $1$, so this sample's gradients become $\\Delta W_2$ and $\\Delta b_2$. The symbol $\\eta$ is the learning rate, or step size. Each parameter is replaced by its old value minus $\\eta$ times its gradient: $w_{\\text{new}} = w_{\\text{old}} - \\eta\\,\\frac{\\partial C}{\\partial w}$. Increasing $\\eta$ makes the weights move farther on each update; decreasing $\\eta$ makes each update more cautious.",
  [Phase.HiddenDelta]:
    "Step 5 is the bridge that lets us repeat the same Phase 3 $\\to$ Phase 4 idea for the first layer. We already used $\\delta_2$ to update $W_2$ and $b_2$; now we push that error backward through $W_2$ to find the hidden layer's error signal $\\delta_1$:\n" +
    "$$\\delta_{1,c} = \\left(\\sum_k w_{h_c \\to y_k}\\delta_{2,k}\\right)\\sigma'(z_{1,c})$$" +
    "Once we have $\\delta_1$, the network can update $W_1$ and $b_1$ in the same way Phase 4 updated $W_2$ and $b_2$. This demo focuses the visible interaction on the final layer, but the first-layer update is happening in the background.",
};

export class PhaseChips {
  readonly element: HTMLElement;
  readonly explainerElement: HTMLElement;
  private chipEls: HTMLElement[] = [];
  private currentPhase: Phase = Phase.Forward;
  private stepMode = true;
  private enabled = false;

  constructor() {
    this.element = document.createElement('div');
    this.element.className = 'phase-strip';

    for (let i = 0; i < PhaseCount; i++) {
      const phase = i as Phase;
      const chip = document.createElement('div');
      chip.className = 'phase-chip';
      chip.textContent = CHIP_LABELS[phase];
      chip.addEventListener('click', () => {
        if (!this.enabled || !this.stepMode) return;
        this.element.dispatchEvent(new CustomEvent('phase-jump', { detail: phase, bubbles: true }));
      });
      this.chipEls.push(chip);
      this.element.appendChild(chip);
    }

    this.explainerElement = document.createElement('div');
    this.explainerElement.className = 'phase-explainer';
    this.refresh();
  }

  setPhase(phase: Phase): void {
    this.currentPhase = phase;
    this.refresh();
  }

  setStepMode(on: boolean): void {
    this.stepMode = on;
    this.refresh();
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    this.refresh();
  }

  private refresh(): void {
    for (let i = 0; i < PhaseCount; i++) {
      const el = this.chipEls[i];
      el.classList.remove('active', 'done', 'pinned', 'disabled');
      el.setAttribute('aria-disabled', `${!this.enabled}`);
      if (!this.enabled) {
        el.classList.add('disabled');
      } else if (!this.stepMode) {
        el.classList.add('pinned');
      } else if (i < this.currentPhase) {
        el.classList.add('done');
      } else if (i === this.currentPhase) {
        el.classList.add('active');
      }
    }
    const explainer = this.enabled ? EXPLAINERS[this.currentPhase] : LOCKED_EXPLAINER;
    this.explainerElement.innerHTML =
      `<div class="phase-detail">${escapeHtml(explainer).replace(/\n/g, '<br>')}</div>`;
    renderMath(this.explainerElement);
  }
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
