
export type LayerWeights = number[][];
export type LayerBiases = number[];

export class NeuralNetwork {
  layers: number[];
  weights: LayerWeights[];
  biases: LayerBiases[];
  activations: number[][];
  zs: number[][];
  learningRate: number;

  constructor(layerSizes: number[], learningRate: number = 0.1) {
    this.layers = layerSizes;
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.zs = [];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const rows = layerSizes[i + 1];
      const cols = layerSizes[i];
      const weightMatrix = Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => Math.random() * 2 - 1)
      );
      this.weights.push(weightMatrix);
      this.biases.push(Array.from({ length: rows }, () => Math.random() * 2 - 1));
    }
  }

  sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDerivative(x: number): number {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }

  feedForward(input: number[]): number[] {
    this.activations = [input];
    this.zs = [];

    let current = input;
    for (let i = 0; i < this.weights.length; i++) {
      const w = this.weights[i];
      const b = this.biases[i];
      const next: number[] = [];
      const layerZs: number[] = [];

      for (let j = 0; j < w.length; j++) {
        let sum = b[j];
        for (let k = 0; k < w[j].length; k++) {
          sum += w[j][k] * current[k];
        }
        layerZs.push(sum);
        next.push(this.sigmoid(sum));
      }
      this.zs.push(layerZs);
      this.activations.push(next);
      current = next;
    }
    return current;
  }

  backpropagate(target: number[]): void {
    const outputLayerIndex = this.activations.length - 1;
    let errors: number[] = [];

    // Output layer error
    const outputActivations = this.activations[outputLayerIndex];
    for (let i = 0; i < outputActivations.length; i++) {
      errors.push(outputActivations[i] - target[i]);
    }

    for (let i = this.weights.length - 1; i >= 0; i--) {
      const layerZs = this.zs[i];
      const prevActivations = this.activations[i];
      const currentWeights = this.weights[i];
      const currentBiases = this.biases[i];

      const gradients = layerZs.map((z, idx) => errors[idx] * this.sigmoidDerivative(z));

      // Update weights and biases
      for (let j = 0; j < currentWeights.length; j++) {
        for (let k = 0; k < currentWeights[j].length; k++) {
          currentWeights[j][k] -= this.learningRate * gradients[j] * prevActivations[k];
        }
        currentBiases[j] -= this.learningRate * gradients[j];
      }

      // Propagate error to previous layer
      if (i > 0) {
        const nextErrors: number[] = new Array(prevActivations.length).fill(0);
        for (let j = 0; j < currentWeights.length; j++) {
          for (let k = 0; k < currentWeights[j].length; k++) {
            nextErrors[k] += currentWeights[j][k] * gradients[j];
          }
        }
        errors = nextErrors;
      }
    }
  }

  train(input: number[], target: number[]): number {
    const output = this.feedForward(input);
    this.backpropagate(target);
    
    // Calculate MSE loss
    let loss = 0;
    for (let i = 0; i < target.length; i++) {
      loss += Math.pow(target[i] - output[i], 2);
    }
    return loss / target.length;
  }

  exportState() {
    return {
      weights: this.weights,
      biases: this.biases
    };
  }

  importState(state: { weights: LayerWeights[], biases: LayerBiases[] }) {
    if (state.weights && state.biases) {
      this.weights = state.weights;
      this.biases = state.biases;
    }
  }

  resetWeights() {
    this.weights = [];
    this.biases = [];
    for (let i = 0; i < this.layers.length - 1; i++) {
      const rows = this.layers[i + 1];
      const cols = this.layers[i];
      const weightMatrix = Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => Math.random() * 2 - 1)
      );
      this.weights.push(weightMatrix);
      this.biases.push(Array.from({ length: rows }, () => Math.random() * 2 - 1));
    }
  }
}
