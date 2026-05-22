export type Matrix = number[][];

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export function identity(size: number): Matrix {
  const out = zeros(size, size);
  for (let i = 0; i < size; i += 1) out[i][i] = 1;
  return out;
}

export function transpose(a: Matrix): Matrix {
  return a[0].map((_, col) => a.map((row) => row[col]));
}

export function multiply(a: Matrix, b: Matrix): Matrix {
  const out = zeros(a.length, b[0].length);
  for (let i = 0; i < a.length; i += 1) {
    for (let k = 0; k < b.length; k += 1) {
      if (a[i][k] === 0) continue;
      for (let j = 0; j < b[0].length; j += 1) {
        out[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return out;
}

export function multiplyVector(a: Matrix, v: number[]): number[] {
  return a.map((row) => row.reduce((sum, value, i) => sum + value * v[i], 0));
}

export function solveLinearSystem(a: Matrix, b: number[]): number[] {
  const n = a.length;
  const m = a.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }

    if (Math.abs(m[pivot][col]) < 1e-10) {
      throw new Error("全局刚度矩阵奇异，请检查约束、机构或零长度单元。");
    }

    [m[col], m[pivot]] = [m[pivot], m[col]];
    const pivotValue = m[col][col];
    for (let j = col; j <= n; j += 1) m[col][j] /= pivotValue;

    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = m[row][col];
      if (factor === 0) continue;
      for (let j = col; j <= n; j += 1) {
        m[row][j] -= factor * m[col][j];
      }
    }
  }

  return m.map((row) => row[n]);
}

export function choleskyFactor(a: Matrix): Matrix {
  const n = a.length;
  const l = zeros(n, n);
  const maxDiagonal = Math.max(1, ...a.map((row, index) => Math.abs(row[index] ?? 0)));
  const tolerance = maxDiagonal * 1e-12;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let sum = 0;
      for (let k = 0; k < j; k += 1) sum += l[i][k] * l[j][k];
      if (i === j) {
        const diagonal = a[i][i] - sum;
        if (diagonal <= tolerance) throw new Error("矩阵不是正定矩阵。");
        l[i][j] = Math.sqrt(diagonal);
      } else {
        l[i][j] = (a[i][j] - sum) / l[j][j];
      }
    }
  }
  return l;
}

export function solveCholeskyFactor(factor: Matrix, b: number[]): number[] {
  const n = factor.length;
  const y = Array(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    let sum = 0;
    for (let k = 0; k < i; k += 1) sum += factor[i][k] * y[k];
    y[i] = (b[i] - sum) / factor[i][i];
  }

  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i -= 1) {
    let sum = 0;
    for (let k = i + 1; k < n; k += 1) sum += factor[k][i] * x[k];
    x[i] = (y[i] - sum) / factor[i][i];
  }
  return x;
}

export function solveSymmetricPositiveDefinite(a: Matrix, b: number[]): number[] {
  return solveCholeskyFactor(choleskyFactor(a), b);
}
