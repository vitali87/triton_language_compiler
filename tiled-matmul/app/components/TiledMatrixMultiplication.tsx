"use client";

import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';

const TiledMatrixMultiplication = () => {
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);
  
  // Matrix dimensions (using small dimensions for clarity)
  const M = 6;
  const N = 6;
  const K = 6;
  const blockM = M / 3;
  const blockN = N / 3;
  const blockK = K / 3;

  // Initialize matrices with simple values
  const matrixA = Array.from({ length: M }, (_, i) => 
    Array.from({ length: K }, (_, j) => (i + j) % 3 + 1)
  );

  const matrixB = Array.from({ length: K }, (_, i) => 
    Array.from({ length: N }, (_, j) => (i * j) % 3 + 1)
  );

  const [outputMatrix, setOutputMatrix] = useState(
    Array.from({ length: M }, () => Array.from({ length: N }, () => 0))
  );

  // Generate steps for visualization
  const generateSteps = () => {
    const steps = [];
    for (let startM = 0; startM < M; startM += blockM) {
      for (let startN = 0; startN < N; startN += blockN) {
        for (let startK = 0; startK < K; startK += blockK) {
          steps.push({
            m: startM,
            n: startN,
            k: startK,
          });
        }
      }
    }
    return steps;
  };

  const steps = generateSteps();

  // Compute tile multiplication result
  const computeTileResult = (startM, startN, startK) => {
    const result = Array.from({ length: blockM }, () => 
      Array.from({ length: blockN }, () => 0)
    );

    for (let i = 0; i < blockM; i++) {
      for (let j = 0; j < blockN; j++) {
        let sum = 0;
        for (let k = 0; k < blockK; k++) {
          sum += matrixA[startM + i][startK + k] * matrixB[startK + k][startN + j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  };

  useEffect(() => {
    let interval;
    if (playing) {
      interval = setInterval(() => {
        setStep((prevStep) => {
          if (prevStep >= steps.length - 1) {
            setPlaying(false);
            return prevStep;
          }
          return prevStep + 1;
        });
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [playing, steps.length]);

  useEffect(() => {
    const currentStep = steps[step];
    if (!currentStep) return;

    const { m, n, k } = currentStep;
    const tileResult = computeTileResult(m, n, k);
    
    setOutputMatrix(prevOutput => {
      const newOutput = [...prevOutput.map(row => [...row])];
      for (let i = 0; i < blockM; i++) {
        for (let j = 0; j < blockN; j++) {
          if (k === 0) {
            newOutput[m + i][n + j] = tileResult[i][j];
          } else {
            newOutput[m + i][n + j] += tileResult[i][j];
          }
        }
      }
      return newOutput;
    });
  }, [step]);

  const currentStep = steps[step] || { m: 0, n: 0, k: 0 };

  const renderCell = (matrix, i, j, value, highlight) => {
    const isHighlighted = highlight(i, j);
    return (
      <div
        key={`${i}-${j}`}
        className={`w-12 h-12 border border-gray-600 flex items-center justify-center text-sm
          ${isHighlighted ? 'bg-blue-800 text-white font-bold' : 'bg-gray-900 text-gray-200'}`}
      >
        {value !== undefined ? value.toFixed(0) : ''}
      </div>
    );
  };

  const renderMatrix = (label, matrix, rows, cols, highlightFn) => {
    return (
      <div className="flex flex-col items-center m-4">
        <div className="text-lg font-bold mb-2 text-gray-200">{label}</div>
        <div className="border-2 border-gray-600 p-2 bg-gray-900 rounded-lg">
          {Array.from({ length: rows }).map((_, i) => (
            <div key={i} className="flex">
              {Array.from({ length: cols }).map((_, j) => 
                renderCell(label, i, j, matrix[i][j], highlightFn)
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const highlightMatrixA = (i, j) => {
    return (
      i >= currentStep.m &&
      i < currentStep.m + blockM &&
      j >= currentStep.k &&
      j < currentStep.k + blockK
    );
  };

  const highlightMatrixB = (i, j) => {
    return (
      i >= currentStep.k &&
      i < currentStep.k + blockK &&
      j >= currentStep.n &&
      j < currentStep.n + blockN
    );
  };

  const highlightOutput = (i, j) => {
    return (
      i >= currentStep.m &&
      i < currentStep.m + blockM &&
      j >= currentStep.n &&
      j < currentStep.n + blockN
    );
  };

  return (
    <div className="w-full max-w-5xl p-6 bg-gray-800 rounded-lg shadow-xl">
      <div className="flex justify-center mb-4">
        <button
          className="mx-2 p-2 border border-gray-600 rounded hover:bg-gray-700 text-gray-200"
          onClick={() => setPlaying(!playing)}
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button
          className="mx-2 p-2 border border-gray-600 rounded hover:bg-gray-700 text-gray-200"
          onClick={() => {
            setStep(0);
            setPlaying(false);
            setOutputMatrix(Array.from({ length: M }, () => 
              Array.from({ length: N }, () => 0)
            ));
          }}
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
      
      <div className="text-center mb-4 text-gray-200">
        Step {step + 1} of {steps.length}
      </div>

      <div className="flex flex-wrap justify-center items-center">
        {renderMatrix('Matrix A', matrixA, M, K, highlightMatrixA)}
        <div className="text-2xl mx-4 text-gray-200">×</div>
        {renderMatrix('Matrix B', matrixB, K, N, highlightMatrixB)}
        <div className="text-2xl mx-4 text-gray-200">=</div>
        {renderMatrix('Output', outputMatrix, M, N, highlightOutput)}
      </div>

      <div className="mt-4 text-center text-sm text-gray-400">
        Computing tile: Block ({Math.floor(currentStep.m/blockM)}, {Math.floor(currentStep.n/blockN)}) 
        with K-block {Math.floor(currentStep.k/blockK)}
      </div>
    </div>
  );
};

export default TiledMatrixMultiplication;