/* PharmaVision: browser-only inference with ONNX Runtime Web.
   Matching the Kaggle reconstruction: RGB, resize to 224 x 224,
   ImageNet normalization, class index 0 = defective, threshold = 0.40.
   Research prototype. Never use to release pharmaceutical product. */
(() => {
  'use strict';

  const MODEL_URL = '/model/efficientnetv2_s.onnx';
  const ONNX_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  const IMAGE_SIZE = 224;
  const THRESHOLD = 0.40;
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];
  const MAX_FILE_SIZE = 20 * 1024 * 1024;
  const MAX_PREVIEW_SIDE = 1400;
  const acceptedTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);

  const $ = (id) => document.getElementById(id);
  const ui = {
    dropZone: $('dropZone'), canvas: $('previewCanvas'), empty: $('emptyState'),
    camera: $('cameraInput'), file: $('fileInput'), cameraButton: $('cameraButton'),
    uploadButton: $('uploadButton'), imageMeta: $('imageMeta'), cropTools: $('cropTools'),
    resetCrop: $('resetCrop'), analyze: $('analyzeButton'), analyzeText: $('analyzeText'),
    status: $('statusMessage'), indicator: $('modelIndicator'), idle: $('idleResult'),
    result: $('resultContent'), predictionLabel: $('predictionLabel'),
    predictionPill: $('predictionPill'), predictionExplanation: $('predictionExplanation'),
    defectPercent: $('defectPercent'), normalPercent: $('normalPercent'),
    inferenceTime: $('inferenceTime'), probBar: $('probBar'),
    groundTruth: $('groundTruth'), downloadLog: $('downloadLog'),
  };

  const ctx = ui.canvas.getContext('2d', { willReadFrequently: false });
  let sourceImage = null;
  let sourceName = '';
  let selection = null;
  let dragStart = null;
  let session = null;
  let loadingPromise = null;
  let busy = false;
  let records = [];
  let currentRecord = null;

  function status(message, error = false) {
    ui.status.textContent = message;
    ui.status.classList.toggle('is-error', error);
  }

  function modelIndicator(state, message) {
    ui.indicator.classList.toggle('is-ready', state === 'ready');
    ui.indicator.classList.toggle('is-error', state === 'error');
    ui.indicator.lastElementChild.textContent = message;
  }

  function refreshAnalyzeState() {
    ui.analyze.disabled = busy || !sourceImage;
    ui.analyzeText.textContent = busy ? 'Analyzing…' : 'Analyze image';
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function defaultSelection() {
    const w = ui.canvas.width, h = ui.canvas.height;
    return { x: w * 0.1, y: h * 0.1, w: w * 0.8, h: h * 0.8 };
  }

  function drawSelection() {
    if (!sourceImage || !selection) return;
    const w = ui.canvas.width, h = ui.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(sourceImage, 0, 0, w, h);
    const { x, y, w: sw, h: sh } = selection;
    ctx.fillStyle = 'rgba(14, 24, 20, 0.44)';
    ctx.fillRect(0, 0, w, y);
    ctx.fillRect(0, y, x, sh);
    ctx.fillRect(x + sw, y, w - x - sw, sh);
    ctx.fillRect(0, y + sh, w, h - y - sh);
    const stroke = Math.max(2, Math.min(w, h) / 360);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = stroke;
    ctx.strokeRect(x + stroke / 2, y + stroke / 2, Math.max(sw - stroke, 1), Math.max(sh - stroke, 1));
    // Small corner accents make crop boundaries visible on light tablets.
    const corner = Math.min(Math.min(sw, sh) / 4, 30);
    ctx.strokeStyle = '#366c54';
    ctx.lineWidth = stroke * 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y);
    ctx.moveTo(x + sw - corner, y); ctx.lineTo(x + sw, y); ctx.lineTo(x + sw, y + corner);
    ctx.moveTo(x + sw, y + sh - corner); ctx.lineTo(x + sw, y + sh); ctx.lineTo(x + sw - corner, y + sh);
    ctx.moveTo(x + corner, y + sh); ctx.lineTo(x, y + sh); ctx.lineTo(x, y + sh - corner);
    ctx.stroke();
  }

  function mouseToCanvas(event) {
    const rect = ui.canvas.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / rect.width * ui.canvas.width, 0, ui.canvas.width),
      y: clamp((event.clientY - rect.top) / rect.height * ui.canvas.height, 0, ui.canvas.height),
    };
  }

  function selectionFromPoints(a, b) {
    const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
    return { x, y, w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
  }

  ui.canvas.addEventListener('pointerdown', (event) => {
    if (!sourceImage || busy) return;
    event.preventDefault();
    dragStart = mouseToCanvas(event);
    ui.canvas.setPointerCapture(event.pointerId);
  });

  ui.canvas.addEventListener('pointermove', (event) => {
    if (!dragStart || busy) return;
    selection = selectionFromPoints(dragStart, mouseToCanvas(event));
    drawSelection();
  });

  function endDrag(event) {
    if (!dragStart) return;
    const next = selectionFromPoints(dragStart, mouseToCanvas(event));
    const minSize = Math.max(14, Math.min(ui.canvas.width, ui.canvas.height) / 65);
    if (next.w > minSize && next.h > minSize) {
      selection = next;
      clearPreviousResult();
      status('Crop changed. Analyze the image to see a new result.');
    } else if (!selection || selection.w < minSize || selection.h < minSize) {
      selection = defaultSelection();
    }
    dragStart = null;
    drawSelection();
  }
  ui.canvas.addEventListener('pointerup', endDrag);
  ui.canvas.addEventListener('pointercancel', () => { dragStart = null; selection = selection || defaultSelection(); drawSelection(); });

  function clearPreviousResult() {
    ui.result.hidden = true;
    ui.idle.hidden = false;
    ui.groundTruth.disabled = true;
    ui.groundTruth.value = '';
    currentRecord = null;
  }

  async function toDisplayableImage(file) {
    if (typeof createImageBitmap === 'function') {
      try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); }
      catch (_) { /* Safari and some older browsers prefer an HTMLImageElement. */ }
    }
    return await new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
      image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read this image.')); };
      image.src = url;
    });
  }

  async function loadFile(file) {
    if (!file) return;
    if (!acceptedTypes.has(file.type)) {
      status('Please choose a JPG, PNG or WEBP photograph.', true);
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      status('The image is larger than 20 MB. Please use a smaller photograph.', true);
      return;
    }
    try {
      const next = await toDisplayableImage(file);
      if (!next.width || !next.height) throw new Error('The file does not contain a valid image.');
      if (sourceImage && typeof sourceImage.close === 'function') sourceImage.close();
      sourceImage = next;
      sourceName = file.name;
      const scale = Math.min(1, MAX_PREVIEW_SIDE / Math.max(next.width, next.height));
      ui.canvas.width = Math.max(1, Math.round(next.width * scale));
      ui.canvas.height = Math.max(1, Math.round(next.height * scale));
      selection = defaultSelection();
      ui.empty.hidden = true;
      ui.canvas.hidden = false;
      ui.cropTools.hidden = false;
      ui.imageMeta.textContent = `${file.name} · ${next.width} × ${next.height}`;
      drawSelection();
      clearPreviousResult();
      refreshAnalyzeState();
      status('Photograph ready. Adjust the crop around one tablet, then analyze.');
    } catch (error) {
      status(error.message || 'Could not open this image.', true);
    }
  }

  ui.cameraButton.addEventListener('click', () => ui.camera.click());
  ui.uploadButton.addEventListener('click', () => ui.file.click());
  ui.camera.addEventListener('change', (event) => { loadFile(event.target.files?.[0]); event.target.value = ''; });
  ui.file.addEventListener('change', (event) => { loadFile(event.target.files?.[0]); event.target.value = ''; });
  ui.resetCrop.addEventListener('click', () => {
    if (!sourceImage || busy) return;
    selection = defaultSelection();
    drawSelection();
    clearPreviousResult();
    status('Crop reset. Adjust the crop around the tablet before analyzing.');
  });

  // Drag-and-drop is useful on desktops; the same inputs support touch devices.
  ui.dropZone.addEventListener('dragover', (event) => { event.preventDefault(); ui.dropZone.classList.add('drag-over'); });
  ui.dropZone.addEventListener('dragleave', () => ui.dropZone.classList.remove('drag-over'));
  ui.dropZone.addEventListener('drop', (event) => {
    event.preventDefault(); ui.dropZone.classList.remove('drag-over');
    loadFile(event.dataTransfer.files?.[0]);
  });

  function cropAsTensor() {
    if (!sourceImage || !selection || selection.w < 2 || selection.h < 2) {
      throw new Error('Please select a larger crop that includes the tablet.');
    }
    const offscreen = document.createElement('canvas');
    offscreen.width = IMAGE_SIZE;
    offscreen.height = IMAGE_SIZE;
    const offCtx = offscreen.getContext('2d', { willReadFrequently: true });
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = 'high';
    const xScale = sourceImage.width / ui.canvas.width;
    const yScale = sourceImage.height / ui.canvas.height;
    offCtx.drawImage(sourceImage,
      selection.x * xScale, selection.y * yScale,
      selection.w * xScale, selection.h * yScale,
      0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const pixels = offCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
    const planeSize = IMAGE_SIZE * IMAGE_SIZE;
    const tensor = new Float32Array(3 * planeSize);
    for (let i = 0; i < planeSize; i++) {
      const p = i * 4;
      tensor[i] = ((pixels[p] / 255) - MEAN[0]) / STD[0];
      tensor[planeSize + i] = ((pixels[p + 1] / 255) - MEAN[1]) / STD[1];
      tensor[planeSize * 2 + i] = ((pixels[p + 2] / 255) - MEAN[2]) / STD[2];
    }
    return new window.ort.Tensor('float32', tensor, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  }

  async function getModel() {
    if (session) return session;
    if (loadingPromise) return loadingPromise;
    loadingPromise = (async () => {
      if (!window.ort) throw new Error('Model runtime did not load. Check your internet connection and reload.');
      modelIndicator('loading', 'Loading model');
      status('Loading the 80+ MB model. This can take a while on mobile data…');
      window.ort.env.wasm.numThreads = 1; // Reliable without cross-origin isolation on Safari/Chrome.
      window.ort.env.wasm.wasmPaths = ONNX_CDN;
      const response = await fetch(MODEL_URL, { method: 'HEAD', cache: 'no-store' });
      if (!response.ok) throw new Error('Model file not found. Add public/model/efficientnetv2_s.onnx to your repository and redeploy.');
      const newSession = await window.ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      if (!newSession.inputNames.length || !newSession.outputNames.length) {
        throw new Error('The model does not expose the expected input and output.');
      }
      session = newSession;
      modelIndicator('ready', 'Model ready');
      return session;
    })();
    try { return await loadingPromise; }
    catch (error) {
      modelIndicator('error', 'Model unavailable');
      throw error;
    } finally { loadingPromise = null; }
  }

  function probabilityOfDefective(logits) {
    if (!logits || logits.length !== 2) throw new Error('Expected two model logits (defective and normal).');
    const max = Math.max(Number(logits[0]), Number(logits[1]));
    const a = Math.exp(Number(logits[0]) - max), b = Math.exp(Number(logits[1]) - max);
    const p = a / (a + b);
    if (!Number.isFinite(p)) throw new Error('Model returned invalid prediction values.');
    return p;
  }

  function showPrediction(p, elapsed) {
    const defective = p >= THRESHOLD;
    const pred = defective ? 'defective' : 'normal';
    ui.idle.hidden = true;
    ui.result.hidden = false;
    ui.result.classList.toggle('is-defective', defective);
    ui.predictionLabel.textContent = defective ? 'Defective' : 'Normal';
    ui.predictionPill.textContent = 'Visual screening';
    ui.predictionExplanation.textContent = defective
      ? 'Visible abnormality suspected. Refer the tablet for qualified visual review.'
      : 'No visual defect flagged by this model. This does not certify tablet quality.';
    ui.defectPercent.textContent = `${(p * 100).toFixed(1)}%`;
    ui.normalPercent.textContent = `${((1 - p) * 100).toFixed(1)}%`;
    ui.inferenceTime.textContent = `${Math.round(elapsed)} ms`;
    ui.probBar.style.width = `${(p * 100).toFixed(2)}%`;
    ui.groundTruth.value = '';
    ui.groundTruth.disabled = false;

    currentRecord = {
      timestamp: new Date().toISOString(), filename: sourceName, prediction: pred,
      defect_probability: p.toFixed(6), normal_probability: (1 - p).toFixed(6),
      threshold: THRESHOLD, inference_ms: Math.round(elapsed),
      observed_condition: '',
      crop: JSON.stringify({
        x: Math.round(selection.x), y: Math.round(selection.y),
        width: Math.round(selection.w), height: Math.round(selection.h),
        image_width: ui.canvas.width, image_height: ui.canvas.height,
      }),
    };
    records.push(currentRecord);
    ui.downloadLog.disabled = false;
  }

  ui.analyze.addEventListener('click', async () => {
    if (!sourceImage || busy) return;
    busy = true;
    refreshAnalyzeState();
    try {
      const inferenceSession = await getModel();
      status('Running local image analysis…');
      await new Promise((resolve) => setTimeout(resolve, 40));
      const tensor = cropAsTensor();
      const start = performance.now();
      const output = await inferenceSession.run({ [inferenceSession.inputNames[0]]: tensor });
      const elapsed = performance.now() - start;
      const logits = output[inferenceSession.outputNames[0]]?.data;
      const p = probabilityOfDefective(logits);
      showPrediction(p, elapsed);
      status('Analysis complete. Record the observed condition for field evaluation.');
    } catch (error) {
      console.error('Inference failed:', error);
      status(`Unable to analyze: ${error.message || 'Unknown error'}`, true);
    } finally {
      busy = false;
      refreshAnalyzeState();
    }
  });

  ui.groundTruth.addEventListener('change', (event) => {
    if (currentRecord) currentRecord.observed_condition = event.target.value;
  });

  function csvCell(input) {
    let value = String(input ?? '');
    // Prevent spreadsheet formula injection from filenames or free text.
    if (/^[=+@\-\t\r]/.test(value)) value = "'" + value;
    return `"${value.replace(/"/g, '""')}"`;
  }

  ui.downloadLog.addEventListener('click', () => {
    if (!records.length) return;
    const fields = ['timestamp', 'filename', 'prediction', 'defect_probability',
      'normal_probability', 'threshold', 'inference_ms', 'observed_condition', 'crop'];
    const csv = [fields.map(csvCell).join(','),
      ...records.map((item) => fields.map((field) => csvCell(item[field])).join(','))].join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pharmavision-field-log-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  });

  refreshAnalyzeState();
})();
